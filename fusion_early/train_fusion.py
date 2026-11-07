#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pro/3 多模态融合训练 —— 三分支图像编码器(冻结) + 点云编码器 + 导航状态编码器 → CrossModalFusion
================================================================================================
对应主架构: [土建/轨道/设备 图像编码器] + [pcl_enc 点云 token] + [nav_enc 导航] → ④融合(自注+交叉注+门控)
→ 适配 → 分支 Detect 头 → 检测。在【真实配对 sim 数据】(output_*: 图像+4ch属性图+位姿+标签同帧)上训练。

【为什么逐分支】
  pro/3 三支是三类病害子集(土建3/轨道3/设备2)，各用独立 Detect 头。融合时三支图像编码器都冻结、
  特征 concat 后做跨分支 self-attn + 点云 cross-attn + 导航 query 注入，输出融合金字塔；
  适配器把 256→检测头各尺度通道(256/512/512)，接【被测分支】原 Detect(继承)。监督=该分支类子集
  (从 sim 8cls 标签按 sim_cls 映射过滤)。每次测一个分支(--branch)，看"另两支+点云+导航"能否抬升该支。

【真实 loss —— 复用 ultralytics v8DetectionLoss】
  不手搓 loss。被测分支 DetectionModel 作 wrapper: model.model[-1]=原 Detect, model.args=hyp 默认+覆盖。
  前向: encoders(no_grad,冻) → concat → fusion → adapters → Detect → preds(dict) → criterion(preds,batch)。

【数据 : output_* 配对帧】
  output_* 每帧 图像/4ch attrmap/位姿/标签 同 stem 同目录。按分支取匹配的 train_sets, 过滤 0 框帧。
  图像与 attrmap 用【同一 letterbox 几何】(同 scale/offset)，保证图像↔点云空间对齐。
  位姿 json(x,y,z,qx,qy,qz,qw) → [x,y,z,sin(yaw),cos(yaw)] → NavEncoder→(B,256)。

【用法(.venv, torch cu130)】
    python train_fusion.py --branch 轨道 --epochs 60 --imgsz 640 --batch 2 --device 0
  --smoke: 合成 batch 验证 前端→fusion→Detect→loss→backward 全链路 + 显存
  --probe: 同 smoke 但只前向(报显存), 用于在 8GB 4060 上探可行 imgsz/max_tokens

【注意】8GB 4060 跑不动 imgsz 1024 的 3 编码器; 默认 640, 需要再小就 --imgsz 512 / 减 --max-tokens。
"""
from __future__ import annotations

import argparse, json, math, os, random, sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn

_P = os.path.dirname(os.path.abspath(__file__))
if _P not in sys.path:
    sys.path.insert(0, _P)
import det_encoders_runtime as R            # pro/3 三支图像编码器加载
import fusion                                # CrossModalFusion / total_channels_per_scale
import pcl_enc                               # PointCloudEncoder
from fusion import CrossModalFusion, total_channels_per_scale

# ---------------- 分支配置 ----------------
_BRANCHES = {
    "土建": dict(weight="weights/best_土建.pt", nc=3,
                 sim_cls={0: 0, 1: 1, 6: 2},                    # crack→0, leakage→1, spalling→2
                 train_sets=["output_data1_bing", "output_big_s0", "output_big_s1", "output_big_s2"],
                 test_set="output_big_s3"),
    "轨道": dict(weight="weights/best_轨道.pt", nc=3,
                 sim_cls={2: 0, 3: 1, 4: 2},                    # broken→0, missing→1, loose→2
                 train_sets=["output_rail_s0", "output_rail_s1", "output_rail_s2"],
                 test_set="output_rail_s1"),
    "设备": dict(weight="weights/best_设备.pt", nc=2,
                 sim_cls={5: 0, 7: 1},                          # intrusion→0, bracket_loose→1
                 train_sets=["output_data1_bing", "output_big_s0", "output_big_s1", "output_big_s2"],
                 test_set="output_big_s3"),
}
_HEAD_CH = (256, 512, 512)                   # 检测头各尺度输入通道(neck)
_HYP = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}  # ultralytics 检测损失增益
_DEPTH_SCALE, _HEIGHT_SCALE = 30.0, 3.0      # 与 train_pcl_detect 一致
_PCL_WEIGHT = os.path.join(_P, 'runs', 'pcl_pretrain_full', 'pcl_encoder.pt')


class NavEncoder(nn.Module):
    """导航状态编码器: 位姿向量 [x,y,z,sin(yaw),cos(yaw)] → (B, embed_dim)。作为融合条件查询。"""

    def __init__(self, in_dim: int = 5, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, embed_dim))

    def forward(self, nav: torch.Tensor) -> torch.Tensor:      # (B, in_dim) -> (B, embed_dim)
        return self.net(nav)


def yaw_from_pose(pose: dict) -> float:
    qx, qy, qz, qw = pose.get('qx', 0), pose.get('qy', 0), pose.get('qz', 0), pose.get('qw', 1)
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


# ---------------- Wrapper: 编码器(冻)+融合+适配+Detect ----------------
class FusionWrapper(nn.Module):
    def __init__(self, model, encs, pcl_enc, nav_enc, fusion, adapters):
        super().__init__()
        self.model = model                      # 被测分支 DetectionModel(带原 Detect/.args/.class_weights)
        self.encs = encs                        # {支: {mode: enc}} 冻结
        self.pcl_enc = pcl_enc                  # 点云编码器(冻结)
        self.nav_enc = nav_enc                  # 导航编码器(可训, 很轻)
        self.fusion = fusion
        self.adapters = adapters                # ModuleList[Conv2d(256->每尺度头通道)]
        det = self.model.model[-1]
        det.train()
        if hasattr(det, "end2end"):
            det.end2end = False
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import IterableSimpleNamespace
        a = getattr(self.model, "args", None)
        d = dict(vars(get_cfg()))
        if isinstance(a, dict):
            d.update(a)
        elif a is not None:
            d.update(vars(a))
        self.model.args = IterableSimpleNamespace(**d)

    @torch.no_grad()
    def _image_branch_feats(self, img):
        bf = {}
        for b, md in self.encs.items():
            f = md["neck"](img)                  # [P3,P4,P5]
            bf[b] = {"neck": [t for t in f]}
        return bf

    def forward(self, img, attr, nav, batch=None):
        bf = self._image_branch_feats(img)
        # 点云 token(no_grad, 冻结)
        with torch.no_grad():
            pcl_feats = self.pcl_enc(attr)       # list[(B,Np,256)] x3
        nav_embed = self.nav_enc(nav)            # (B,256)  可训
        fused = self.fusion(bf, pcl_feats, nav_embed)     # {P3,P4,P5} @256
        names = ("P3", "P4", "P5")
        feats = [self.adapters[i](fused[n]) for i, n in enumerate(names) if n in fused]
        preds = self.model.model[-1](feats)      # Detect 训练输出 dict(含 'boxes')
        if batch is not None:
            loss, loss_dict = self.criterion(preds, batch)
            return loss.sum(), loss_dict
        return preds

    @property
    def criterion(self):
        if getattr(self, "_crit", None) is None:
            from ultralytics.utils.loss import v8DetectionLoss
            self._crit = v8DetectionLoss(self.model)
        return self._crit


# ---------------- 数据装载(图像+属性图+导航 同帧配对) ----------------
def build_letterbox(img_hw: tuple[int, int], imgsz: int) -> tuple[float, int, int, int, int]:
    """由原图 (H,W) 计算 letterbox 几何: r, nw, nh, dw, dh —— 图像与 attrmap 共用, 保证对齐。"""
    H, W = img_hw
    imgsz = (int(imgsz) + 31) // 32 * 32
    r = min(imgsz / H, imgsz / W)
    nw, nh = round(W * r), round(H * r)
    dw, dh = (imgsz - nw) // 2, (imgsz - nh) // 2
    return r, nw, nh, dw, dh, imgsz


def letterbox_img(img: np.ndarray, g) -> np.ndarray:
    """HWC uint8 RGB -> (3,imgsz,imgsz) float[0,1], 同 attrmap 几何, 灰 114 填充。"""
    r, nw, nh, dw, dh, imgsz = g
    if (nw, nh) != img.shape[1::-1]:
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((imgsz, imgsz, img.shape[2]), 114, np.uint8)
    out[dh:dh + nh, dw:dw + nw] = img
    return torch.from_numpy(np.ascontiguousarray(out).astype(np.float32) / 255.0).permute(2, 0, 1)


def letterbox_attr(a: np.ndarray, g) -> np.ndarray:
    """4通道 attrmap(原始 depth/refl/height/mask 未归一) -> (4,imgsz,imgsz) [0,1], 0 填充。"""
    r, nw, nh, dw, dh, imgsz = g
    out = np.zeros((4, imgsz, imgsz), np.float32)
    for c in range(4):
        ch = a[c]
        if (nw, nh) != ch.shape[1::-1]:
            ch = cv2.resize(ch, (nw, nh), interpolation=cv2.INTER_AREA)
        out[c, dh:dh + nh, dw:dw + nw] = ch
    return torch.from_numpy(np.ascontiguousarray(normalize4(out)))


def normalize4(m: np.ndarray) -> np.ndarray:
    d, r, h, mk = m[0], m[1], m[2], m[3]
    return np.stack([
        np.clip(d / _DEPTH_SCALE, 0, 1),
        np.clip(r, 0, 1),
        np.clip(h / _HEIGHT_SCALE, 0, 1),
        mk,
    ]).astype(np.float32)


def letterbox_boxes(labels_norm, g):
    """YOLO 归一化标签 (N,5)[cls,xc,yc,w,h] -> letterbox 到 imgsz 的 xyxy(N,4)@imgsz + cls。"""
    if labels_norm is None or len(labels_norm) == 0:
        return np.empty((0,), np.float32), np.empty((0, 4), np.float32)
    labels_norm = np.asarray(labels_norm, np.float32)
    r, nw, nh, dw, dh, imgsz = g
    H, W = 720, 1280                          # attrmap/标签 原几何(720×1280), 固定
    cls = labels_norm[:, 0]
    x = labels_norm[:, 1] * W * r + dw
    y = labels_norm[:, 2] * H * r + dh
    w = labels_norm[:, 3] * W * r
    h = labels_norm[:, 4] * H * r
    xyxy = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
    return cls, np.clip(xyxy, 0, imgsz)


def resolve_label_paths(root: str):
    """output_* 内 images/*/labels/* 对齐: images 目录可能嵌套。返回 {(dir,stem)} 帧清单由 attrmaps 定。"""
    d = os.path.join(root, 'attrmaps')
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith('.npz'))


def img_path_for(root: str, stem: str):
    p = os.path.join(root, 'images', f'{stem}.jpg')
    if os.path.isfile(p):
        return p
    return os.path.join(root, 'images', f'{stem}.png')


def load_sim_frame(root: str, stem: str, imgsz: int, sim_cls: dict, nc: int):
    """一帧 -> (img(3,imgsz,imgsz)[0,1], attr(4,imgsz,imgsz)[0,1], nav(5,), cls(N,), xyxy(N,4)@imgsz)。
    标签过滤到分支子集并 remap; 0 框帧由调用方丢弃。"""
    # 图像
    ip = img_path_for(root, stem)
    img = cv2.imread(ip)
    img = img[..., ::-1] if img is not None else None      # BGR->RGB
    if img is None:
        return None
    g = build_letterbox(img.shape[:2], imgsz)
    # attrmap (4ch, 原始值)
    a = np.load(os.path.join(root, 'attrmaps', f'{stem}.npz'))
    attr = np.stack([a['depth'], a['reflec'], a['height'], a['mask']]).astype(np.float32)
    # 位姿 -> nav
    pose = {}
    pp = os.path.join(root, 'poses', f'{stem}.json')
    if os.path.isfile(pp):
        pose = json.load(open(pp))
    yaw = yaw_from_pose(pose)
    nav = np.array([pose.get('x', 0), pose.get('y', 0), pose.get('z', 0),
                    math.sin(yaw), math.cos(yaw)], np.float32)
    # 标签: 过滤+remap
    lbl = os.path.join(root, 'labels', f'{stem}.txt')
    if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
        labels = np.loadtxt(lbl, dtype=np.float32).reshape(-1, 5)
    else:
        labels = np.empty((0, 5))
    keep = [i for i, l in enumerate(labels) if int(l[0]) in sim_cls]
    labels = labels[keep] if keep else np.empty((0, 5))
    if labels.shape[0]:
        labels = labels.copy(); labels[:, 0] = [sim_cls[int(c)] for c in labels[:, 0]]
    cls, xyxy = letterbox_boxes(labels, g)
    ximg_, attr_ = letterbox_img(img, g), letterbox_attr(attr, g)
    return ximg_, attr_, nav, cls, xyxy


def collate_fusion(samples, imgsz: int):
    """list[load_sim_frame 非None] -> (img, attr, nav, batch)。只保留含目标框的帧;
    全为背景帧返回 None(避免纯背景 batch 把 cls 损失打爆, 与 pcl 预训一致)。"""
    frames = [s for s in samples if s[4].size]            # s[4]=xyxy, size>0 才有目标
    if not frames:
        return None
    n = len(frames)
    img = torch.stack([s[0] for s in frames])              # (B,3,imgsz,imgsz)
    attr = torch.stack([s[1] for s in frames])             # (B,4,imgsz,imgsz)
    nav = torch.stack([s[2] if isinstance(s[2], torch.Tensor) else torch.from_numpy(s[2])
                       for s in frames])                   # (B,5)
    cls_list, xyxy_list, bidx = [], [], []
    for bi, s in enumerate(frames):
        cls_list.append(s[3]); xyxy_list.append(s[4])
        bidx.append(np.full(s[4].shape[0], bi, np.float32))
    cls = np.concatenate(cls_list)
    xyxy = np.concatenate(xyxy_list)
    bidx = np.concatenate(bidx)
    xc = (xyxy[:, 0] + xyxy[:, 2]) / 2; yc = (xyxy[:, 1] + xyxy[:, 3]) / 2
    w = xyxy[:, 2] - xyxy[:, 0]; h = xyxy[:, 3] - xyxy[:, 1]
    xywh = np.stack([xc, yc, w, h], 1) / imgsz                        # 归一化 [0,1]
    batch = {'cls': torch.from_numpy(cls), 'bboxes': torch.from_numpy(xywh),
             'batch_idx': torch.from_numpy(bidx)}
    return img, attr, nav, batch


# ---------------- 构建模型 ----------------
def _freeze(m):
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def build_model(branch, img_imgsz=640, max_tokens=4096, device='cpu', pcl_weight=None):
    from ultralytics import YOLO
    conf = _BRANCHES[branch]
    model = YOLO(os.path.join(_P, conf['weight'])).model
    model.eval()
    for layer in model.model[0:23]:
        _freeze(layer)
    # 该 checkpoint 载入时全参数 requires_grad=False(推理态) —— 实测 evidence。
    # 融合要吃新特征分布, Detect 头必须可训, 这里显式解冻。
    det = model.model[-1]
    for p in det.parameters():
        p.requires_grad_(True)
    encs = R.load_all_encoders()                     # {支: {neck: enc, backbone: enc}}
    for b, md in encs.items():
        for mode, enc in md.items():
            _freeze(enc)
    if pcl_weight is None:
        pcl_weight = _PCL_WEIGHT
    if os.path.isfile(pcl_weight):
        penc = torch.load(pcl_weight, map_location='cpu', weights_only=False).eval()
    else:
        print(f"[warn] 无点云预训权重 {pcl_weight}, 用随机 pcl_enc")
        penc = pcl_enc.PointCloudEncoder(in_channels=4, embed_dim=256, out_stages=3).eval()
    _freeze(penc)
    fusion = CrossModalFusion(total_channels_per_scale('neck'), embed_dim=256, heads=4,
                              max_tokens=max_tokens)
    adapters = nn.ModuleList([nn.Conv2d(256, c, 1) for c in _HEAD_CH])
    wrapper = FusionWrapper(model, encs, penc, NavEncoder(), fusion, adapters)
    wrapper.to(device)
    for md in wrapper.encs.values():                  # 编码器是普通 dict, nn.Module.to 不递归, 手动搬
        for enc in md.values():
            enc.to(device)
    penc.to(device)
    return wrapper, conf['nc']


# ---------------- smoke / probe ----------------
def smoke(branch='轨道', imgsz=640, max_tokens=4096, device='cpu', probe_only=False):
    torch.manual_seed(0)
    nc = _BRANCHES[branch]['nc']
    wrapper, nc = build_model(branch, img_imgsz=imgsz, max_tokens=max_tokens, device=device)
    for p in wrapper.parameters():
        if p.requires_grad:
            p.to(device)
    wrapper.train(); wrapper.model.model[-1].train()
    if probe_only:
        torch.set_grad_enabled(False)
    img = torch.rand(2, 3, imgsz, imgsz, device=device)
    attr = torch.rand(2, 4, imgsz, imgsz, device=device)
    nav = torch.rand(2, 5, device=device)
    preds = wrapper(img, attr, nav)
    print(f"  forward: preds type={type(preds).__name__}", end='')
    if isinstance(preds, dict):
        print(f" keys={list(preds.keys())} boxes={tuple(preds['boxes'].shape)}")
    else:
        raise SystemExit('Detect 未走训练分支(dict)')
    if probe_only:
        print(f"  [probe] 显存: "
              f"{torch.cuda.memory_allocated()/1e9:.2f}GB 峰值 "
              f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        print('PROBE_OK')
        return
    batch = {'cls': torch.randint(0, nc, (6, 1), device=device),
             'bboxes': torch.rand(6, 4, device=device),          # xywh 归一化 [0,1] 与 v8DetectionLoss 一致
             'batch_idx': torch.tensor([0, 0, 0, 1, 1, 1], device=device)}
    loss, ld = wrapper.criterion(preds, batch)
    ls = float(torch.as_tensor(loss).detach().sum())
    ldd = {k: float(v.detach().item()) for k, v in ld.items()}
    print(f"  loss={ls:.4f}  loss_detach={ldd}")

    def _fin(p):
        return p.grad is not None and float(p.grad.abs().sum())
    g = [_fin(p) for p in wrapper.fusion.parameters() if p.requires_grad]
    gn = [_fin(p) for p in wrapper.nav_enc.parameters() if p.requires_grad]
    ga = [_fin(p) for p in wrapper.adapters.parameters() if p.requires_grad]
    gd = [_fin(p) for p in wrapper.model.model[-1].parameters() if p.requires_grad]
    print(f"  backward OK → 带梯度: fusion={sum(x is not None for x in g)}/{len(g)} "
          f"nav={sum(x is not None for x in gn)}/{len(gn)} "
          f"adapter={sum(x is not None for x in ga)}/{len(ga)} "
          f"detect={sum(x is not None for x in gd)}/{len(gd)}")
    print('FUSION_SMOKE_OK')


# ---------------- 真实训练(sim 配对数据) ----------------
def main():
    ap = argparse.ArgumentParser(description='pro/3 多模态融合训练(逐分支增益)')
    ap.add_argument('--branch', default='轨道', choices=list(_BRANCHES))
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=os.path.join(_P, 'runs', 'fusion'))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--pcl-weight', default=None, help='点云编码器权重(缺省=全量预训 pcl_encoder.pt)')
    ap.add_argument('--data-roots', default=None, help='覆盖 train_sets(逗号分隔 pcdata 目录), 缺省用分支默认')
    ap.add_argument('--save', default=None, help='保存 wrapper state_dict 路径')
    ap.add_argument('--seed-eval-frames', type=int, default=0, help='>0 则每 epoch 在 train 集抽 N 帧做 val loss')
    args = ap.parse_args()

    device = 'cuda' if args.device != 'cpu' else 'cpu'
    if args.device != 'cpu':
        device = torch.device('cuda:' + args.device if args.device.isdigit() else args.device)

    if args.probe:
        smoke(branch=args.branch, imgsz=args.imgsz, max_tokens=args.max_tokens, device=device, probe_only=True)
        return
    if args.smoke:
        smoke(branch=args.branch, imgsz=args.imgsz, max_tokens=args.max_tokens, device=device)
        return

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    nc = _BRANCHES[args.branch]['nc']; sim_cls = _BRANCHES[args.branch]['sim_cls']
    roots = (args.data_roots.split(',') if args.data_roots
             else [os.path.join(_P, 'pcdata', x) for x in _BRANCHES[args.branch]['train_sets']])
    print(f"\n[数据] 分支={args.branch} nc={nc}  roots={len(roots)} 台")
    frames = []                                    # [(root, stem, nbox)]
    for root in roots:
        for stem in resolve_label_paths(root):
            lbl = os.path.join(root, 'labels', f'{stem}.txt')
            if not os.path.isfile(lbl) or os.path.getsize(lbl) == 0:
                continue
            try:
                lab = np.loadtxt(lbl, dtype=np.float32).reshape(-1, 5)
            except Exception:
                continue
            nb = sum(1 for l in lab if int(l[0]) in sim_cls)
            if nb:
                frames.append((root, stem))
    assert frames, f"[数据] {args.branch} 在 roots 里找不到含目标类的帧"
    print(f"[数据] {len(frames)} 帧(含目标框)")

    wrapper, nc = build_model(args.branch, img_imgsz=args.imgsz, max_tokens=args.max_tokens,
                              device=device, pcl_weight=args.pcl_weight)
    for p in wrapper.parameters():
        if p.requires_grad:
            p.to(device)
    for md in wrapper.encs.values():
        for enc in md.values():
            enc.to(device)
    wrapper.pcl_enc.to(device)
    wrapper.train(); wrapper.model.model[-1].train()

    det = wrapper.model.model[-1]
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    n_t = sum(p.numel() for p in trainable) / 1e6
    n_f = sum(p.numel() for p in wrapper.parameters()) / 1e6 - n_t
    print(f"[参数] 冻结(编码器)={n_f:.1f}M  训练(fusion+nav+adapter+detect)={n_t:.1f}M")
    opt = torch.optim.AdamW([
        {'params': [p for p in wrapper.fusion.parameters() if p.requires_grad], 'lr': args.lr},
        {'params': [p for p in wrapper.nav_enc.parameters() if p.requires_grad], 'lr': args.lr},
        {'params': [p for p in wrapper.adapters.parameters() if p.requires_grad], 'lr': args.lr},
        {'params': [p for p in det.parameters() if p.requires_grad], 'lr': args.lr * 0.1},
    ], weight_decay=1e-4)
    epochs = args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: max(0.0, 0.5 * (1.0 + math.cos(math.pi * ep / epochs))))
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f'fusion_{args.branch}.log')
    flog = open(log_path, 'w'); flog.write('epoch,loss\n')

    def run_step(idxs, grad):
        samples = [load_sim_frame(r, s, args.imgsz, sim_cls, nc) for (r, s) in idxs]
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        img, attr, nav, batch = collate_fusion(samples, args.imgsz)
        if img is None:                     # 全为背景帧
            return None
        img = img.to(device); attr = attr.to(device); nav = nav.to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        if grad:
            loss, ld = wrapper(img, attr, nav, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            return float(loss.detach())
        with torch.no_grad():
            loss, ld = wrapper(img, attr, nav, batch)
            return float(loss.detach())

    for ep in range(1, epochs + 1):
        random.shuffle(frames)
        tot, cnt = 0.0, 0
        for bi in range(0, len(frames), args.batch):
            ls = run_step(frames[bi:bi + args.batch], True)
            if ls is None:
                continue
            tot += ls; cnt += 1
        sched.step()
        avg = tot / max(1, cnt)
        flog.write(f"{ep},{avg:.4f}\n"); flog.flush()
        if ep % 10 == 0 or ep == 1:
            print(f"[ep{ep:02d}] train_loss={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}", flush=True)
    flog.close()
    if args.save:
        torch.save(wrapper.state_dict(), args.save)
        print(f"[save] {args.save}")
    print("TRAIN_FUSION_DONE")


if __name__ == '__main__':
    main()
