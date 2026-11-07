#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云编码器检测式预训 —— pro/3 (主项目版, 自 inspection_robot/fusion/train_assist.py 裁剪)
==========================================================================================
目标: 用【自有 4 通道属性图 + 缺陷框】检测式预训 PointCloudEncoder(pcl_enc), 让点云编码器
学出"能检出缺陷"的特征, 再摘编码器给 pro/3 融合层用 —— 与图像支(检测训练→摘编码器)对齐。

与 inspection_robot 的 PointCloudAssistedYOLO 不同: 本脚本【纯点云、无图像编码器/无位姿/无融合】,
只走 pcl_enc → Detect head, 即"点云独资检测器"。检测头接法/损失/数据管线照搬已跑通的那套:
    _LossContainer(detect, SimpleNamespace(**HYP)) 包成假模型壳 → v8DetectionLoss(复用 TAL+DFL)。

【架构】4通道 attrmap(深度/反射/高度/遮罩, letterbox 到 640×640, [0,1])
  → pcl.forward_maps() → [P3',P4',P5'] = (B,256,80/40/20,80/40/20)  stride 8/16/32
  → Detect(nc, reg_max=16, end2end=False, ch=(256,256,256), stride=(8,16,32))
  → v8DetectionLoss (假设 batch 字段 cls/bboxes(xywh@640)/batch_idx)

用法(用 .venv —— torch+cu130+CUDA):
    .venv 路径: /home/ylf/workspace/inspection_robot/.venv/bin/python
    python train_pcl_detect.py                                   # 默认 output_data1_bing
    python train_pcl_detect.py --data pcdata/output_big_s0 --quick   # 2 epoch 冒烟
    python train_pcl_detect.py --data pcdata/output_data1_bing --epochs 80 --batch 8

产物:
    {out}/last.pt               —— 全 state_dict(含 BN 统计), 供载回
    {out}/pcl_encoder.pt        —— 摘出的点云编码器(整模块, 载入需 import pcl_enc)
"""

import argparse
import json
import os
import random
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

_P = os.path.dirname(os.path.abspath(__file__))
if _P not in sys.path:
    sys.path.insert(0, _P)
import pcl_enc                         # noqa: E402  (本文件同目录)
from pcl_enc import PointCloudEncoder  # noqa: E402

# ---------------- 冻结常量(与 inspection_robot/fusion/config.py 对齐) ----------------
HYP = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}      # ultralytics 检测损失增益
REG_MAX = 16
DETECT_STRIDE = (8.0, 16.0, 32.0)
YOLO_SIZE = 640                              # letterbox 目标(32 倍数, stride 32)
DEPTH_SCALE = 30.0                           # depth∈~[1,28]m -> /30 夹 [0,1]
HEIGHT_OFFSET = 0.0
HEIGHT_SCALE = 3.0
NECK_CH = (256, 256, 256)                    # pcl.forward_maps 三尺度均为 embed_dim=256
# nc 从所抓数据集的 data.yaml 读


class _LossContainer(nn.Module):
    """v8DetectionLoss 需要 'model.args' + 'model.model[-1]' 的假模型壳(照搬已跑通)。"""

    def __init__(self, detect, args):
        super().__init__()
        self.model = nn.ModuleList([nn.Identity(), detect])
        self.args = args
        self.class_weights = None


class PointCloudDetector(nn.Module):
    """点云独资检测器: pcl.forward_maps → Detect head → v8DetectionLoss。"""

    def __init__(self, nc=8, device=""):
        super().__init__()
        self.nc = nc
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pcl = PointCloudEncoder(in_channels=4, embed_dim=256, out_stages=3)
        from ultralytics.nn.modules.head import Detect
        self.detect = Detect(nc=nc, reg_max=REG_MAX, end2end=False, ch=NECK_CH)
        self.detect.stride = torch.tensor(DETECT_STRIDE)
        # Detect.__init__ 的 bias_init() 按 nc 设强负 cls 先验(sigmoid≈0), 冻结主干+从零训无 warmup
        # 必塌缩成"0 检测"; 零初始化 cls 末层, 让头能学(照搬 train_assist.neutralize_cls_head)。
        self._neutralize_cls_head()
        self.to(self.device)
        # 必须在 .to(device) 之后创建, 否则 criterion 内部模块留在创建时刻 device
        from ultralytics.utils.loss import v8DetectionLoss
        self.criterion = v8DetectionLoss(_LossContainer(self.detect, SimpleNamespace(**HYP)))

    def _neutralize_cls_head(self):
        """零初始化 Detect cls 末层 1×1 的 weight/bias, 逃出 bias_init 的 -9 饱和区。"""
        with torch.no_grad():
            for i in range(len(self.detect.one2many['cls_head'])):
                cls_conv = self.detect.one2many['cls_head'][i][-1]   # 末层 1×1 分类 conv
                if hasattr(cls_conv, 'weight'):
                    cls_conv.weight.data.zero_()
                if hasattr(cls_conv, 'bias'):
                    cls_conv.bias.data.zero_()

    def forward(self, attr4, batch=None):
        """attr4: (B,4,640,640) [0,1]。给 batch 返回 (loss.sum(), loss_dict)。"""
        maps = self.pcl.forward_maps(attr4)      # [ (B,256,80,80),(B,256,40,40),(B,256,20,20) ]
        preds = self.detect(maps)
        if batch is not None:
            loss, loss_dict = self.criterion(preds, batch)
            return loss.sum(), loss_dict
        return preds


# ================================================================
# 数据装载 (4 通道 attrmap + label, 照搬 train_assist 的 letterbox/collate 规范)
# ================================================================
def normalize4(m: np.ndarray) -> np.ndarray:
    """(4,H,W) 原始 attrmap(depth/reflec/height/mask) -> [0,1]。mask 原样(0/1)。"""
    d, r, h, mk = m[0], m[1], m[2], m[3]
    return np.stack([
        np.clip(d / DEPTH_SCALE, 0.0, 1.0),
        np.clip(r, 0.0, 1.0),
        np.clip((h - HEIGHT_OFFSET) / HEIGHT_SCALE, 0.0, 1.0),
        mk,
    ]).astype(np.float32)


def letterbox_4ch(m: np.ndarray) -> np.ndarray:
    """(4,H,W) [0,1] -> letterbox 到 YOLO_SIZE×YOLO_SIZE, 灰 0 填充, 不缩放变形。"""
    H, W = m.shape[-2], m.shape[-1]
    r = min(YOLO_SIZE / H, YOLO_SIZE / W)
    nw, nh = round(W * r), round(H * r)
    dw, dh = (YOLO_SIZE - nw) // 2, (YOLO_SIZE - nh) // 2
    out = np.zeros((4, YOLO_SIZE, YOLO_SIZE), np.float32)
    for c in range(4):
        ch = m[c]
        if (nw, nh) != (W, H):
            ch = cv2.resize(ch, (nw, nh), interpolation=cv2.INTER_AREA)
        out[c, dh:dh + nh, dw:dw + nw] = ch
    return out


def letterbox_boxes(labels_norm, img_hw, new_size=YOLO_SIZE):
    """YOLO 归一化标签 (N,5)[cls,xc,yc,w,h] -> letterbox 后图像像素 xyxy@new_size(N,4)。"""
    if labels_norm is None or len(labels_norm) == 0:
        return np.empty((0,), np.float32), np.empty((0, 4), np.float32)
    labels_norm = np.asarray(labels_norm, np.float32)
    H, W = img_hw
    r = min(new_size / H, new_size / W)
    nw, nh = round(W * r), round(H * r)
    dw, dh = (new_size - nw) / 2.0, (new_size - nh) / 2.0
    cls = labels_norm[:, 0]
    x = labels_norm[:, 1] * W * r + dw
    y = labels_norm[:, 2] * H * r + dh
    w = labels_norm[:, 3] * W * r
    h = labels_norm[:, 4] * H * r
    xyxy = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
    xyxy = np.clip(xyxy, 0.0, new_size)
    return cls, xyxy


# 全量 8cls 集(统一 8 类 taxonomy, 覆盖三域 + 真实配对 pcl/导航): 用于"全量预训"。
_FULL_8CLS = [
    'output_data1_bing', 'output_big_s0', 'output_big_s1', 'output_big_s2',
    'output_big_s3', 'output_rail_s0', 'output_rail_s1',
]


def resolve_roots(arg: str | None) -> list[str]:
    """解析 --data 为 pcdata 数据集目录列表: 逗号分隔 或 glob 或 'all'(用全部 8cls 集)。"""
    if not arg or arg == 'all':
        return [os.path.join(_P, 'pcdata', x) for x in _FULL_8CLS]
    import glob as _g
    bases = []
    for chunk in arg.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        cand = os.path.join(_P, 'pcdata', chunk) if not os.path.isabs(chunk) else chunk
        has = _g.glob(cand)
        bases.extend(has if has else [cand])
    return bases


def list_samples(roots: list[str]):
    """以 attrmaps/*.npz 为帧清单, 聚合多集 -> 排序好的 [(root, stem), ...]。跳过无 attrmaps 的集。"""
    out = []
    for root in roots:
        d = os.path.join(root, 'attrmaps')
        if not os.path.isdir(d):
            print(f"[skip] {root} 无 attrmaps/, 跳过")
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.npz'):
                out.append((root, os.path.splitext(f)[0]))
    return out


def load_attr_sample(stem: str, root: str, cache: dict | None = None):
    """读一帧 4ch attrmap + label -> (attrmap(4,640,640)[0,1], cls(N,), xyxy(N,4)@640)。
    cache 提供时零磁盘读(缓存存 uint8, 这里 /255 还原); 否则从 npz 读+letterbox一次。"""
    if cache is not None and (root, stem) in cache:
        m8, cls, xyxy = cache[(root, stem)]
        m = m8.astype(np.float32) / 255.0
        return m, cls, xyxy
    a = np.load(os.path.join(root, 'attrmaps', f'{stem}.npz'))
    m = np.stack([a['depth'], a['reflec'], a['height'], a['mask']]).astype(np.float32)
    m = letterbox_4ch(normalize4(m))

    lbl = os.path.join(root, 'labels', f'{stem}.txt')
    if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
        labels = np.loadtxt(lbl, dtype=np.float32).reshape(-1, 5)
    else:
        labels = np.empty((0, 5))
    cls, xyxy = letterbox_boxes(labels, (720, 1280))     # attrmap 为 720×1280, 同 images 几何
    return m, cls, xyxy


def build_attr_cache(stems: list) -> dict:
    """把要用到的帧一次性 letterbox 为 uint8 (4,640,640) 缓存进 RAM, 消除每步 npz 读+resize。
    每帧 ≤1.6MB(uint8), 千帧级 ≈2GB; attrmap [0,1]×255 量化到 uint8 对特征提取热启动足够。"""
    cache: dict = {}
    for root, stem in stems:
        m, cls, xyxy = load_attr_sample(stem, root)          # 磁盘读+resize 仅此一次
        cache[(root, stem)] = (np.clip(np.round(m * 255), 0, 255).astype(np.uint8), cls, xyxy)
    return cache


def augment_attr_sample(s, rng, flip=0.5):
    """hflip 同步翻 attrmap(4,640,640 宽轴)+框(YOLO_SIZE-x); attrmap 无相位, 只翻几何。"""
    m, cls, xyxy = s
    m = m.copy()
    if rng.random() < flip:
        m = m[:, :, ::-1]
        if xyxy.shape[0]:
            xyxy = np.stack([YOLO_SIZE - xyxy[:, 2], xyxy[:, 1],
                             YOLO_SIZE - xyxy[:, 0], xyxy[:, 3]], 1)
    return m, cls, xyxy


def collate_attr_batch(samples):
    """list[load_attr_sample] -> (attr(B,4,640,640), batch dict)。bboxes 为 640 画布 normalized xywh。"""
    attrs = np.stack([s[0] for s in samples])
    cls_list, xyxy_list, bidx_list = [], [], []
    for bi, s in enumerate(samples):
        cls_list.append(s[1]); xyxy_list.append(s[2])
        bidx_list.append(np.full(s[2].shape[0], bi, dtype=np.float32))
    cls = np.concatenate(cls_list) if cls_list and cls_list[0].size else np.empty((0,), np.float32)
    xyxy = np.concatenate(xyxy_list) if xyxy_list and xyxy_list[0].size else np.empty((0, 4), np.float32)
    bidx = np.concatenate(bidx_list) if bidx_list and bidx_list[0].size else np.empty((0,), np.float32)
    if xyxy.shape[0]:
        xc = (xyxy[:, 0] + xyxy[:, 2]) / 2
        yc = (xyxy[:, 1] + xyxy[:, 3]) / 2
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        xywh = np.stack([xc, yc, w, h], 1) / YOLO_SIZE
    else:
        xywh = np.empty((0, 4), np.float32)
    batch = {'cls': torch.from_numpy(cls), 'bboxes': torch.from_numpy(xywh),
             'batch_idx': torch.from_numpy(bidx)}
    return torch.from_numpy(attrs), batch


def resolve_nc(root, args_nc=None):
    if args_nc:
        return args_nc
    p = os.path.join(root, 'data.yaml')
    if os.path.isfile(p):
        m = re.search(r'^nc:\s*(\d+)', open(p, encoding='utf-8').read(), re.M)
        if m:
            return int(m.group(1))
    return 8


def build_lr(opt, lr, epochs):
    import math
    def fn(ep):
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ep / epochs)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=fn)


def parse_args():
    p = argparse.ArgumentParser(description='点云编码器检测式预训 (pro/3)')
    p.add_argument('--data', default='all',
                   help='pcdata 数据集目录: 逗号分隔 / glob / all(缺省=全部 8cls 集)')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--nc', type=int, default=None, help='覆盖 data.yaml 的类别数, 缺省自动读')
    p.add_argument('--no-aug', action='store_true', help='关闭 hflip(默认开)')
    p.add_argument('--quick', action='store_true', help='2 epoch 冒烟(loss 下降验证)')
    p.add_argument('--cache-attrs', action='store_true',
                   help='启动时把 attrmap 全量 letterbox 到 (4,640,640) uint8 缓存进 RAM, '
                        '训练零磁盘读 —— 本任务 I/O 网络字节瓶颈(npz 每步重读), 此参数是关键')
    p.add_argument('--out', default=os.path.join(_P, 'runs', 'pcl_detect'))
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs, args.batch = 2, 4
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    roots = resolve_roots(args.data)
    if not roots:
        sys.exit(f"[data] --data 解析不出任何目录: {args.data!r}")
    for r in roots:
        if not os.path.isdir(r):
            print(f"[warn] {r} 不存在")
    stems = list_samples(roots)               # [(root, stem), ...]
    assert stems, f"[data] 所有集({len(roots)})均无 attrmaps/*.npz"

    # 纯背景(0 框)帧: 送入会把 v8DetectionLoss 打成 cls=~40000 爆炸(除数随 0 正样本退化)。
    # 检测预训学定位特征即可, 负样本抑制由正图背景承担; 过滤零框帧更稳(参考 output_v7 无此问题)。
    def _has_box(rs):
        lbl = os.path.join(rs[0], 'labels', f'{rs[1]}.txt')
        return os.path.isfile(lbl) and os.path.getsize(lbl) > 0
    pos = [s for s in stems if _has_box(s)]
    n_drop = len(stems) - len(pos)
    if n_drop:
        print(f"[data] 过滤 {n_drop} 帧零框背景帧, 训练用 {len(pos)} 帧")
    stems = pos
    random.shuffle(stems)
    n_val = max(1, int(len(stems) * args.val_ratio))
    val_stems, train_stems = stems[:n_val], stems[n_val:]
    cache = build_attr_cache(stems) if args.cache_attrs else None
    if cache is not None:
        print(f"[cache] attrmaps 全量缓存 {len(cache)} 帧 -> RAM, 训练零磁盘读")
    print(f"[data] 全量预训 {len(roots)} 集: {len(stems)} 帧, "
          f"train {len(train_stems)} / val {len(val_stems)}, device {device}")

    nc = resolve_nc(roots[0], args.nc)         # 全 8cls 集共享同一 taxonomy, nc=8
    model = PointCloudDetector(nc=nc, device=device)
    print(f"[nc] data.yaml 声明 {nc} 类; {model.pcl.describe()}")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = build_lr(opt, args.lr, args.epochs)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, 'train.log')
    with open(log_path, 'w') as flog:
        flog.write('epoch,box,cls,dfl,val_box,val_cls,val_dfl\n')

    def run_batch(idxs, grad, aug=False):
        samples = [load_attr_sample(i[1], i[0], cache) for i in idxs]
        if aug:
            samples = [augment_attr_sample(s, aug_rng) for s in samples]
        attrs, batch = collate_attr_batch(samples)
        attrs = attrs.to(device)
        for k, v in batch.items():
            batch[k] = v.to(device)
        if grad:
            loss, ld = model(attrs, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            return {k: float(v) for k, v in ld.items()}
        with torch.no_grad():
            _, ld = model(attrs, batch)
            return {k: float(v) for k, v in ld.items()}

    aug_rng = random.Random(args.seed + 1)
    for ep in range(1, args.epochs + 1):
        random.shuffle(train_stems)
        for bi in range(0, len(train_stems), args.batch):
            ld = run_batch(train_stems[bi:bi + args.batch], grad=True, aug=not args.no_aug)
            if bi % (args.batch * 4) == 0 or args.quick:
                print(f"  ep{ep:02d} it{bi//args.batch:03d} box={ld['box_loss']:.3f} "
                      f"cls={ld['cls_loss']:.1f} dfl={ld['dfl_loss']:.3f}")
        sched.step()
        vld = {k: [] for k in ('box_loss', 'cls_loss', 'dfl_loss')}
        for bi in range(0, len(val_stems), args.batch):
            ldv = run_batch(val_stems[bi:bi + args.batch], grad=False)
            for k in vld:
                vld[k].append(ldv[k])
        vavg = {k: float(np.mean(v)) for k, v in vld.items()}
        with open(log_path, 'a') as flog:
            flog.write(f"{ep},{ld['box_loss']:.3f},{ld['cls_loss']:.1f},{ld['dfl_loss']:.3f},"
                       f"{vavg['box_loss']:.3f},{vavg['cls_loss']:.1f},{vavg['dfl_loss']:.3f}\n")
        print(f"[ep{ep:02d}] val box={vavg['box_loss']:.3f} cls={vavg['cls_loss']:.1f} "
              f"dfl={vavg['dfl_loss']:.3f} lr={sched.get_last_lr()[0]:.2e}")

    # ---- 存档: 全 state_dict(含 BN 统计) + 摘出的点云编码器 ----
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({'nc': nc, 'epochs': args.epochs, 'hyp': dict(HYP), 'model': sd},
               os.path.join(args.out, 'last.pt'))
    torch.save(model.pcl, os.path.join(args.out, 'pcl_encoder.pt'))
    print(f"[save] {os.path.join(args.out, 'last.pt')} + pcl_encoder.pt  (enc: {model.pcl.describe()})")


if __name__ == '__main__':
    main()
