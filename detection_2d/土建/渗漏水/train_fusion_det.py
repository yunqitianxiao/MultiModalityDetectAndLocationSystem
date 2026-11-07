#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨分支自注意力融合训练 —— "编码器(冻结) + 融合(self) + 适配 + 检测头"
================================================================================
对应架构: 检测编码器 → ④ 融合(跨分支 self-attn) → ⑤ 检测解码器。
目标: 验证【跨分支自注意力对单分支检测的增益】。

【实验设计 —— 逐分支看增益】
  三个检测编码器(轨道/土建/设备)全部冻结, 各自从同一张图抽 P3/P4/P5;
  CrossModalFusion 把三支 concat 后做 ①Self-Attn → 门控, 得融合特征金字塔;
  适配器把融合 256 → 检测头各尺度通道(256/512/512), 接【被测分支原 Detect 头(继承)】。
  用被测分支的检测数据监督: 看"额外两支的上下文"能否抬升该分支 mAP。
  每次只测一个分支(--branch), 换分支重跑; 比较三支相对各自基线的增益。

【为什么这样训不是浪费】
  - 三支数据集是三类不同病害标注, 但都是同一套巡检车拍同一类隧道图。
    被测分支训练时, 把同一张图喂给三支冻结编码器, 融合层学到跨分支上下文;
    监督只有被测分支的真标签, 但融合层可以借另外两支的冻结特征。
  - 保留原 Detect 头权重(load_detect_from_pretrained), 只重训 融合+适配+头微调。

【真实 loss —— 复用 ultralytics v8DetectionLoss】
  不手搓 loss。直接用被测分支的 ultralytics DetectionModel 作 wrapper:
    wrapper.model[-1] = 原 Detect(没动) → criterion = v8DetectionLoss(wrapper)
    读 wrapper.model[-1] 的 stride/nc/reg_max + wrapper.args 的 hyp, 全是原样的。
  前向: encoders(冻结) → concat → fusion → adapters → Detect → preds(dict, 含 'boxes')
  loss = criterion(preds, batch); backward 只过 fusion+adapters+detect。

【用法(服务器 8×4090)】
  python train_fusion_det.py --branch 轨道 --epochs 200 --imgsz 1024 --batch 2 --device 0
  python train_fusion_det.py --branch 土建 --epochs 200 --imgsz 1024 --batch 2 --device 1
  python train_fusion_det.py --branch 设备 --epochs 200 --imgsz 1024 --batch 2 --device 2
  # 先本地验证接线:
  python train_fusion_det.py --smoke
"""
from __future__ import annotations

import argparse, os, copy

import torch
import torch.nn as nn

import det_encoders_runtime as R
from fusion import CrossModalFusion, total_channels_per_scale
from det_heads import FusionDetectHead

_BRANCH_ORDER = ("轨道", "土建", "设备")
_BRANCHES = {
    "轨道": dict(weight="guidao_best.pt", data="data1/轨道_3cls_均衡", nc=3),
    "土建": dict(weight="tujian_best.pt", data="data1/土建_病害_yolo_3cls", nc=3),
    "设备": dict(weight="shebei_best.pt", data="data1/设备_异物支架_合并_yolo", nc=2),
}
_HEAD_CH = (256, 512, 512)     # 检测头各尺度输入通道(neck)


def _find_repo():
    """向上找仓库根: 第一个含 data1 / guidao_best.pt / data(含已知数据集) 的目录。
    本地用 data1, 服务器用 data/设备_异物支架_合并_yolo 作根标志, 脚本放哪层都能对上。"""
    d = os.path.dirname(os.path.abspath(__file__))
    srv_marker = os.path.join("data", "设备_异物支架_合并_yolo", "data.yaml")
    for _ in range(6):
        if (os.path.isdir(os.path.join(d, "data1"))
                or os.path.exists(os.path.join(d, "guidao_best.pt"))
                or os.path.exists(os.path.join(d, srv_marker))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(os.path.abspath(__file__))   # 兜底: 脚本所在目录


_REPO = _find_repo()


# ------------------------------------------------------------------ Wrapper
class FusionWrapper(nn.Module):
    """把 冻结编码器+融合+适配 挂到被测分支的原 DetectionModel 上。
    forward 不改动原模型结构, 只在 Detect 前插入融合; 复用原 criterion/args/stride。"""

    def __init__(self, model, encs, fusion, adapters):
        super().__init__()
        self.model = model          # ultralytics DetectionModel(带原 Detect/.args/.class_weights)
        self.encs = encs            # {branch: {mode: encoder}} 冻结
        self.fusion = fusion
        self.adapters = adapters    # ModuleList[Conv2d(256->每尺度头通道)]
        # 关掉 end2end, v8DetectionLoss 只认非 one2one 的 assigner 路径
        det = self.model.model[-1]
        det.train()                 # Detect 走训练分支(forward_head 返回 dict)
        if hasattr(det, "end2end"):
            det.end2end = False
        # v8DetectionLoss 读 model.args 用属性访问(self.hyp.box/.cls/.dfl)。加载的 best.pt
        # 里 args 缺 loss 增益(只有 task/data/imgsz 等), 用 ultralytics get_cfg 默认值打底再覆盖。
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import IterableSimpleNamespace
        a = getattr(self.model, "args", None)
        d = dict(vars(get_cfg()))
        if isinstance(a, dict):
            d.update(a)
        elif a is not None:
            d.update(vars(a))
        self.model.args = IterableSimpleNamespace(**d)

    def forward(self, x):
        bf = {b: {"neck": md["neck"](x)} for b, md in self.encs.items()}
        fused = self.fusion(bf)                       # {P3,P4,P5} @embed_dim
        names = ("P3", "P4", "P5")
        feats = [self.adapters[i](fused[n]) for i, n in enumerate(names) if n in fused]
        return self.model.model[-1](feats)            # Detect 训练输出 dict(含 'boxes')

    @property
    def criterion(self):
        # 建在 DetectionModel 上: model.model[-1]=Detect, model.args, model.class_weights 全齐
        if getattr(self, "_crit", None) is None:
            from ultralytics.utils.loss import v8DetectionLoss
            self._crit = v8DetectionLoss(self.model)
        return self._crit

    def loss(self, preds, batch):
        return self.criterion(preds, batch)


def _resolve_yaml(branch_conf):
    data_root = os.path.join(_REPO, branch_conf["data"])
    p = os.path.join(data_root, "data.yaml")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{data_root}/ 无 data.yaml")
    return p


def build_model(branch, data_override=None):
    conf = dict(_BRANCHES[branch])
    if data_override:
        conf["data"] = data_override          # 服务器数据集若不在 data1/ 下, 用 --data 指到实际目录
    yaml_path = _resolve_yaml(conf)

    # 被测分支的原 DetectionModel(带原 Detect 头 + args + 真实 criterion 依赖)
    from ultralytics import YOLO
    model = YOLO(os.path.join(_REPO, conf["weight"])).model
    model.eval()

    # 冻结被测分支编码器层 0..22(backbone+neck), 仅 Detect[-1] 可训
    for layer in model.model[0:23]:
        for p in layer.parameters():
            p.requires_grad_(False)

    # 三支编码器(冻结): 当前被测分支复用 model.model[0:23]? 不 —— 统一用摘出的独立编码器
    encs = R.load_all_encoders()
    for b, md in encs.items():
        for mode, enc in md.items():
            for p in enc.parameters():
                p.requires_grad_(False)

    # 融合层 + 适配器
    fusion = CrossModalFusion(total_channels_per_scale("neck"), embed_dim=256, heads=4)
    adapters = nn.ModuleList([nn.Conv2d(256, c, 1) for c in _HEAD_CH])

    wrapper = FusionWrapper(model, encs, fusion, adapters)
    return wrapper, yaml_path, conf["nc"]


# ------------------------------------------------------------------ Smoke test(CPU, 验证接线)
def smoke(imgsz: int = 256, device: str = "cpu"):
    print("[smoke] CPU 小尺寸合成输入, 验证 编码器→融合→适配→Detect→loss→backward 全链路")
    torch.set_grad_enabled(True)
    wrapper, _yaml, nc = build_model("轨道")
    wrapper.train()
    for p in wrapper.parameters():
        if p.requires_grad:
            p.to(device)
    wrapper.to(device)

    x = torch.rand(2, 3, imgsz, imgsz, device=device)   # [0,1]
    preds = wrapper(x)
    print(f"  forward: preds type={type(preds).__name__}", end="")
    if isinstance(preds, dict):
        print(f" keys={list(preds.keys())} boxes={tuple(preds['boxes'].shape)}")
    else:
        print(f" keys={list(preds.keys())}")
        raise SystemExit("Detect 未走训练分支(forward_head dict)—— 请确认 detect.train()")

    # 造一个最小 batch
    bs = x.shape[0]
    batch = {
        "img": x,
        "cls": torch.randint(0, nc, (6, 1), device=device),
        "bboxes": torch.rand(6, 4, device=device) * imgsz,
        "batch_idx": torch.tensor([0, 0, 0, 1, 1, 1], device=device),
    }
    loss, loss_detach = wrapper.criterion(preds, batch)
    ls = float(torch.as_tensor(loss).detach().sum())
    ld = {k: float(v.detach().item()) for k, v in loss_detach.items()} if isinstance(loss_detach, dict) else loss_detach
    print(f"  loss={ls:.4f}  loss_detach={ld}")
    loss.sum().backward()
    g = [p.grad is not None and float(p.grad.abs().sum()) for p in wrapper.fusion.parameters() if p.requires_grad]
    print(f"  backward OK, fusion 可训参数带梯度的={sum(1 for gg in g if gg is not None)}/{len(g)}")
    print("FUSION_SMOKE_OK")


# ------------------------------------------------------------------ Train
def _parsed_device(s: str) -> torch.device:
    """把 --device 字符串规格化成 torch.device: '0'->cuda:0, 'cpu'->cpu, 'cuda:0' 原样。"""
    s = str(s)
    if s == "cpu":
        return torch.device("cpu")
    if s.isdigit():
        return torch.device("cuda:" + s)
    return torch.device(s)


def main():
    ap = argparse.ArgumentParser(description="跨分支自注意力融合训练(看增益)")
    ap.add_argument("--branch", default="轨道", choices=list(_BRANCHES))
    ap.add_argument("--data", default=None, help="覆盖数据集 yaml 所在目录(相对 root 或绝对); 服务器数据集不在 data1/ 时用")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="CPU 小图验证接线后退出")
    ap.add_argument("--save", default=None, help="保存 wrapper 权重路径")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    device = _parsed_device(args.device)
    wrapper, yaml_path, nc = build_model(args.branch, data_override=args.data)
    wrapper.to(device)
    # 编码器在 FusionWrapper 里是普通 dict(非注册子模块), nn.Module.to 不会递归;
    # 它们各自是独立 nn.Module, 必须手动搬到 device, 否则前向会 device 不匹配。
    for _md in wrapper.encs.values():
        for _enc in _md.values():
            _enc.to(device)
    wrapper.train()
    model_detect = wrapper.model.model[-1]
    model_detect.train()          # Detect 走训练分支(forward_head dict)

    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable) / 1e6
    n_frozen = sum(p.numel() for p in wrapper.parameters()) / 1e6 - n_train
    opt = torch.optim.AdamW([
        {"params": [p for p in wrapper.fusion.parameters() if p.requires_grad], "lr": args.lr},
        {"params": [p for p in wrapper.adapters.parameters() if p.requires_grad], "lr": args.lr},
        {"params": [p for p in model_detect.parameters() if p.requires_grad], "lr": args.lr * 0.1},
    ], weight_decay=1e-4)

    print(f"\n[训练] branch={args.branch}  nc={nc}  epochs={args.epochs} imgsz={args.imgsz} "
          f"batch={args.batch} device={args.device}")
    print(f"[参数量] 冻结(编码器+原骨干)={n_frozen:.1f}M  训练(融合+适配+头)={n_train:.1f}M")

    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data.build import build_yolo_dataset, build_dataloader
    wrapper.model.args.imgsz = args.imgsz          # 让 YOLODataset 统一 letterbox 到 imgsz, 与编码器对齐
    data = check_det_dataset(yaml_path)
    gs = max(int(wrapper.model.model[-1].stride.max()), 32)
    train_ds = build_yolo_dataset(wrapper.model.args, data["train"], args.batch, data,
                                  mode="train", rect=False, stride=gs)
    loader = build_dataloader(train_ds, batch=args.batch, workers=args.workers,
                              shuffle=True, rank=-1, device=device)

    for epoch in range(args.epochs):
        wrapper.train(); model_detect.train()
        loss_sum, n = 0.0, 0
        for batch in loader:
            imgs = batch["img"].to(device, non_blocking=True)      # (B,3,H,W) uint8(loader 未归一化)
            if imgs.dtype == torch.uint8:
                imgs = imgs.float() / 255.0                          # -> [0,1] float, 对齐编码器权重训练域(BGR 顺序不变)
            # v8DetectionLoss 只读 cls/bboxes/batch_idx(不读 img, 也不要 im_file/shape 等非张量键)
            b = {k: batch[k].to(device) for k in ("cls", "bboxes", "batch_idx")}
            preds = wrapper(imgs)
            loss, _detach = wrapper.criterion(preds, b)
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            loss_sum += float(loss.sum()); n += 1
        print(f"  epoch {epoch}: loss={loss_sum / max(1, n):.4f}", flush=True)

    if args.save:
        torch.save(wrapper.state_dict(), args.save)
        print(f"已保存 -> {args.save}")
    print("TRAIN_FUSION_DONE")


if __name__ == "__main__":
    main()
