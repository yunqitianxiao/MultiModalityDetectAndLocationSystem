#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
土建渗漏水 · YOLO26x 检测训练（纯检测，验证性能/数据用）
========================================================
目的：先用标准 YOLO 检测看看 yolo26x 在渗漏水数据上的精度/收敛，
      以及数据本身能不能用（标签质量、难易）。不接多模态/多任务架构。

硬件：7×V100 32G（合计 224G 显存），单机多卡 DDP。

环境：
  pip install -U ultralytics   # 8.4.120 不含 YOLO26，必须升级到最新版
  torch 2.x + CUDA（本项目 .venv 已是 torch 2.5.1+cu121）

注意：
  - V100 = sm_70，不支持 bf16；混合精度用 fp16（ultralytics 默认 amp 即 fp16）。
  - 首次运行自动从官方 release 下载 yolo26x.pt（需联网，可设代理）。

用法：
  python train_leakage_yolo26x.py --verify-only   # 只校验数据（不训练，可本地跑）
  python train_leakage_yolo26x.py --epochs 100    # 完整训练（在 7×V100 服务器上）
"""

import argparse
import os
import sys
from pathlib import Path

# ========== 配置区（唯一需要改的地方） ==========
# PROJ_ROOT 可用环境变量覆盖，例如: PROJ_ROOT=/root/autodl-tmp/leakage_train python ...
PROJ_ROOT = Path(os.environ.get("PROJ_ROOT", Path(__file__).resolve().parent))
DATASET   = "土建_渗漏水_yolo"                     # data/ 下的数据集目录
MODEL     = "yolo26x.pt"                          # yolo26 xlarge
IMGSZ     = 1280                                  # 输入分辨率（渗漏水区域病害，用大图）
EPOCHS    = 100
DEVICE    = "0,1,2,3,4,5,6"                       # 7×V100
BATCH     = 0.85                                  # 用 ~85% 显存：7×32×0.85 ≈ 190G 有效
WORKERS   = 8                                     # 每卡 dataloader 进程数
PROJECT   = "runs_leakage"                        # 输出根目录
SEED      = 42
# ==============================================

DATA_YAML = PROJ_ROOT / "data" / DATASET / "data.yaml"


def parse_nc(data_yaml: Path) -> int:
    """从 data.yaml 解析 nc（类别数），供校验用。"""
    nc = 1
    if data_yaml.exists():
        for line in data_yaml.read_text().splitlines():
            line = line.strip()
            if line.startswith("nc:"):
                try:
                    nc = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    return nc


def parse_splits(data_yaml: Path) -> dict:
    """从 data.yaml 解析 train/val/test 的相对目录（兼容 train/images 与 images/train 两种布局）。"""
    splits = {}
    if data_yaml.exists():
        for line in data_yaml.read_text().splitlines():
            line = line.strip()
            for key in ("train", "val", "test"):
                if line.startswith(key + ":"):
                    splits[key] = line.split(":", 1)[1].strip()
    return splits


def verify_dataset() -> bool:
    """数据自检：图片/标签一一对应，label 行格式/坐标范围/类别编号合法。"""
    nc = parse_nc(DATA_YAML)
    splits = parse_splits(DATA_YAML)
    print(f"data.yaml nc = {nc}  布局 = {splits}")
    ok = True
    for split, rel in splits.items():
        img_dir = DATA_YAML.parent / rel
        lbl_dir = DATA_YAML.parent / rel.replace("images", "labels")
        imgs = sorted(img_dir.glob("*"))
        lbls = sorted(lbl_dir.glob("*.txt"))
        print(f"[{split}] images={len(imgs)}  labels={len(lbls)}", end="")
        if len(imgs) != len(lbls):
            ok = False
            print("  ✗ 图片/标签数量不一致", end="")
        print()

        bad, empty = 0, 0
        for p in lbls:
            lines = [ln for ln in p.read_text().strip().splitlines() if ln.strip()]
            if not lines:
                empty += 1
                continue
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    bad += 1
                    continue
                try:
                    c, cx, cy, w, h = map(float, parts)
                except ValueError:
                    bad += 1
                    continue
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and w > 0 and h > 0) or c != int(c):
                    bad += 1
                elif not (0 <= int(c) < nc):
                    bad += 1
        print(f"    非法行={bad}  空标签={empty}")
        if bad:
            ok = False
    print("数据集校验:", "✓ 通过" if ok else "✗ 有问题，先修数据再训练")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true", help="只校验数据，不训练")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--batch", type=float, default=BATCH,
                    help="int=batch 数（多卡必须整数，DDP 按卡均分），或 0~1=显存比例（仅单卡）")
    args = ap.parse_args()
    if args.batch >= 1:
        args.batch = int(args.batch)   # 多卡 DDP 均分后必须是整数，否则 BatchSampler 报错

    if not DATA_YAML.exists():
        sys.exit(f"✗ 找不到 {DATA_YAML}\n  请检查 PROJ_ROOT 是否为服务器实际路径，数据集是否已拷到服务器")

    if args.verify_only:
        sys.exit(0 if verify_dataset() else 1)

    verify_dataset()

    from ultralytics import YOLO, __version__ as uver
    print(f"ultralytics {uver}  （若不含 YOLO26，请先 pip install -U ultralytics）")

    model = YOLO(MODEL)  # 首次自动下载 yolo26x.pt

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,   # fraction → 自动按可用显存占比取 batch，拉满且不溢出
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=f"yolo26x_imgsz{args.imgsz}",
        optimizer="auto",
        amp=True,           # fp16（V100 不支持 bf16）
        seed=SEED,
    )


if __name__ == "__main__":
    main()
