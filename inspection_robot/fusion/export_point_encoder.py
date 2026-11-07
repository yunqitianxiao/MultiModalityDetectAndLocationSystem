# -*- coding: utf-8 -*-
"""
导出训练后的点云编码器为独立权重 (复用组件)
================================================================
train_assist.py 产出的 {out}/last.pt 里, 点云编码器以 `point_encoder.*` 前缀
存于全模型 state_dict。本脚本把这一子模块抽出来, 去掉前缀, 存成与
pretrain_point.py 产物同格式的 checkpoint:

    {'encoder': <PointCloudEncoder state_dict>, 'nc', 'use_point', 'source'}

这样直接喂给 `train_assist.py --point-ckpt` (它按 ck['encoder'] 取) 或任何
需要 attrmap 编码的地方, 复用方式与 MAE 预训练 weights 完全一致。

用法:
    .venv/bin/python fusion/export_point_encoder.py \
        --ckpt runs/assist_data1/last.pt \
        --out runs/point_encoder_finetuned_data1.pt
"""
import argparse
import os
import sys

import torch

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from fusion.pointcloud_encoder import PointCloudEncoder  # noqa: E402


def extract_encoder_sd(model_sd, prefix='point_encoder.'):
    """从全模型 state_dict 抽点云编码器并去掉前缀。"""
    out = {}
    for k, v in model_sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='runs/assist_data1/last.pt')
    p.add_argument('--out', default='runs/point_encoder_finetuned_data1.pt')
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        sys.exit(f"❌ 读不到 {args.ckpt}")
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = extract_encoder_sd(ck['model'])

    # 校验: 能完整 load 进 PointCloudEncoder (strict=False 查缺失/意外)
    enc = PointCloudEncoder()
    miss, unexp = enc.load_state_dict(sd, strict=False)
    assert not miss, f"导出缺键: {miss}"
    print(f"✅ 抽出点云编码器 state_dict {len(sd)} 键 (missing=0, unexpected={len(unexp)})")

    out_ck = {
        'encoder': sd,
        'nc': ck.get('nc'),
        'use_point': ck.get('use_point'),
        'source': f"{args.ckpt} (ep{ck.get('epochs', '?')}, finetuned)",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out_ck, args.out)
    print(f"✅ 已导出: {args.out}")

    # ---- forward 冒烟: 载入导出的编码器跑真实 attrmap, 确认 shape/有限 ----
    import numpy as np
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc2 = PointCloudEncoder(device=device)
    enc2.load_state_dict(sd, strict=False)
    enc2.eval()
    # 随机 attrmap 也要能过(不需要真实文件, 契约 shape 检查)
    x = torch.rand(1, 3, 720, 1280, device=device)
    with torch.no_grad():
        o = enc2(x)
    ok_shape = tuple(o['feat'].shape) == (1, 256, 45, 80) and tuple(o['tokens'].shape) == (1, 3600, 256)
    ok_fin = bool(torch.isfinite(o['feat']).all() and torch.isfinite(o['tokens']).all())
    print(f"   forward: feat {tuple(o['feat'].shape)} tokens {tuple(o['tokens'].shape)} | "
          f"finite={ok_fin} | shape_ok={ok_shape}")
    assert ok_shape and ok_fin, "导出编码器前向异常"
    print("   ✔ 导出的编码器可独立前向")


if __name__ == '__main__':
    main()
