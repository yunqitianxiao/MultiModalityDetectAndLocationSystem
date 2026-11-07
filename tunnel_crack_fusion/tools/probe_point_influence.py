#!/usr/bin/env python3
"""判据：点云的取值到底有没有进入前向图。

对验证帧分别喂 真点云 / 全零 / 随机 / 换帧 四种输入，比较预测张量的最大差。
三种注入下若都恒为 0，说明点云支没有影响任何输出（融合被写成了恒等），
此前的 A/B 差值只可能来自其他模块/初始化，不能归因于点云。

用法:
    PYTHONPATH=. python3 tools/probe_point_influence.py --ckpt runs/xxx/last.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tunnel_tri import config as C
from tunnel_tri.data import list_frames, load_sample, split_indices
from tunnel_tri.evaluate import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='')
    ap.add_argument('--n-frames', type=int, default=4)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ck = load_model(args.ckpt, dev)
    assert model.use_point, '这是 rgb 档，没有点云支'
    root = args.data or (ck.get('args') or {}).get('data') or C.DATA_ROOT

    frames = list_frames(root)
    val_idx, _ = split_indices(frames, 0.2, 'sequential', 42)
    idx = val_idx[:args.n_frames]

    imgs, attrs = [], []
    for i in idx:
        img, attr, *_ = load_sample(i, root, with_attr=True)
        imgs.append(img)
        attrs.append(attr)
    img = torch.from_numpy(np.stack(imgs)).float().to(dev)
    a = torch.from_numpy(np.stack(attrs)).float().to(dev)

    cases = {
        '全零': torch.zeros_like(a),
        '随机': torch.randn_like(a),
        '换帧': a.roll(1, dims=0),
    }
    with torch.no_grad():
        p0 = model(img, a)[0]
        print(f'预测 {tuple(p0.shape)} | 分数范围 [{p0.min():.4f}, {p0.max():.4f}]')
        worst = 0.0
        for name, x in cases.items():
            d = (model(img, x)[0] - p0).abs().max().item()
            worst = max(worst, d)
            print(f'  {name:<4}: 最大差 {d:.3e}')
    print(f'\n最大差 = {worst:.3e}')
    print('=> 点云对预测零影响（架构死支）' if worst == 0
          else '=> 点云已进入前向图')
    return 0 if worst > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
