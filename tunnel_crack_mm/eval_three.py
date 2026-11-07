#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三方对照: 原版 v8 / RangeQFL(γ=2,β=0.5) / PQA 选出(γ=3,β=0.25)。"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

from eval_qfl import run  # noqa: E402
from fusion.evaluate_pr import train_val_split  # noqa: E402
from compare_branches import DATA, gt_from_fusion_val  # noqa: E402

_IR = os.environ.get(
    'IR_ROOT',
    os.path.abspath(os.path.join(_HERE, '..', 'inspection_robot')),
)
CKPTS = [
    ('原版 v8', os.path.join(_IR, 'runs/assist_crack/last.pt')),
    ('QFL g2b.5', os.path.join(_IR, 'runs/assist_crack_qfl/last.pt')),
    ('QFL g3b.25', os.path.join(_IR, 'runs/assist_crack_qfl_g3b025/last.pt')),
]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_img = len([f for f in os.listdir(os.path.join(_IR, DATA, 'images'))
                 if f.endswith('.jpg')])
    splits = {
        'heldout': list(range(max(1, int(n_img * 0.2)))),
        'shuffled': train_val_split(n_img, 0.2, 42)[0],
    }
    out = {}
    for name, idx in splits.items():
        print('\n' + '=' * 70 + f'\n评测集 {name}: {len(idx)} 帧\n' + '=' * 70)
        gts = gt_from_fusion_val(idx)
        out[name] = {}
        for tag, ck in CKPTS:
            if not os.path.exists(ck):
                print(f'[{tag}] 缺 {ck}')
                continue
            out[name][tag] = run(tag, ck, idx, gts, device)
    dst = os.path.join(_HERE, 'runs', 'eval_three.json')
    with open(dst, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n[done] {dst}')


if __name__ == '__main__':
    main()
