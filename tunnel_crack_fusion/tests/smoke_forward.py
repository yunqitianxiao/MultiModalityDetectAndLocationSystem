#!/usr/bin/env python3
"""前向 + 反向冒烟：两档各跑一个真实样本 batch。"""

import sys
import time

import torch

from tunnel_tri import config as C
from tunnel_tri.data import collate_batch, load_sample
from tunnel_tri.model import DualModalYOLO


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', dev, '| data:', C.DATA_ROOT)
    samples = [load_sample(i) for i in (0, 1)]
    imgs, attrs, batch = collate_batch(samples)
    print('imgs', tuple(imgs.shape), '| attrs', None if attrs is None else tuple(attrs.shape),
          '| gt boxes', batch['bboxes'].shape[0])

    for mod in ('rgb', 'rgb+pc'):
        t0 = time.time()
        model = DualModalYOLO(modalities=mod, device=dev)
        model.train()
        loss, ld = model(imgs, attrs, batch)
        loss.backward()
        n_bad = sum(1 for p in model.trainable_params()
                    if p.grad is None or not bool(torch.isfinite(p.grad).all()))
        n_train = sum(1 for _ in model.trainable_params())
        print(f'[{mod}] loss={float(loss):.4f} '
              f'box={float(ld["box_loss"]):.4f} cls={float(ld["cls_loss"]):.4f} '
              f'dfl={float(ld["dfl_loss"]):.4f} | 可训练 tensor {n_train} 个, '
              f'梯度异常 {n_bad} 个 | {time.time()-t0:.1f}s')
        assert n_bad == 0, f'{mod} 梯度异常'
        del model
    print('SMOKE OK')


if __name__ == '__main__':
    sys.exit(main())
