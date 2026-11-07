#!/usr/bin/env python3
"""训练双模态裂缝检测基线。

用法:
    python3 -m tunnel_tri.train --modalities rgb     --out runs/rgb
    python3 -m tunnel_tri.train --modalities rgb+pc  --out runs/rgb_pc

默认按序列尾部留出验证集（sequential），避免相邻帧跨集泄漏。
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as C
from .data import PairedDataset, collate_batch, list_frames, split_indices
from .model import DualModalYOLO


def parse_args():
    p = argparse.ArgumentParser(description='双模态裂缝检测训练')
    p.add_argument('--modalities', default='rgb+pc', choices=('rgb', 'rgb+pc'))
    p.add_argument('--data', default=C.DATA_ROOT)
    p.add_argument('--yolo', default=C.YOLO_CKPT, help='图像主干 ckpt（默认 COCO yolo11l）')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-backbone', type=float, default=1e-5)
    p.add_argument('--unfreeze-neck', type=int, default=0, help='解锁末 N 个 neck 模块')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--split', default='sequential', choices=('sequential', 'random'))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--amp', action='store_true', help='混合精度（默认关，融合里 LN/SDPA 在 fp16 下不稳）')
    p.add_argument('--out', default='runs/baseline')
    p.add_argument('--snapshot-every', type=int, default=20)
    p.add_argument('--max-iters', type=int, default=0, help='每 epoch 最多迭代数，0=不限制（冒烟用）')
    return p.parse_args()


def build_model(args, device):
    return DualModalYOLO(ckpt=args.yolo, nc=C.NC, modalities=args.modalities,
                         freeze_backbone=(args.unfreeze_neck == 0),
                         unfreeze_neck=args.unfreeze_neck, device=device)


def optimizer_groups(model, args):
    backbone, main = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if n.startswith('image_encoder.model.') else main).append(p)
    groups = [{'params': main, 'lr': args.lr}]
    if backbone:
        groups.append({'params': backbone, 'lr': args.lr_backbone})
    return groups


def save_state(model, args, path):
    skip = 'image_encoder.model.' if not model.image_encoder._unfrozen else None
    state = {k: v for k, v in model.state_dict().items()
             if not (skip and k.startswith(skip))}
    torch.save({'model': state, 'args': vars(args), 'modalities': model.modalities,
                'nc': model.nc}, path)


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    frames = list_frames(args.data)
    val_idx, train_idx = split_indices(frames, args.val_ratio, args.split, args.seed)
    print(f'[data] {args.data} | 总 {len(frames)} 帧 | train {len(train_idx)} / val {len(val_idx)}'
          f' | split={args.split}')
    meta = {'args': vars(args), 'n_train': len(train_idx), 'n_val': len(val_idx),
            'val_idx': val_idx, 'zero_reflec': C.ZERO_REFLEC,
            'fixed_attr': os.environ.get('TUNNEL_FIXED_ATTR')}
    with open(os.path.join(args.out, 'split.json'), 'w') as f:
        json.dump(meta, f)

    ds = PairedDataset(train_idx, args.data, with_attr=(args.modalities == 'rgb+pc'))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate_batch, pin_memory=(device == 'cuda'),
                    drop_last=len(ds) > args.batch)

    model = build_model(args, device)
    print(model)
    groups = optimizer_groups(model, args)
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, len(dl)))
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    log_path = os.path.join(args.out, 'train.log')
    log = open(log_path, 'w')
    log.write('epoch,iter,loss,box,cls,dfl,lr\n')
    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        sums, n_iter = None, 0
        t0 = time.time()
        for it, (imgs, attrs, batch) in enumerate(dl):
            if args.max_iters and it >= args.max_iters:
                break
            imgs = imgs.to(device, non_blocking=True)
            if attrs is not None:
                attrs = attrs.to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast('cuda', enabled=args.amp):
                loss, ld = model(imgs, attrs, batch)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(model.trainable_params()), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            vals = [float(loss.detach())] + [float(ld[k].detach()) for k in
                                             ('box_loss', 'cls_loss', 'dfl_loss')]
            sums = vals if sums is None else [a + b for a, b in zip(sums, vals)]
            n_iter += 1
            if it % 20 == 0:
                lr = opt.param_groups[0]['lr']
                log.write(f'{epoch},{it},{vals[0]:.5f},{vals[1]:.5f},{vals[2]:.5f},'
                          f'{vals[3]:.5f},{lr:.6f}\n')
                log.flush()
                print(f'  e{epoch:02d} it{it:03d} loss={vals[0]:.4f} '
                      f'box={vals[1]:.4f} cls={vals[2]:.3f} dfl={vals[3]:.4f}')

        if sums is not None:
            avg = [v / n_iter for v in sums]
            print(f'[ep {epoch:02d}] loss={avg[0]:.4f} box={avg[1]:.4f} '
                  f'cls={avg[2]:.3f} dfl={avg[3]:.4f} | {time.time()-t0:.1f}s')
        if (epoch + 1) % args.snapshot_every == 0 or epoch == args.epochs - 1:
            save_state(model, args, os.path.join(args.out, f'ep{epoch+1:03d}.pt'))

    save_state(model, args, os.path.join(args.out, 'last.pt'))
    log.close()
    print(f'[done] {time.time()-t_start:.0f}s | last.pt -> {args.out}')


if __name__ == '__main__':
    sys.exit(main())
