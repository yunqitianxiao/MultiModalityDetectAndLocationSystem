#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pro/3 多模态融合训练 —— 5×V100S DDP 版 (torchrun --nproc_per_node=5)
================================================================
与 train_fusion.py 同构, 仅把数据流/优化改为分布式:
  - torchrun 设 RANK/WORLD_SIZE/LOCAL_RANK; nccl 初始化; cuda:<RANK>。
  - 每 rank 分到 frames[rank::world] (先同种子 shuffle, 各 rank 切片互补不重).
  - frames 构建时已按分支目标类过滤 -> 每帧必有框 -> collate_fusion 从不返回 None,
    保证各 rank 每 epoch 的 backward 步数一致(DDP 不挂起)。
  - DDP(wrapper, find_unused_parameters=True): 编码器全冻, fusion/nav/adapter/detect 可训。
  - rank 0 计时每 epoch wall-clock -> 报 ETA。

【用法 (server, 5×V100)】
    cd /root/autodl-tmp/pro3
    torchrun --nproc_per_node=5 --master_port=29501 train_fusion_ddp.py \
        --branch 轨道 --epochs 30 --imgsz 640 --batch 2 --save runs/fusion_ddp/best.pt

【注意】三支编码器+best 权重+pcdata 都传 /root/autodl-tmp/pro3 下; 依赖 train_fusion 的
build_model/load_sim_frame/collate_fusion(同目录 import)。V100=sm_70 需 cu121 的 torch。
"""
from __future__ import annotations

import argparse, math, os, random, sys, time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

_P = os.path.dirname(os.path.abspath(__file__))
if _P not in sys.path:
    sys.path.insert(0, _P)
import train_fusion as T     # 复用 build_model / load_sim_frame / collate_fusion / _BRANCHES


def setup(rank: int, world: int):
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    # 同种子 -> 各 rank 对 frames 做同一 shuffle, 随后按 rank 切片互补不重、全覆盖。
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)


def steal_frames(branch: str, roots: list[str], sim_cls: dict):
    """按训练集目录收集含该分支目标类的帧 (root, stem)。由各 rank 各自构建(同种子但只选自己切片)。"""
    frames = []
    for root in roots:
        for stem in T.resolve_label_paths(root):
            lbl = os.path.join(root, 'labels', f'{stem}.txt')
            if not os.path.isfile(lbl) or os.path.getsize(lbl) == 0:
                continue
            try:
                lab = np.loadtxt(lbl, dtype=np.float32).reshape(-1, 5)
            except Exception:
                continue
            if any(int(l[0]) in sim_cls for l in lab):
                frames.append((root, stem))
    return frames


def main():
    ap = argparse.ArgumentParser(description='5×V100 DDP 多模态融合训练')
    ap.add_argument('--branch', default='轨道', choices=list(T._BRANCHES))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--batch', type=int, default=2)         # 每 rank 批大小 (effective = batch*world)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out', default=os.path.join(_P, 'runs', 'fusion_ddp'))
    ap.add_argument('--val-every', type=int, default=1)      # 每 N epoch 在 test_set 上算 val loss
    ap.add_argument('--max-val', type=int, default=64)       # val 每 rank 上最多帧数(控制时长)
    ap.add_argument('--val-roots', default=None, help='拍分号隔开; 缺省=轨道验证集 output_rail_s2')
    ap.add_argument('--test-roots', default=None, help='拍分号隔开(最终评测保留); 缺省=分支 test_set(轨道=output_rail_s1)')
    ap.add_argument('--save', default=None)
    ap.add_argument('--pcl-weight', default=None)
    args = ap.parse_args()

    rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
    setup(rank, world)
    device = torch.device('cuda', rank)

    conf = T._BRANCHES[args.branch]; sim_cls = conf['sim_cls']; nc = conf['nc']
    roots = [os.path.join(_P, 'pcdata', x) for x in conf['train_sets']]
    if args.val_roots is None:
        val_roots = [os.path.join(_P, 'pcdata', 'output_rail_s2')]        # 轨道验证集(s2)
    else:
        val_roots = [os.path.join(_P, x) for x in args.val_roots.split(';')]
    if args.test_roots is None:
        test_roots = [os.path.join(_P, 'pcdata', conf['test_set'])]       # 轨道测试集(s1), 最终评测用
    else:
        test_roots = [os.path.join(_P, x) for x in args.test_roots.split(';')]

    if rank == 0:
        print(f"[数据] branch={args.branch} nc={nc} "
              f"train=({', '.join(roots)}) val=({', '.join(val_roots)}) test=({', '.join(test_roots)})")

    # ---- train frames (各 rank 同 shuffle, 切片互补) ----
    frames = steal_frames(args.branch, roots, sim_cls)
    random.shuffle(frames)
    pool = frames[: (len(frames) // world) * world]          # 丢掉不足以均分的尾部
    chunk = len(pool) // world
    mine = pool[rank * chunk:(rank + 1) * chunk]             # 本 rank 帧(全部含目标框)
    if rank == 0:
        print(f"[数据] 总帧={len(frames)} 均补齐={len(pool)} 每rank={chunk} 步={math.ceil(chunk/args.batch)}")

    # ---- val frames (同种子切片; val_loss 用验证集 s2) ----
    vframes = steal_frames(args.branch, val_roots, sim_cls)
    random.shuffle(vframes)
    vframes = vframes[:args.max_val]
    vmine = vframes[rank::world]

    # ---- 构建融合模型 (各 rank 各自 load, 权重同) ----
    wrapper, nc = T.build_model(args.branch, img_imgsz=args.imgsz, max_tokens=args.max_tokens,
                                device=device, pcl_weight=args.pcl_weight)
    wrapper.train(); wrapper.model.model[-1].train()
    ddp = DDP(wrapper, device_ids=[rank], find_unused_parameters=True)

    det = wrapper.model.model[-1]
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    if rank == 0:
        n_t = sum(p.numel() for p in trainable) / 1e6
        n_f = sum(p.numel() for p in wrapper.parameters()) / 1e6 - n_t
        print(f"[参数] 冻结={n_f:.1f}M 训练(fusion+nav+adapter+detect)={n_t:.1f}M")

    opt = torch.optim.AdamW([
        {'params': [p for p in wrapper.fusion.parameters() if p.requires_grad], 'lr': args.lr},
        {'params': [p for p in wrapper.nav_enc.parameters() if p.requires_grad], 'lr': args.lr},
        {'params': [p for p in wrapper.adapters.parameters() if p.requires_grad], 'lr': args.lr},
        {'params': [p for p in det.parameters() if p.requires_grad], 'lr': args.lr * 0.1},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: max(0.0, 0.5 * (1.0 + math.cos(math.pi * ep / args.epochs))))

    os.makedirs(args.out, exist_ok=True)
    if rank == 0:
        log_path = os.path.join(args.out, f'fusion_{args.branch}_ddp.log')
        flog = open(log_path, 'w'); flog.write('epoch,train_loss,val_loss,epoch_s\n')

    def step(idxs, grad):
        samples = [T.load_sim_frame(r, s, args.imgsz, sim_cls, nc) for (r, s) in idxs]
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        img, attr, nav, batch = T.collate_fusion(samples, args.imgsz)
        if img is None:
            return None
        img = img.to(device); attr = attr.to(device); nav = nav.to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(grad):
            loss, _ = ddp(img, attr, nav, batch)
            if grad:
                opt.zero_grad(); loss.backward(); opt.step()
            return float(loss.detach())

    dist.barrier()
    torch.cuda.synchronize()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tot = 0.0; cnt = 0
        for bi in range(0, len(mine), args.batch):
            ls = step(mine[bi:bi + args.batch], True)
            if ls is None:
                continue
            tot += ls; cnt += 1
        sched.step()

        # val loss (test_set)
        vsum = vt = 0.0
        if args.val_every and ep % args.val_every == 0:
            with torch.no_grad():
                for bi in range(0, len(vmine), args.batch):
                    ls = step(vmine[bi:bi + args.batch], False)
                    if ls is None:
                        continue
                    vsum += ls; vt += 1
        # 聚合 train/val loss 到 rank0
        vals = torch.tensor([tot, cnt, vsum, vt], device=device)
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        tr = vals[0].item() / max(1, vals[1].item())
        vl = vals[2].item() / max(1, vals[3].item())
        torch.cuda.synchronize()
        ep_s = time.time() - t0
        if rank == 0:
            line = f"{ep},{tr:.4f},{vl:.4f},{ep_s:.2f}"
            flog.write(line + '\n'); flog.flush()
            print(f"[ep{ep:02d}/{args.epochs}] train={tr:.4f} val={vl:.4f}  {ep_s:.1f}s/epoch", flush=True)

    if rank == 0:
        flog.close()
        if args.save:
            torch.save(wrapper.state_dict(), args.save)
            print(f"[save] {args.save}")
        print("TRAIN_FUSION_DDP_DONE")


if __name__ == '__main__':
    main()
