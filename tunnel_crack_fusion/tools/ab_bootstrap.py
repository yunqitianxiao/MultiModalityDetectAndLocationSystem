#!/usr/bin/env python3
"""A/B 两档连续 AP 差值的 bootstrap 区间。

逐帧推理后，用每帧的 (分数, 是否 TP) 做有放回重采样，重算全体 AP 之差。
同时给出「按连续块重采样」的结果：相邻帧高度相关，按帧独立重采样会低估方差，
块 bootstrap 更接近真实不确定性。

    python3 tools/ab_bootstrap.py --a runs/rgb/last.pt --b runs/rgb_pc/last.pt
"""

import argparse
import json

import numpy as np
import torch

from tunnel_tri import config as C
from tunnel_tri.data import list_frames, split_indices
from tunnel_tri.evaluate import (MATCH_IOU, continuous_ap, ground_truth, infer,
                                 load_model, match_one, voc11_ap)


def collect(ckpt, frames, root, device, batch):
    model, _ = load_model(ckpt, device)
    preds = infer(model, frames, root, device, batch)
    gts = ground_truth(frames, root)
    per_frame = []
    for p, (gb, gc) in zip(preds, gts):
        if len(p):
            pb, ps, pc = p[:, :4].numpy(), p[:, 4].numpy(), p[:, 5].long().numpy()
        else:
            pb, ps, pc = np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
        tp, _, n_gt = match_one(pb, ps, pc, gb, gc, MATCH_IOU)
        per_frame.append((ps, tp, n_gt))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per_frame


def pr_of(frame_list):
    if not frame_list:
        return np.zeros(0), np.zeros(0), 0
    s = np.concatenate([f[0] for f in frame_list])
    tp = np.concatenate([f[1] for f in frame_list])
    n_gt = int(sum(f[2] for f in frame_list))
    if len(s) == 0 or n_gt == 0:
        return np.zeros(0), np.zeros(0), n_gt
    o = np.argsort(-s, kind='stable')
    tp = tp[o]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1)
    return rec, prec, n_gt


def ap_pair(frame_list, fn):
    rec, prec, n_gt = pr_of(frame_list)
    return 0.0 if n_gt == 0 else fn(rec, prec)


def bootstrap(pa, pb, n_boot, block, seed):
    rng = np.random.default_rng(seed)
    N = len(pa)
    d_cont, d_voc = [], []
    for _ in range(n_boot):
        if block <= 1:
            idx = rng.integers(0, N, N)
        else:
            n_blk = int(np.ceil(N / block))
            starts = rng.integers(0, N, n_blk)
            idx = np.concatenate([(np.arange(s, s + block)) % N for s in starts])[:N]
        sa = [pa[i] for i in idx]
        sb = [pb[i] for i in idx]
        d_cont.append(ap_pair(sb, continuous_ap) - ap_pair(sa, continuous_ap))
        d_voc.append(ap_pair(sb, voc11_ap) - ap_pair(sa, voc11_ap))
    return np.array(d_cont), np.array(d_voc)


def summarize(name, d, obs):
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_pos = float((d > 0).mean())
    print(f'  {name}: Δ={obs:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  '
          f'P(Δ>0)={p_pos:.3f}')
    return {'delta': obs, 'ci_lo': float(lo), 'ci_hi': float(hi), 'p_pos': p_pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='A 档 ckpt（rgb）')
    ap.add_argument('--b', required=True, help='B 档 ckpt（rgb+pc）')
    ap.add_argument('--data', default=C.DATA_ROOT)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--block', type=int, default=10, help='块 bootstrap 的块长')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames = list_frames(args.data)
    val_idx, _ = split_indices(frames, 0.2, 'sequential', args.seed)
    print(f'val {len(val_idx)} 帧 | device={device}')

    print(f'[A] {args.a}')
    pa = collect(args.a, val_idx, args.data, device, args.batch)
    print(f'[B] {args.b}')
    pb = collect(args.b, val_idx, args.data, device, args.batch)
    assert len(pa) == len(pb)

    res = {'a': args.a, 'b': args.b, 'n_val': len(val_idx), 'n_boot': args.boot}
    ci_iid, dv_iid = bootstrap(pa, pb, args.boot, 1, args.seed)
    ci_blk, dv_blk = bootstrap(pa, pb, args.boot, args.block, args.seed)

    for tag, fn, d_iid, d_blk in (('continuous', continuous_ap, ci_iid, ci_blk),
                                  ('voc11', voc11_ap, dv_iid, dv_blk)):
        a_val = ap_pair(pa, fn)
        b_val = ap_pair(pb, fn)
        obs = b_val - a_val
        print(f'[{tag}] A={a_val:.4f} B={b_val:.4f}')
        res[tag] = {'A': a_val, 'B': b_val}
        res[tag]['iid'] = summarize('按帧重采样', d_iid, obs)
        res[tag]['block'] = summarize(f'块重采样(块={args.block})', d_blk, obs)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f'[save] {args.out}')


if __name__ == '__main__':
    main()
