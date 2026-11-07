"""
开放集/不确定性门控 (precision 杠杆) —— 独立于置信度阈值的又一道精度闸
================================================================
检测头的置信度只反映"最像哪一类", 不含"类间竞争程度"。一个框若类分布很平
(高熵: 两个类近乎打平), 哪怕 max 置信度不低, 也更容易是误检(类别算错)。用
逐锚点类熵 H = -Σ p_c log p_c 做第二道闸, 把高熵(类不确定)的框剔除, 可独立
于置信度阈值抬高 precision。

本模块自带 CPU 端解码 + 逐锚点熵 + 类无关 NMS(绕开本机 torchvision CUDA nms
缺失问题), 产出 (xyxy, conf, cls, entropy)。在固定 conf 阈值下对比"只看 conf"
vs"conf 且 entropy≤E"的 PR, 量化熵门控是否带来 precision/mAP 增益。

诚实说明: 熵门控只在该置信度偏低、类竞争强的低数据场景有效; 在大数据/强分类器
下收益趋零。它是有条件的 precision 优化, 不是无脑保证。

用法:
    .venv/bin/python fusion/ood_gate.py --ckpt runs/v2_point80/last.pt --out runs/ood
产物: {out}/ood_gate_pr.json —— conf-only 与 conf+entropy 的 mAP@0.5 对比
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torchvision.ops import box_convert, nms

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from fusion.config import OUTPUT_V7  # noqa: E402
from fusion.evaluate_pr import (  # noqa: E402
    load_model, train_val_split, per_class_pr, curve_for_class, ap_from_curve,
)
from fusion.pointcloud_assist_detector import collate_batch, load_sample  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='熵门控 precision 增益量化')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', default=OUTPUT_V7)
    p.add_argument('--conf-min', type=float, default=0.001)
    p.add_argument('--conf-max', type=float, default=0.9, help='熵门控对比的固定置信度阈值')
    p.add_argument('--iou-thres', type=float, default=0.5)
    p.add_argument('--out', default='runs/ood')
    return p.parse_args()


@torch.no_grad()
def infer_with_entropy(model, frames, root, device, conf_min):
    """逐帧 (M, 7): xyxy,conf,cls,entropy。CPU 解码+类无关 NMS。"""
    all_preds = []
    for bi in range(0, len(frames), 8):
        batch = frames[bi:bi + 8]
        imgs, attrmaps, poses, _ = collate_batch([load_sample(i, root=root) for i in batch])
        out = model(imgs.to(device), attrmaps.to(device), poses.to(device))
        G = out[0] if isinstance(out, tuple) else out
        G = G.detach().cpu()
        B, nc, N = G.shape
        box = box_convert(G[:, :4].transpose(1, 2).contiguous(), 'xywh', 'xyxy')   # (B,N,4)
        prob = G[:, 4:].transpose(1, 2).clamp(min=1e-7)                            # (B,N,nc)
        conf, cls = prob.max(-1)
        ent = -(prob * prob.log()).sum(-1)                                          # 类熵
        for b in range(B):
            k = conf[b] >= conf_min
            xy = box[b][k]; sc = conf[b][k]; cl = cls[b][k]; en = ent[b][k]
            keep = nms(xy, sc, 0.45)                                                  # 类无关
            out_b = torch.zeros((len(keep), 7))
            out_b[:, :4] = xy[keep]; out_b[:, 4] = sc[keep]
            out_b[:, 5] = cl[keep].float(); out_b[:, 6] = en[keep]
            all_preds.append(out_b.numpy())
    return all_preds


def evaluate(preds, frames_gt, nc, iou_thres, max_conf, max_ent):
    """在 conf<=max_conf 且 (max_ent>=0 则再加 entropy<=max_ent) 下算 mAP。"""
    gated = []
    for p in preds:
        keep = np.ones(len(p), bool)
        if len(p):
            keep &= p[:, 4] <= max_conf
            if max_ent >= 0:
                keep &= p[:, 6] <= max_ent
        gated.append(p[keep])
    det = per_class_pr(gated, frames_gt, nc, iou_thres)
    aps = []
    for c in range(nc):
        cur = curve_for_class(det[c])
        if cur is not None:
            aps.append(ap_from_curve(*cur[1:]))
    return round(float(np.mean(aps)), 4) if aps else None


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ck = load_model(args.ckpt, device)
    nc = model.nc
    n_img = len([f for f in os.listdir(os.path.join(args.data, 'images')) if f.endswith('.jpg')])
    val_idx, _ = train_val_split(n_img, 0.2, 42)
    print(f"[ood] {os.path.basename(args.ckpt)} | val={len(val_idx)} 帧")
    with torch.no_grad():
        preds = infer_with_entropy(model, val_idx, args.data, device, args.conf_min)
    n_det = sum(len(p) for p in preds)
    print(f"[ood] 解码+类无关NMS 后 {n_det} 框 (conf<={args.conf_max}, 含熵信息)")

    frames_gt = []
    for i in val_idx:
        _, _, _, cls, xyxy = load_sample(i, root=args.data)
        frames_gt.append((xyxy.astype(np.float32), cls.astype(np.int64)))

    res = {
        'conf_only': evaluate(preds, frames_gt, nc, args.iou_thres, args.conf_max, -1),
        'conf+entropy@0.7': evaluate(preds, frames_gt, nc, args.iou_thres, args.conf_max, 0.7),
        'conf+entropy@0.5': evaluate(preds, frames_gt, nc, args.iou_thres, args.conf_max, 0.5),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'ood_gate_pr.json'), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n=== 熵门控精度对比 (conf<= {args.conf_max}) ===")
    for k, v in res.items():
        print(f"  {k:22s} mAP@0.5 = {v}")
    print(f"[save] {os.path.join(args.out, 'ood_gate_pr.json')}")


if __name__ == '__main__':
    main()
