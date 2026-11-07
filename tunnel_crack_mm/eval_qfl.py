#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Range-QFL vs 原版 v8 loss —— 同尺子对照
================================================================
不只报 mAP: 本项目 mAP 已接近饱和(0.957), 增益会被噪声吃掉。真正要对准的是
两个**实测失败模式**:

  A. 分数尺度异常: 原模型全部框 conf ∈ [0.010, 0.022], AP 0.9 却要手调
     conf_thr=0.0057 才出框。QFL 用 TAL 对齐度作软标签, 理论上把分数推回
     "分数≈定位质量"的可解释区间。判据: 最高分 / p99.9 / opt_thresh 占最高分比例。

  B. 远处小目标漏检: 检测数随里程单调衰减 13->0。判据: 按 GT 面积三分位分桶,
     报各桶 recall@最优工作点。若只有远处桶升、近处持平 => 尺度加权起作用。

评测集给两套:
  heldout  = 帧 0..123 (train_assist 训练时留出的 val, 与训练帧 124..623 不相交)
  shuffled = seed42 随机 124 帧 (compare_branches.py 用的那套, 与历史数字对齐,
             但其中约 100 帧在训练范围内 => 绝对值偏乐观, 只看 A/B 差值)

用法: python eval_qfl.py
"""
import json
import os
import sys

_IR = os.environ.get(
    'IR_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inspection_robot')),
)
if _IR not in sys.path:
    sys.path.insert(0, _IR)

from fix_nms import patch_torchvision_nms  # noqa: E402
patch_torchvision_nms()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fusion.evaluate_pr import (  # noqa: E402
    ap_from_curve, curve_for_class, load_model, per_class_pr, train_val_split,
)
from compare_branches import (  # noqa: E402
    DATA, IOU_THRES, NMS_IOU, CONF_MIN, MAX_DET, gt_from_fusion_val,
    infer_assist, continuous_ap,
)

BASE = os.path.join(_IR, 'runs/assist_crack/last.pt')
QFL = os.path.join(_IR, 'runs/assist_crack_qfl/last.pt')

VAL_RATIO, SEED = 0.2, 42


def score_branch(preds, gts, nc=1):
    det = per_class_pr(preds, gts, nc, IOU_THRES)
    th, prec, rec = curve_for_class(det[0])
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    j = int(np.argmax(f1))
    return {
        'ap_voc11': round(float(ap_from_curve(rec, prec)), 4),
        'ap_cont': round(float(continuous_ap(rec, prec)), 4),
        'opt_thresh': round(float(th[j]), 5),
        'opt_precision': round(float(prec[j]), 4),
        'opt_recall': round(float(rec[j]), 4),
        'det': det[0],
        'conf_at_opt': th[j],
    }


def conf_stats(preds):
    s = np.concatenate([p[:, 4].numpy() for p in preds if len(p)]) if any(len(p) for p in preds) \
        else np.zeros(1)
    return {
        'n_box': int(len(s)),
        'max': round(float(s.max()), 4),
        'p99_9': round(float(np.percentile(s, 99.9)), 4),
        'p99': round(float(np.percentile(s, 99)), 4),
        'p50': round(float(np.percentile(s, 50)), 4),
        'mean': round(float(s.mean()), 5),
    }


def _match_pred_to_gt(pb, ps, gt, iou_thres):
    """贪心一对一匹配, 返回每个 pred 匹配到的 GT 索引 (-1 = 未匹配)。
    evaluate_pr.match_one 只回 tp 布尔, 不告诉你匹配到哪个 GT, 分桶需要这个索引。"""
    from torchvision.ops import box_iou
    out = np.full(len(pb), -1, dtype=np.int64)
    used = np.zeros(len(gt), dtype=bool)
    if len(pb) == 0 or len(gt) == 0:
        return out
    iou = box_iou(torch.from_numpy(pb), torch.from_numpy(gt)).numpy()
    for k in np.argsort(-ps):
        row = iou[k].copy()
        row[used] = -1.0
        j = int(np.argmax(row))
        if row[j] >= iou_thres:
            out[k] = j
            used[j] = True
    return out


def bucket_recall(preds, gts, thr, n_bucket=3):
    """按 GT 面积三分位分桶, 报 recall@thr (IoU>=0.5, 贪心一对一)。"""
    areas = np.array([w * h for gb, _ in gts for (x1, y1, x2, y2) in gb
                      for w, h in [(x2 - x1, y2 - y1)]])
    if len(areas) == 0:
        return {}
    cuts = np.quantile(areas, [1 / n_bucket, 2 / n_bucket])
    names = ['small', 'mid', 'large']
    hit = {n: 0 for n in names}
    tot = {n: 0 for n in names}

    def bucket_of(a):
        return names[0] if a <= cuts[0] else (names[1] if a <= cuts[1] else names[2])

    for pred, (gb, _gc) in zip(preds, gts):
        areas_f = np.array([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in gb])
        for a in areas_f:
            tot[bucket_of(a)] += 1
        if len(pred) == 0 or len(gb) == 0:
            continue
        keep = pred[:, 4].numpy() >= thr
        pb = pred[keep, :4].numpy()
        ps = pred[keep, 4].numpy()
        if len(pb) == 0:
            continue
        mi = _match_pred_to_gt(pb, ps, gb.astype(np.float32), IOU_THRES)
        for j in mi[mi >= 0]:
            hit[bucket_of(areas_f[j])] += 1
    res = {n: {'recall': round(hit[n] / tot[n], 4) if tot[n] else None,
               'n_gt': int(tot[n])} for n in names}
    res['_cuts_px2'] = [round(float(c), 1) for c in cuts]
    return res


def run(tag, ckpt, val_idx, gts, device):
    model, ck = load_model(ckpt, device)
    preds = infer_assist(model, val_idx, device)
    s = score_branch(preds, gts)
    cs = conf_stats(preds)
    br = bucket_recall(preds, gts, s['conf_at_opt'])
    n_det = len(s['det']['s'])
    print(f'\n[{tag}]  {len(val_idx)} 帧  n_gt={s["det"]["n_gt"]}  n_det={n_det}')
    print(f'  AP: VOC11={s["ap_voc11"]:.4f}  连续={s["ap_cont"]:.4f}')
    print(f'  工作点: thr={s["opt_thresh"]:.5f}  P={s["opt_precision"]:.3f} R={s["opt_recall"]:.3f}')
    print(f'  分数: max={cs["max"]:.4f} p99.9={cs["p99_9"]:.4f} p99={cs["p99"]:.4f} '
          f'median={cs["p50"]:.4f} | thr/max={s["opt_thresh"]/max(cs["max"],1e-9):.3f}')
    print(f'  分桶召回 (面积切点 px^2 {br["_cuts_px2"]}):')
    for n in ('small', 'mid', 'large'):
        print(f'    {n:5s} R={br[n]["recall"]}  (n_gt={br[n]["n_gt"]})')
    del model
    torch.cuda.empty_cache()
    return {'ap_voc11': s['ap_voc11'], 'ap_cont': s['ap_cont'],
            'opt_thresh': s['opt_thresh'], 'opt_p': s['opt_precision'],
            'opt_r': s['opt_recall'], 'conf': cs, 'bucket': br,
            'n_gt': s['det']['n_gt'], 'n_det': len(s['det']['s'])}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_img = len([f for f in os.listdir(os.path.join(DATA, 'images')) if f.endswith('.jpg')])
    n_val = max(1, int(n_img * VAL_RATIO))
    splits = {
        'heldout': list(range(n_val)),                       # train_assist 的 val 段
        'shuffled': train_val_split(n_img, VAL_RATIO, SEED)[0],  # compare_branches 的那套
    }
    out = {}
    for split_name, val_idx in splits.items():
        print('\n' + '=' * 70)
        print(f'评测集 {split_name}: {len(val_idx)} 帧')
        print('=' * 70)
        gts = gt_from_fusion_val(val_idx)
        out[split_name] = {}
        for tag, ck in (('原版 v8', BASE), ('RangeQFL', QFL)):
            if not os.path.exists(ck):
                print(f'\n[{tag}] 缺权重 {ck}, 跳过')
                continue
            out[split_name][tag] = run(tag, ck, val_idx, gts, device)

    dst = os.path.join(os.path.dirname(__file__), 'runs', 'eval_qfl.json')
    with open(dst, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=lambda o: None)
    print(f'\n[done] {dst}')


if __name__ == '__main__':
    main()
