#!/usr/bin/env python3
"""统一尺子评测：连续插值 AP + VOC11 AP + 最优工作点。

用法:
    python3 -m tunnel_tri.evaluate --ckpt runs/rgb/last.pt
    python3 -m tunnel_tri.evaluate --ckpt runs/rgb_pc/last.pt --out runs/rgb_pc/eval.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from . import config as C
from .data import list_frames, load_sample, split_indices
from .model import DualModalYOLO

CONF_MIN, NMS_IOU, MAX_DET, MATCH_IOU = 0.001, 0.7, 300, 0.5


def parse_args():
    p = argparse.ArgumentParser(description='双模态裂缝检测评测')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', default='')
    p.add_argument('--out', default='')
    p.add_argument('--split', default='', choices=('', 'sequential', 'random'))
    p.add_argument('--val-ratio', type=float, default=0.0)
    p.add_argument('--batch', type=int, default=8)
    return p.parse_args()


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ck.get('args') or {}
    model = DualModalYOLO(ckpt=a.get('yolo') or C.YOLO_CKPT, nc=ck.get('nc', C.NC),
                          modalities=ck.get('modalities', 'rgb+pc'), device=device)
    miss, unexp = model.load_state_dict(ck['model'], strict=False)
    miss = [m for m in miss if not m.startswith('image_encoder.model.')]
    if miss:
        print(f'⚠ 缺键 {len(miss)}: {miss[:5]}')
    if unexp:
        print(f'⚠ 多余键 {len(unexp)}: {unexp[:5]}')
    model.eval()
    return model, ck


@torch.no_grad()
def infer(model, frames, root, device, batch_size=8):
    from ultralytics.utils.nms import non_max_suppression
    use_point = model.use_point
    preds = []
    for bi in range(0, len(frames), batch_size):
        chunk = frames[bi:bi + batch_size]
        samples = [load_sample(i, root, with_attr=use_point) for i in chunk]
        imgs = torch.from_numpy(np.stack([s[0] for s in samples])).to(device)
        attrs = (torch.from_numpy(np.stack([s[1] for s in samples])).to(device)
                 if use_point else None)
        out = model(imgs, attrs)
        grid = out[0] if isinstance(out, tuple) else out
        nms = non_max_suppression(grid.detach().cpu(), conf_thres=CONF_MIN,
                                  iou_thres=NMS_IOU, max_det=MAX_DET)
        preds.extend([d if len(d) else torch.zeros((0, 6)) for d in nms])
    return preds


def ground_truth(frames, root):
    gts = []
    for i in frames:
        _, _, cls, xyxy = load_sample(i, root, with_attr=False)
        gts.append((xyxy.astype(np.float32), cls.astype(np.int64)))
    return gts


def match_one(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, iou_thres):
    from torchvision.ops import box_iou
    tp = np.zeros(len(pred_boxes), dtype=bool)
    used = np.zeros(len(gt_boxes), dtype=bool)
    for k in np.argsort(-pred_scores):
        cand = np.where(gt_cls == pred_cls[k])[0]
        if len(cand) == 0:
            continue
        iou = box_iou(torch.from_numpy(pred_boxes[k:k + 1]),
                      torch.from_numpy(gt_boxes[cand])).numpy()[0]
        j = int(np.argmax(iou))
        if iou[j] >= iou_thres and not used[cand[j]]:
            tp[k] = True
            used[cand[j]] = True
    return tp, int((~used).sum()), len(gt_boxes)


def per_class_pr(frames_preds, frames_gt, nc, iou_thres):
    det = {c: {'s': [], 'tp': [], 'n_gt': 0} for c in range(nc)}
    for preds, (gt_boxes, gt_cls) in zip(frames_preds, frames_gt):
        if len(preds):
            pb = preds[:, :4].numpy()
            ps = preds[:, 4].numpy()
            pc = preds[:, 5].long().numpy()
        else:
            pb, ps, pc = np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)
        tp, _, _ = match_one(pb, ps, pc, gt_boxes, gt_cls, iou_thres)
        for c in range(nc):
            det[c]['n_gt'] += int((gt_cls == c).sum())
            det[c]['s'].extend(ps[pc == c].tolist())
            det[c]['tp'].extend(tp[pc == c].tolist())
    return det


def continuous_ap(recall, precision):
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def voc11_ap(recall, precision):
    ap = 0.0
    for t in np.arange(0.0, 1.0 + 1e-9, 0.1):
        m = recall >= t
        if m.any():
            ap += precision[m].max() / 11.0
    return float(ap)


def score(frames_preds, frames_gt, nc=1):
    det = per_class_pr(frames_preds, frames_gt, nc, MATCH_IOU)
    res = {'classes': {}, 'n_frames': len(frames_preds)}
    conts, vocs = [], []
    for c in range(nc):
        d = det[c]
        if not d['s'] or d['n_gt'] == 0:
            res['classes'][c] = {'n_gt': d['n_gt'], 'n_det': len(d['s']),
                                 'ap_continuous': None, 'ap_voc11': None}
            continue
        s = np.array(d['s'])
        tp = np.array(d['tp'])
        order = np.argsort(-s)
        s, tp = s[order], tp[order]
        cs = np.concatenate([[0.0], np.cumsum(tp)])
        cfp = np.concatenate([[0.0], np.cumsum(1 - tp)])
        recall = cs / d['n_gt']
        precision = cs / (cs + cfp).clip(min=1)
        th = np.concatenate([[s[0] + 1e-6], s])
        precision, recall = precision[1:], recall[1:]
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        j = int(np.argmax(f1))
        ac, av = continuous_ap(recall, precision), voc11_ap(recall, precision)
        conts.append(ac)
        vocs.append(av)
        res['classes'][c] = {
            'n_gt': d['n_gt'], 'n_det': len(d['s']), 'n_tp': int(tp.sum()),
            'ap_continuous': round(ac, 4), 'ap_voc11': round(av, 4),
            'opt_thresh': round(float(th[j]), 6),
            'opt_precision': round(float(precision[j]), 4),
            'opt_recall': round(float(recall[j]), 4),
        }
    res['mAP_continuous'] = round(float(np.mean(conts)), 4) if conts else None
    res['mAP_voc11'] = round(float(np.mean(vocs)), 4) if vocs else None
    return res


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ck = load_model(args.ckpt, device)
    a = ck.get('args') or {}
    root = args.data or a.get('data') or C.DATA_ROOT

    split = args.split or a.get('split', 'sequential')
    val_ratio = args.val_ratio or a.get('val_ratio', 0.2)
    seed = a.get('seed', 42)

    split_json = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), 'split.json')
    if os.path.exists(split_json) and not args.split and not args.val_ratio:
        with open(split_json) as f:
            val_idx = json.load(f)['val_idx']
    else:
        val_idx, _ = split_indices(list_frames(root), val_ratio, split, seed)

    print(f'[eval] {args.ckpt} | {model.modalities} | val {len(val_idx)} 帧 | '
          f'split={split} | device {device}')
    preds = infer(model, val_idx, root, device, args.batch)
    gts = ground_truth(val_idx, root)
    res = score(preds, gts, model.nc)
    res['ckpt'] = args.ckpt
    res['modalities'] = model.modalities
    res['split'] = split
    res['n_val'] = len(val_idx)
    res['zero_reflec'] = C.ZERO_REFLEC
    res['fixed_attr'] = os.environ.get('TUNNEL_FIXED_ATTR')

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), 'eval.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    c0 = res['classes'][0]
    print(f"\nmAP(连续)={res['mAP_continuous']}  mAP(VOC11)={res['mAP_voc11']}")
    print(f"n_gt={c0['n_gt']}  n_det={c0.get('n_det')}  TP={c0.get('n_tp')}  "
          f"工作点 th={c0.get('opt_thresh')} P={c0.get('opt_precision')} R={c0.get('opt_recall')}")
    print(f'[save] {out}')


if __name__ == '__main__':
    sys.exit(main())
