#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断: 图像单支在 crack_1cls_big/val 上, 逐帧统计 gt/pred/TP, 找出拖低 recall 的帧。"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
_IR = os.environ.get('IR_ROOT', os.path.abspath(os.path.join(_HERE, '..', 'inspection_robot')))
sys.path.insert(0, _IR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fix_nms import patch_torchvision_nms  # noqa: E402
patch_torchvision_nms()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.ops import box_iou  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402

from fusion.config import YOLO_SIZE  # noqa: E402
from fusion.pointcloud_assist_detector import letterbox_boxes  # noqa: E402
from compare_branches import continuous_ap  # noqa: E402
from fusion.evaluate_pr import ap_from_curve, curve_for_class, per_class_pr  # noqa: E402

CLS = os.path.join(_IR, 'data_gen', 'crack_1cls_big')
BEST = os.path.join(_HERE, 'runs', 'yolo11l_crack_1cls', 'weights', 'best.pt')


def prep(fn):
    img = cv2.imread(os.path.join(CLS, 'images/val', fn))
    H, W = img.shape[:2]
    r = min(YOLO_SIZE / H, YOLO_SIZE / W)
    nw, nh = round(W * r), round(H * r)
    dw, dh = (YOLO_SIZE - nw) // 2, (YOLO_SIZE - nh) // 2
    canvas = np.zeros((YOLO_SIZE, YOLO_SIZE, 3), np.uint8) + 114
    canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(img, (nw, nh))
    x = np.transpose(canvas.astype(np.float32) / 255., (2, 0, 1))
    lp = os.path.join(CLS, 'labels/val', os.path.splitext(fn)[0] + '.txt')
    lab = (np.loadtxt(lp, dtype=np.float32).reshape(-1, 5)
           if os.path.getsize(lp) > 0 else np.empty((0, 5)))
    _, gx = letterbox_boxes(lab, (H, W))
    return x, gx.astype(np.float32)


def frame_tp(pred_boxes, gt_boxes, thr=0.5):
    """高分优先贪心匹配, 返回 (n_tp, max_iou_of_unmatched_gt)"""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0, 0.0
    iou = box_iou(torch.from_numpy(pred_boxes), torch.from_numpy(gt_boxes)).numpy()
    used_g = np.zeros(len(gt_boxes), bool)
    tp = 0
    for k in range(len(pred_boxes)):          # pred 已按 conf 降序(NMS 输出)
        j = int(np.argmax(iou[k]))
        if iou[k, j] >= thr and not used_g[j]:
            tp += 1
            used_g[j] = True
    leftover = iou[:, ~used_g]
    mx = float(leftover.max()) if leftover.size else 0.0
    return tp, mx


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    y = YOLO(BEST).model.float().eval().to(dev)
    files = sorted(f for f in os.listdir(os.path.join(CLS, 'images/val')) if f.endswith('.jpg'))
    preds, gts, rows = [], [], []
    B = 8
    for bi in range(0, len(files), B):
        xs, gs = [], []
        for fn in files[bi:bi + B]:
            x, gx = prep(fn)
            xs.append(x)
            gs.append(gx)
        with torch.no_grad():
            out = y(torch.from_numpy(np.stack(xs)).to(dev))
        grid = out[0] if isinstance(out, (tuple, list)) else out
        d = non_max_suppression(grid.detach().cpu(), conf_thres=0.001, iou_thres=0.45, max_det=300)
        for k, pred in enumerate(d):
            pred = pred if len(pred) else torch.zeros((0, 6))
            preds.append(pred)
            gts.append((gs[k], np.zeros(len(gs[k]), np.int64)))
            rows.append((files[bi + k], len(gs[k]), len(pred)))
    det = per_class_pr(preds, gts, 1, 0.5)
    n_gt, n_det, n_tp = det[0]['n_gt'], len(det[0]['s']), int(np.sum(det[0]['tp']))
    print(f'[总] gt={n_gt} pred={n_det} TP={n_tp} FN={n_gt - n_tp} FP={n_det - n_tp}')
    th, prec, rec = curve_for_class(det[0])
    print(f'[AP] recall 上限={rec.max():.4f}  AP_VOC11={ap_from_curve(rec, prec):.4f}  '
          f'AP_cont={continuous_ap(rec, prec):.4f}')

    # 逐帧找短板: n_tp < n_gt 的帧
    print('\n[短板帧] 有 GT 却没全匹配上的 (前 12):')
    shown = 0
    for (fn, ngt, _), pred, (gx, _) in zip(rows, preds, gts):
        if ngt == 0:
            continue
        ntp, mx = frame_tp(pred[:, :4].numpy(), gx)
        if ntp < ngt and shown < 12:
            print(f'  {fn}: gt={ngt} pred={len(pred)} tp={ntp} 未匹配GT的最大IoU={mx:.3f} '
                  f'最高conf={float(pred[0,4]) if len(pred) else 0:.3f}')
            shown += 1


if __name__ == '__main__':
    main()
