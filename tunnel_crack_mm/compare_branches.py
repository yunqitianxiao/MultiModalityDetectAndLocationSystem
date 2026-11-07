#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一尺子对比: 图像单支 vs 融合 vs 无点云
================================================================
动机: 之前的结论把"融合 0.8164 vs 图像单支 0.9742"的 0.16 差距归给输入分辨率。
但 best.pt 的 args.yaml 显示它本来就是 --imgsz 640 训练的, 分辨率这条不成立。
真正混在一起的是: (a) AP 算法不同(融合用 VOC 11 点插值, 单支用 ultralytics
连续插值); (b) 评测集不同(融合=output_crack_big seed42 随机 124 帧, 单支=
crack_1cls_big 顺序划分的 94 帧); (c) 训练方式不同(单支 freeze=10 即 neck+head
全微调, 融合冻结主干+随机 scale_proj/head_proj/新 Detect 头)。

本脚本把所有分支放到同一评测集 + 同一 AP 算法下重测, 分离"尺子/集"造成的假差
与真实差。四种组合:
  1. 图像单支 @ 融合 val 集(核心可比)
  2. 融合     @ 融合 val 集(应复现 evaluate_pr 的 0.8164)
  3. 无点云   @ 融合 val 集(应复现 0.8161)
  4. 图像单支 @ 单支自己的 val 集(帧 530-623, 看它"主场"数字)

每种都同时给 VOC 11 点插值 AP 与连续插值 AP, 让尺子差异可见。

用法:
  python compare_branches.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_IR = os.environ.get('IR_ROOT', os.path.abspath(os.path.join(_HERE, '..', 'inspection_robot')))
if _IR not in sys.path:
    sys.path.insert(0, _IR)

# CUDA 上 torchvision 缺 nms, 必须在 import ultralytics 前打补丁
from fix_nms import patch_torchvision_nms  # noqa: E402
patch_torchvision_nms()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fusion.config import YOLO_SIZE  # noqa: E402
from fusion.evaluate_pr import (  # noqa: E402
    ap_from_curve, curve_for_class, load_model, per_class_pr, train_val_split,
)
from fusion.pointcloud_assist_detector import (  # noqa: E402
    collate_batch, letterbox_boxes, load_sample,
)
from ultralytics.utils.nms import non_max_suppression  # noqa: E402

DATA = os.path.join(_IR, 'data_gen', 'output_crack_big')       # 融合侧数据集(单类裂缝, 624 帧)
CLS_DATA = os.path.join(_IR, 'data_gen', 'crack_1cls_big')     # 单支侧数据集(530 train / 94 val)
BEST = os.path.join(_HERE, 'runs', 'yolo11l_crack_1cls', 'weights', 'best.pt')
FUSION_CKPT = os.path.join(_IR, 'runs/assist_crack/last.pt')
NOPOINT_CKPT = os.path.join(_IR, 'runs/assist_crack_nopoint/last.pt')

# NMS IoU 必须与 ultralytics val 默认一致(0.7)。evaluate_pr.py 默认 0.45 更激进,
# 会把相邻裂缝框合并 -> 多裂缝帧漏检 -> recall 假性偏低(实测单支 R 上限 0.877 vs 0.951)。
CONF_MIN, NMS_IOU, MAX_DET, IOU_THRES = 0.001, 0.7, 300, 0.5
VAL_RATIO, SEED = 0.2, 42


# ================================================================
# 推理 (统一输入): 三类模型都吃 collate_batch 产出的同一批 (imgs, attrmaps, poses)
# ================================================================
def infer_yolo(model, frames, device):
    """原生 ultralytics DetectionModel: (B,3,640,640) -> 框 (M,6)"""
    preds = []
    for bi in range(0, len(frames), 8):
        imgs, _, _, _ = collate_batch([load_sample(i, root=DATA) for i in frames[bi:bi + 8]])
        imgs = imgs.to(device)
        with torch.no_grad():
            out = model(imgs)
        grid = out[0] if isinstance(out, (tuple, list)) else out
        nms = non_max_suppression(grid.detach().cpu(), conf_thres=CONF_MIN,
                                  iou_thres=NMS_IOU, max_det=MAX_DET)
        preds.extend([d if len(d) else torch.zeros((0, 6)) for d in nms])
    return preds


def infer_assist(model, frames, device):
    """PointCloudAssistedYOLO: (imgs, attrmaps, poses) -> 框 (M,6)"""
    preds = []
    for bi in range(0, len(frames), 8):
        imgs, attrmaps, poses, _ = collate_batch([load_sample(i, root=DATA) for i in frames[bi:bi + 8]])
        imgs, attrmaps, poses = imgs.to(device), attrmaps.to(device), poses.to(device)
        with torch.no_grad():
            out = model(imgs, attrmaps, poses)
        grid = out[0] if isinstance(out, tuple) else out
        nms = non_max_suppression(grid.detach().cpu(), conf_thres=CONF_MIN,
                                  iou_thres=NMS_IOU, max_det=MAX_DET)
        preds.extend([d if len(d) else torch.zeros((0, 6)) for d in nms])
    return preds


# ================================================================
# 图像单支自己的 val 集 (crack_1cls_big/images/val, 顺序划分)
# ================================================================
def infer_yolo_cls_val(model, device):
    """读 crack_1cls_big/images/val/*.jpg + labels/val/*.txt, 返回 (preds, gt)。"""
    import cv2
    from ultralytics.data.augment import LetterBox
    img_dir = os.path.join(CLS_DATA, 'images', 'val')
    lbl_dir = os.path.join(CLS_DATA, 'labels', 'val')
    files = sorted(f for f in os.listdir(img_dir) if f.endswith('.jpg'))
    lb = LetterBox(new_shape=(YOLO_SIZE, YOLO_SIZE), auto=False, stride=32)
    preds, gt = [], []
    B = 8
    for bi in range(0, len(files), B):
        tensors, gts = [], []
        for fn in files[bi:bi + B]:
            bgr = cv2.imread(os.path.join(img_dir, fn))
            x = lb(image=bgr).astype(np.float32) / 255.0
            tensors.append(np.transpose(x, (2, 0, 1)))
            stem = os.path.splitext(fn)[0]
            lp = os.path.join(lbl_dir, stem + '.txt')
            labels = (np.loadtxt(lp, dtype=np.float32).reshape(-1, 5)
                      if os.path.exists(lp) and os.path.getsize(lp) > 0 else np.empty((0, 5)))
            cls, xyxy = letterbox_boxes(labels, bgr.shape[:2])
            gts.append((xyxy.astype(np.float32), cls.astype(np.int64)))
        imgs = torch.from_numpy(np.stack(tensors)).to(device)
        with torch.no_grad():
            out = model(imgs)
        grid = out[0] if isinstance(out, (tuple, list)) else out
        nms = non_max_suppression(grid.detach().cpu(), conf_thres=CONF_MIN,
                                  iou_thres=NMS_IOU, max_det=MAX_DET)
        preds.extend([d if len(d) else torch.zeros((0, 6)) for d in nms])
        gt.extend(gts)
    return preds, gt


def gt_from_fusion_val(val_idx):
    gt = []
    for i in val_idx:
        _, _, _, cls, xyxy = load_sample(i, root=DATA)
        gt.append((xyxy.astype(np.float32), cls.astype(np.int64)))
    return gt


# ================================================================
# AP: 同时给 VOC 11 点插值(融合侧尺子) 与连续插值(ultralytics 侧尺子)
# ================================================================
def continuous_ap(recall, precision):
    """ultralytics 风格: 单调化 precision 后对 recall 做 101 点连续插值积分。"""
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def score(name, preds, gts, nc=1):
    det = per_class_pr(preds, gts, nc, IOU_THRES)
    out = {'name': name, 'n_frames': len(preds), 'classes': {}}
    aps_voc, aps_cont = [], []
    for c in range(nc):
        cur = curve_for_class(det[c])
        if cur is None:
            out['classes'][c] = {'n_gt': det[c]['n_gt'], 'n_det': len(det[c]['s']),
                                 'ap_voc11': None, 'ap_continuous': None}
            continue
        th, prec, rec = cur
        ap_v = ap_from_curve(rec, prec)
        ap_c = continuous_ap(rec, prec)
        aps_voc.append(ap_v); aps_cont.append(ap_c)
        n_tp = int(np.sum(det[c]['tp']))
        out['classes'][c] = {
            'n_gt': det[c]['n_gt'], 'n_det': len(det[c]['s']), 'n_tp': n_tp,
            'ap_voc11': round(float(ap_v), 4), 'ap_continuous': round(float(ap_c), 4),
        }
    out['mAP_voc11'] = round(float(np.mean(aps_voc)), 4) if aps_voc else None
    out['mAP_continuous'] = round(float(np.mean(aps_cont)), 4) if aps_cont else None
    return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_img = len([f for f in os.listdir(os.path.join(DATA, 'images')) if f.endswith('.jpg')])
    val_idx, train_idx = train_val_split(n_img, VAL_RATIO, SEED)
    print(f"[集] output_crack_big {n_img} 帧 -> 融合 val {len(val_idx)} 帧 (seed={SEED})")
    print(f"[设备] {device}\n")
    gt_fusion = gt_from_fusion_val(val_idx)

    results = []

    # ---- 组合 1/4: 图像单支 ----
    from ultralytics import YOLO
    ymodel = YOLO(BEST).model.float().eval().to(device)
    print(f"[1] 图像单支 @ 融合 val ({len(val_idx)} 帧) 推理中 ...")
    p1 = infer_yolo(ymodel, val_idx, device)
    results.append(score('图像单支 @ 融合val集', p1, gt_fusion))

    print("[4] 图像单支 @ 单支自己的 val 集 (94 帧) 推理中 ...")
    p4, gt4 = infer_yolo_cls_val(ymodel, device)
    results.append(score('图像单支 @ 单支val集', p4, gt4))
    del ymodel
    torch.cuda.empty_cache()

    # ---- 组合 2/3: 融合 / 无点云 ----
    for tag, ck, cname in [('融合', FUSION_CKPT, '融合 @ 融合val集'),
                           ('无点云', NOPOINT_CKPT, '无点云 @ 融合val集')]:
        model, _ = load_model(ck, device)
        print(f"[{tag}] {os.path.basename(os.path.dirname(ck))} @ 融合 val 推理中 ...")
        p = infer_assist(model, val_idx, device)
        results.append(score(cname, p, gt_fusion))
        del model
        torch.cuda.empty_cache()

    # ---- 汇总 ----
    print("\n" + "=" * 78)
    print(f"{'组合':<26} {'帧数':>5} {'mAP(VOC11点)':>14} {'mAP(连续)':>12} {'n_gt':>6} {'n_det':>6} {'TP':>5}")
    print("-" * 78)
    for r in results:
        c0 = r['classes'][0]
        print(f"{r['name']:<26} {r['n_frames']:>5} {str(r['mAP_voc11']):>14} "
              f"{str(r['mAP_continuous']):>12} {c0['n_gt']:>6} {c0['n_det']:>6} "
              f"{str(c0.get('n_tp','-')):>5}")
    print("=" * 78)

    outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'compare_branches.json')
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[save] {outp}")


if __name__ == '__main__':
    main()
