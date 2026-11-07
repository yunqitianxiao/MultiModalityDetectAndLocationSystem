#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对指定 best.pt, 搜一个"检出率尽量高、又兼顾精度"的置信度阈值 conf。
================================================================================
思路: 用 model.predict(conf=0.001) 一次性对 test 出所有低阈值框(只推理一次),
      再按 conf 阈值从低到高扫, IoU>=iou 贪心 1:1 匹配 GT, 得到每条 conf 的
      P / R / F1, 输出表格, 并给出:
        - F1 最高的 conf (均衡点)
        - 在 R >= min_r 前提下, P 最大的 conf (用户要的"r最好还能兼顾p")
      本机是 CPU 版 torch -> 只做推理/评估, 数据量小够快。

用法:
  python3 search_conf.py                                # 默认 best.pt(下载) + 轨道 test
  python3 search_conf.py --weights ../weights/best.pt --data 轨道_3cls_sim
  python3 search_conf.py --data 土建_3cls_sim --min-r 0.7 --imgsz 1280
"""
import os, sys, argparse, glob
import numpy as np

# 名称 -> 类别数 -> 类名 (顺序即打印顺序)
DEFAULTS = [
    ("土建_3cls_sim", 3, ["裂缝", "渗漏水", "管片掉块"]),
    ("轨道_3cls_sim", 3, ["断裂", "丢失", "松动歪斜"]),
    ("设备_2cls_sim", 2, ["异物入侵", "管线支架松脱"]),
]


def ds_dir(name):
    """数据集目录 = 本脚本同目录/<name> (与 train/eval 脚本同构, 免硬编码路径)。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def img_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (W, H)


def load_test_gt(name, n_classes):
    """返回 {image_path: [(cls, x0,y0,x1,y1), ...]}, 坐标为绝对像素。"""
    ti = os.path.join(ds_dir(name), "images", "test")
    li = os.path.join(ds_dir(name), "labels", "test")
    gt = {}
    for txt in sorted(glob.glob(os.path.join(li, "*.txt"))):
        stem = os.path.splitext(os.path.basename(txt))[0]
        img = next((os.path.join(ti, stem + e) for e in (".jpg", ".jpeg", ".png")
                    if os.path.exists(os.path.join(ti, stem + e))), None)
        if img is None:
            continue
        W, H = img_size(img)
        boxes = []
        for ln in open(txt, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            c, cx, cy, w, h = [float(x) for x in ln.split()]
            c = int(c)
            if c >= n_classes:
                continue
            x0 = (cx - w / 2) * W
            y0 = (cy - h / 2) * H
            x1 = (cx + w / 2) * W
            y1 = (cy + h / 2) * H
            boxes.append((c, x0, y0, x1, y1))
        gt[img] = boxes
    return gt


def iou(a, b):
    """a=(x0,y0,x1,y1) b=(x0,y0,x1,y1) -> IoU"""
    xi = max(a[0], b[0]); yi = max(a[1], b[1])
    xa = min(a[2], b[2]); ya = min(a[3], b[3])
    if xa <= xi or ya <= yi:
        return 0.0
    inter = (xa - xi) * (ya - yi)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    if aa <= 0 or ab <= 0:
        return 0.0
    return inter / (aa + ab - inter)


def match(boxes_pred, boxes_gt, iou_thr):
    """单图单类的贪心 1:1 匹配。boxes_pred=[[x0,y0,x1,y1]...] boxes_gt=[[x0,y0,x1,y1]...]
    返回 (tp, fp, fn)。"""
    tp = fp = fn = 0
    gt_used = [False] * len(boxes_gt)
    for p in boxes_pred:
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(boxes_gt):
            if gt_used[j]:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            gt_used[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = sum(1 for u in gt_used if not u)
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser(description="搜索最佳置信度阈值(P/R/F1)")
    ap.add_argument("--weights", default="../weights/best.pt", help="best.pt 路径")
    ap.add_argument("--data", default="轨道_3cls_sim", help="数据集名(见 DEFAULTS)")
    ap.add_argument("--imgsz", type=int, default=1280, help="推理尺寸(百万像素病集)")
    ap.add_argument("--iou", type=float, default=0.5, help="匹配 IoU 阈值(默认0.5)")
    ap.add_argument("--min-r", type=float, default=0.5, help="找 P 最大时要求的 R 下限")
    ap.add_argument("--conf-max", type=float, default=0.95)
    ap.add_argument("--conf-min", type=float, default=0.01)
    ap.add_argument("--conf-n", type=int, default=40, help="扫描档数")
    a = ap.parse_args()

    sel = [x for x in DEFAULTS if x[0] == a.data]
    if not sel:
        sys.exit(f"未知数据集: {a.data}\n可选: {[x[0] for x in DEFAULTS]}")
    name, n_classes, cn = sel[0]

    if not os.path.isfile(a.weights):
        sys.exit(f"找不到权重: {a.weights}")
    from ultralytics import YOLO

    gt = load_test_gt(name, n_classes)
    n_img = len(gt)
    if n_img == 0:
        sys.exit(f"{name} 的 test 为空(无标签或图片), 无法评估。")
    print(f"权重: {a.weights}")
    print(f"数据集: {name}  test 真图 {n_img} 张, 类: {cn}")
    print(f"推理 imgsz={a.imgsz}  IoU={a.iou}  匹配阈值下限 R>={a.min_r}")

    model = YOLO(a.weights)
    # 一次性低阈值推理, 收集每图每类的预测框(带 conf)
    # pred_img[img] = {cls: [ [x0,y0,x1,y1,conf], ... ]}
    pred_img = {}
    for img in list(gt):
        res = model.predict(img, imgsz=a.imgsz, conf=0.001, iou=a.iou, verbose=False)[0]
        pred_img[img] = {}
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for k in range(len(xyxy)):
                c = cls[k]
                pred_img[img].setdefault(c, []).append([*xyxy[k].tolist(), float(conf[k])])

    # 统计低阈值检测总量(判断是否对 test 有响应)
    n_det_low = sum(len(v) for img in pred_img for v in pred_img[img].values())
    print(f"conf>=0.001 时共检测到 {n_det_low} 个框 -> "
          f"{'✓ 模型对 test 有响应, 可扫阈值' if n_det_low else '✗ 0 个, 任何 conf 都 R=0, 说明跨域未检出'}")

    # 逐类 GT 框数
    gc = {k: 0 for k in range(n_classes)}
    for img in gt:
        for c, *_ in gt[img]:
            gc[c] = gc.get(c, 0) + 1
    print("GT 各类框数: " + "  ".join(f"{cn[k]}={gc.get(k,0)}" for k in range(n_classes)))

    # 阈值扫描
    thrs = np.linspace(a.conf_min, a.conf_max, a.conf_n)
    rows = []
    for thr in thrs:
        t = {"conf": float(thr)}
        for mode in ("all", "class"):
            pass
        # 汇总 TP/FP/FN
        clT = {k: [0, 0, 0] for k in range(n_classes)}  # [tp,fp,fn]
        for img, gts in gt.items():
            pp = pred_img.get(img, {})
            for c in range(n_classes):
                preds = [b[:4] for b in pp.get(c, []) if b[4] >= thr]
                gboxes = [g[1:] for g in gts if g[0] == c]
                tp, fp, fn = match(preds, gboxes, a.iou)
                clT[c][0] += tp; clT[c][1] += fp; clT[c][2] += fn
        # 整体
        TP = sum(v[0] for v in clT.values())
        FP = sum(v[1] for v in clT.values())
        FN = sum(v[2] for v in clT.values())
        P = TP / (TP + FP) if (TP + FP) else float("nan")
        R = TP / (TP + FN) if (TP + FN) else float("nan")
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        rows.append((t["conf"], P, R, F1, dict(clT)))

    # 打印表格
    print(f"\n{'conf':>6} {'P':>7} {'R':>6} {'F1':>6}")
    for conf, P, R, F1, clT in rows:
        print(f"{conf:>6.3f} {P:>7.3f} {R:>6.3f} {F1:>6.3f}")

    # 最优点
    def best(score):
        return max(rows, key=lambda x: score(x))

    f1b = best(lambda x: x[3])
    r_p = best(lambda x: (x[1] >= a.min_r) * x[2])  # 只在 R>=min_r 里挑 P 最大
    print(f"\n--- 结论 ---")
    print(f"F1 最高:  conf={f1b[0]:.3f}  P={f1b[1]:.3f}  R={f1b[2]:.3f}  F1={f1b[3]:.3f}")
    if r_p[1] >= a.min_r:
        print(f"R>={a.min_r} 下 P 最大:  conf={r_p[0]:.3f}  P={r_p[1]:.3f}  R={r_p[2]:.3f}  F1={r_p[3]:.3f}")
    else:
        print(f"⚠ 全区间 R 都 < {a.min_r} (最高 R={max(x[2] for x in rows):.3f}), "
              f"无法在满足 R>={a.min_r} 下选 P —— 印证模型对 test 跨域未检出, 需改训练数据.")

    # 逐类在最优 F1 conf 下的表现
    conf_f1 = f1b[0]
    ctr = next((clT for c, P, R, F1, clT in rows if abs(c - conf_f1) < 1e-9), None)
    if ctr is not None:
        print(f"\n--- 在 conf={conf_f1:.3f} 下的逐类 ---")
        for k in range(n_classes):
            tp, fp, fn = ctr[k]
            p = tp / (tp + fp) if (tp + fp) else float("nan")
            r = tp / (tp + fn) if (tp + fn) else float("nan")
            print(f"  {cn[k]:<8} TP={tp:>2} FP={fp:>2} FN={fn:>2}  P={p:>6.3f} R={r:>6.3f}")


if __name__ == "__main__":
    main()
