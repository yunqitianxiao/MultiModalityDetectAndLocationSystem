#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v13l vs 11l A/B: 同方法、同测试集(增强版 53 图) 真 P/R 对比
================================================================
做法:
  - 用 conf=0.001 的低阈值 predict 出全部候选框 (iou=0.5 NMS)
  - 对每个 conf 阈值: 过滤候选 → 与 GT 做贪心 IoU 匹配 (IoU≥0.5 算 TP)
  - 输出 P / R / TP / FP / FN 对照表
用法 (服务器, 需 PYTHONPATH 含 yolov13 fork 以加载 v13l 权重):
  /root/autodl-tmp/pro2venv/bin/python ab_v13l_vs_11l.py
"""
import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ---------- 路径: 服务器固定布局 ----------
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
DATA = Path(os.environ.get("LEAKAGE_DATA", PROJECT_ROOT / "data" / "土建_渗漏水_yolo_enh"))
TEST_IMG = DATA / "test" / "images"
TEST_LBL = DATA / "test" / "labels"
RUNS = Path(os.environ.get("LEAKAGE_RUNS", PROJECT_ROOT / "runs_leakage"))

MODELS = [
    ("11l@512 基线", RUNS / "yolo11l_imgsz512_enh/weights/best.pt", 512),
    ("v13l@512 新",  RUNS / "yolov13l_imgsz512_enh/weights/best.pt", 512),
]

CONF_GRID = (0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def parse_gt(lbl_path, w, h):
    boxes = []
    if not os.path.isfile(lbl_path):
        return boxes
    for line in Path(lbl_path).read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            _, cx, cy, bw, bh = map(float, parts)
        except ValueError:
            continue
        boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def match(preds, gts, iou_t=0.5):
    """preds: [(x1,y1,x2,y2,conf)]; gts: [(x1,y1,x2,y2)]. 贪心按 conf 降序匹配."""
    used = [False] * len(gts)
    tp = 0
    for pb in sorted(preds, key=lambda b: -b[4]):
        best_iou, gi = iou_t, -1
        for i, gb in enumerate(gts):
            if used[i]:
                continue
            ov = iou_xyxy(pb[:4], gb)
            if ov > best_iou:
                best_iou, gi = ov, i
        if gi >= 0:
            used[gi] = True
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def eval_model(weights, imgsz):
    print(f"\n🔬 推理 {weights.name} (imgsz={imgsz}, conf=0.001)…")
    model = YOLO(str(weights))
    imgs = sorted(TEST_IMG.glob("*"))
    results = model.predict([str(p) for p in imgs], imgsz=imgsz, conf=0.001,
                            iou=0.5, device="0", batch=16, verbose=False)
    gt_by_img, img_paths = {}, []
    for p in imgs:
        with Image.open(p) as im:
            w, h = im.size
        img_paths.append(p)
        gt_by_img[p.name] = parse_gt(TEST_LBL / (p.stem + ".txt"), w, h)
    n_gt = sum(len(v) for v in gt_by_img.values())

    # 每个 conf 阈值
    stats = {}
    for conf in CONF_GRID:
        tp = fp = fn = 0
        for p, res in zip(img_paths, results):
            pb = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                c = res.boxes.conf.cpu().numpy()
                for bx, cc in zip(xyxy, c):
                    if cc >= conf:
                        pb.append((float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]), float(cc)))
            t, f_p, f_n = match(pb, gt_by_img[p.name])
            tp += t; fp += f_p; fn += f_n
        p_val = tp / (tp + fp) if (tp + fp) else 0.0
        r_val = tp / (tp + fn) if (tp + fn) else 0.0
        stats[conf] = (tp, fp, fn, p_val, r_val)
    return n_gt, stats


def main():
    all_stats = {}
    for name, weights, imgsz in MODELS:
        n_gt, stats = eval_model(weights, imgsz)
        all_stats[name] = (n_gt, stats)

    print("\n" + "=" * 78)
    print(f"v13l vs 11l A/B  (IoU≥0.5 算 TP, 增强测试集 {TEST_IMG} 53 图)")
    print("=" * 78)
    for name, (n_gt, _) in all_stats.items():
        print(f"  模型: {name:<14} GT 框 = {n_gt}")
    print("  " + "-" * 76)
    print(f"  {'conf':>6} | " + " | ".join(f"{name:<14}" for name, _ in all_stats.items()))
    for conf in CONF_GRID:
        cells = []
        for name, (n_gt, stats) in all_stats.items():
            tp, fp, fn, p_val, r_val = stats[conf]
            cells.append(f"P={p_val*100:5.1f}% R={r_val*100:5.1f}%")
        print(f"  {conf:>6.3f} | " + " | ".join(f"{c:<18}" for c in cells))
    print("  " + "-" * 76)

    # 关键点: R@0.25 (部署 conf) 与 R@0.1
    print("\n📌 部署参考 (conf=0.25 与 conf=0.1):")
    for name, (n_gt, stats) in all_stats.items():
        for conf in (0.25, 0.1):
            tp, fp, fn, p_val, r_val = stats[conf]
            print(f"  {name:<14} conf={conf:<5} P={p_val*100:5.1f}%  R={r_val*100:5.1f}%  "
                  f"TP={tp:>3} FP={fp:>3} FN={fn:>3}")

    # 最低 conf 下漏检
    print(f"\n📋 最低 conf=0.001 下仍漏检的图片:")
    for name, weights, imgsz in MODELS:
        model = YOLO(str(weights))
        imgs = sorted(TEST_IMG.glob("*"))
        results = model.predict([str(p) for p in imgs], imgsz=imgsz, conf=0.001,
                                iou=0.5, device="0", batch=16, verbose=False)
        missed = []
        for p, res in zip(imgs, results):
            with Image.open(p) as im:
                w, h = im.size
            gts = parse_gt(TEST_LBL / (p.stem + ".txt"), w, h)
            pb = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                c = res.boxes.conf.cpu().numpy()
                for bx, cc in zip(xyxy, c):
                    pb.append((float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]), float(cc)))
            _, _, fn_n = match(pb, gts)
            if fn_n > 0:
                missed.append((p.name, fn_n))
        print(f"  {name:<14} 漏检图 {len(missed)} 张 / {len(imgs)}: ", end="")
        print(" ".join(f"{n}({k})" for n, k in sorted(missed, key=lambda x: -x[1])[:8]))


if __name__ == "__main__":
    main()
