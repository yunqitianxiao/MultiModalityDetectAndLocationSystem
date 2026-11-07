#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从分割掩膜导出 YOLO 检测框标签(labels/*.txt + data.yaml)

输入: 掩膜目录(*_mask.png, 白=目标) + 对应源图目录
输出: <输出目录>/images/*.jpg + labels/*.txt + data.yaml
     每个连通域转一个检测框(归一化 cx cy w h)。
     可把 设备_正常 作为负样本编入(空标签)。

用法:
  python3 掩膜转YOLO框.py \
      --mask-dir data/设备_异物入侵_人工掩膜 \
      --src-dir  data/设备_异物入侵 \
      --out      data/设备_异物入侵_yolo \
      --normal-dir data/设备_正常 \
      --class-name foreign_object
"""
import argparse
import os
import shutil
import cv2
import numpy as np

MIN_BOX_AREA = 20          # 连通域最小面积(px), 滤掉噪声
VAL_RATIO = 0.1            # 验证集比例


def mask_to_boxes(mask):
    """从二值掩膜提取连通域 bbox, 返回 [(x1,y1,x2,y2), ...]"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):                       # 跳过背景 0
        x, y, w, h, area = stats[i]
        if area < MIN_BOX_AREA:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def write_yolo_txt(path, boxes, img_w, img_h, cls_id=0):
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-dir", required=True, help="掩膜目录")
    ap.add_argument("--src-dir", required=True, help="源图目录")
    ap.add_argument("--out", required=True, help="输出数据集目录")
    ap.add_argument("--normal-dir", default="", help="设备_正常 负样本目录(可选)")
    ap.add_argument("--class-name", default="foreign_object", help="类别名")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    # 收集 (图片, 掩膜) 对
    items = []
    for fn in sorted(os.listdir(args.mask_dir)):
        if not fn.endswith("_mask.png"):
            continue
        name = fn[:-len("_mask.png")]
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG"):
            p = os.path.join(args.src_dir, name + ext)
            if os.path.exists(p):
                img = p
                break
        if img is None:
            print(f"  [跳过] 找不到源图: {name}")
            continue
        items.append((name, img, os.path.join(args.mask_dir, fn)))

    # 负样本(正常图)
    normal = []
    if args.normal_dir and os.path.isdir(args.normal_dir):
        for fn in sorted(os.listdir(args.normal_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                normal.append((os.path.splitext(fn)[0],
                               os.path.join(args.normal_dir, fn), None))

    total = len(items) + len(normal)
    print(f"正样本(有掩膜): {len(items)}  负样本(正常): {len(normal)}  共 {total}")
    if total == 0:
        print("没有可转换的图片。")
        return

    # 划分 train/val
    import random
    rng = random.Random(42)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * VAL_RATIO))
    val_names = {name for name, _, _ in items[:n_val]}
    # 负样本按相同比例进 val
    rng.shuffle(normal)
    n_val_n = max(1, int(len(normal) * VAL_RATIO)) if normal else 0
    val_names_n = {name for name, _, _ in normal[:n_val_n]}

    written = {"train": 0, "val": 0}
    for name, img_path, mask_path in items:
        split = "val" if name in val_names else "train"
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h)) if mask is not None else None
        boxes = mask_to_boxes(mask) if mask is not None else []
        write_yolo_txt(os.path.join(args.out, "labels", split, name + ".txt"),
                       boxes, w, h)
        shutil.copy(img_path, os.path.join(args.out, "images", split, name + ".jpg"))
        written[split] += 1
        if boxes:
            print(f"  [{split}] {name}: {len(boxes)} 框")

    for name, img_path, _ in normal:
        split = "val" if name in val_names_n else "train"
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        open(os.path.join(args.out, "labels", split, name + ".txt"), "w").close()  # 空标签
        shutil.copy(img_path, os.path.join(args.out, "images", split, name + ".jpg"))
        written[split] += 1

    # data.yaml
    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"# 从掩膜自动导出(掩膜转YOLO框.py)\n")
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write(f"train: images/train\nval: images/val\n\n")
        f.write(f"nc: 1\nnames: ['{args.class_name}']\n")

    print(f"\n完成! 输出: {args.out}")
    print(f"  train: {written['train']} 张 (labels 里含框的为正样本, 空为负样本)")
    print(f"  val:   {written['val']} 张")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
