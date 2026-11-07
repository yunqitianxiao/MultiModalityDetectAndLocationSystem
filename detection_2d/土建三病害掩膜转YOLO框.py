#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
土建三病害掩膜 → YOLO 检测框标签(三个独立单类数据集)
====================================================
输入(掩膜, 白=病害, 黑=背景):
  土建_裂缝_手动标注的掩膜     514 张  (公共 bitumen/concrete/masonry/runway 同名 + 隧道 mask_06xx)
  土建_管片掉块_人工掩膜        44 张  (<name>_mask.png)
  土建_渗漏水_手动标注的掩膜    549 张  (与源图同名, .rf.<hash>)

输出(每类一个数据集, images+labels 各含 train/val/test):
  data/土建_裂缝_yolo/        nc:1  names:['crack']
  data/土建_管片掉块_yolo/     nc:1  names:['spalling']
  data/土建_渗漏水_yolo/       nc:1  names:['leakage']

每个连通域 → 一个归一化检测框 (class cx cy w h)。
划分沿用裂缝习惯:
  裂缝   按既有域约定(隧道 5/1/6 分散 + 公共图 24 进测试, 其余 85/15) → 411/73/30
  掉块/渗漏水  随机: 约10% 进测试, 其余 85/15 分训练/验证 (random.Random(42) 可复现)
"""
import os
import re
import random
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

MIN_BOX_AREA = 20      # 连通域最小面积(px), 滤噪声
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

CATEGORIES = [
    {
        "key": "crack",
        "mask_dir": "土建_裂缝_手动标注的掩膜",
        "src_dir": "土建_裂缝",
        "out_dir": "土建_裂缝_yolo",
        "class_name": "crack",
        "mask_suffix": ".png",
        "naming": "crack",        # 同名 或 mask_前缀
    },
    {
        "key": "spalling",
        "mask_dir": "土建_管片掉块_人工掩膜",
        "src_dir": "土建_管片掉块",
        "out_dir": "土建_管片掉块_yolo",
        "class_name": "spalling",
        "mask_suffix": "_mask.png",
        "naming": "suffix",       # 去掉 _mask 后缀
    },
    {
        "key": "leakage",
        "mask_dir": "土建_渗漏水_手动标注的掩膜",
        "src_dir": "土建_渗漏水",
        "out_dir": "土建_渗漏水_yolo",
        "class_name": "leakage",
        "mask_suffix": ".png",
        "naming": "same",         # 与源图同名
    },
]


def resolve_image(cat, mask_fn, src_dir):
    """掩膜名 → 源图路径"""
    stem = mask_fn[: -len(cat["mask_suffix"])] if cat["mask_suffix"] else \
        os.path.splitext(mask_fn)[0]
    if cat["naming"] == "crack":
        candidates = [stem]
        if stem.startswith("mask_"):
            candidates.insert(0, stem[len("mask_"):])
        elif not stem.startswith(("bitumen_", "concrete_", "masonry_", "runway_")):
            candidates.insert(0, "mask_" + stem)
    elif cat["naming"] == "suffix":
        candidates = [stem]
    else:
        candidates = [stem]
    for c in candidates:
        for ext in IMG_EXTS:
            p = os.path.join(src_dir, c + ext)
            if os.path.exists(p):
                return p
    return None


def mask_to_boxes(mask_path, img_w, img_h):
    """二值掩膜 → 连通域 bbox 列表 [(x1,y1,x2,y2), ...]"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    if (mask.shape[1], mask.shape[0]) != (img_w, img_h):
        mask = cv2.resize(mask, (img_w, img_h))
    mask = (mask > 127).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_BOX_AREA:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def write_label(path, boxes, img_w, img_h):
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        bw, bh = min(max(bw, 0.0), 1.0), min(max(bh, 0.0), 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def make_split(cat_key, items):
    rng = random.Random(42)
    if cat_key == "crack":
        # 沿用裂缝既有域约定
        tunnel = [it for it in items if re.match(r"^06\d{2}", os.path.basename(it[1]))]
        public = [it for it in items if not re.match(r"^06\d{2}", os.path.basename(it[1]))]
        rng.shuffle(tunnel)
        rng.shuffle(public)
        tr = tunnel[:5]
        va = tunnel[5:6]
        te = tunnel[6:]
        pub_test = public[:24]
        pub_rest = public[24:]
        pub_train, pub_val = train_test_split(pub_rest, test_size=0.15, random_state=42)
        return tr + pub_train, va + pub_val, te + pub_test
    # 掉块/渗漏水: 随机 10% 测试 + 其余 85/15
    rng.shuffle(items)
    n_test = max(1, round(len(items) * 0.10))
    test = items[:n_test]
    rest = items[n_test:]
    n_val = max(1, round(len(rest) * 0.15))
    val = rest[:n_val]
    train = rest[n_val:]
    return train, val, test


def main():
    for cat in CATEGORIES:
        mask_dir = os.path.join(DATA_ROOT, cat["mask_dir"])
        src_dir = os.path.join(DATA_ROOT, cat["src_dir"])
        out_dir = os.path.join(DATA_ROOT, cat["out_dir"])

        if not os.path.isdir(mask_dir):
            print(f"[{cat['key']}] 掩膜目录不存在, 跳过: {mask_dir}")
            continue

        items = []
        for fn in sorted(os.listdir(mask_dir)):
            if not fn.endswith(cat["mask_suffix"]):
                continue
            img = resolve_image(cat, fn, src_dir)
            if img is None:
                print(f"[{cat['key']}] [跳过] 找不到源图: {fn}")
                continue
            items.append((os.path.splitext(fn)[0], img, os.path.join(mask_dir, fn)))
        items.sort(key=lambda it: os.path.basename(it[1]))

        if not items:
            print(f"[{cat['key']}] 无有效样本, 跳过")
            continue

        train, val, test = make_split(cat["key"], items)
        splits = {"train": train, "val": val, "test": test}

        # 清空重建输出目录
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        for split in splits:
            for sub in ("images", "labels"):
                os.makedirs(os.path.join(out_dir, split, sub), exist_ok=True)

        zero_box = []
        total_boxes = 0
        for split, lst in splits.items():
            for name, img_path, mask_path in lst:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                boxes = mask_to_boxes(mask_path, w, h)
                if not boxes:
                    zero_box.append(os.path.basename(img_path))
                write_label(os.path.join(out_dir, split, "labels", name + ".txt"), boxes, w, h)
                shutil.copy2(img_path, os.path.join(out_dir, split, "images", os.path.basename(img_path)))
                total_boxes += len(boxes)

        # data.yaml
        with open(os.path.join(out_dir, "data.yaml"), "w") as f:
            f.write("# 由 土建三病害掩膜转YOLO框.py 从掩膜自动导出(YOLO 检测框)\n")
            f.write(f"path: {os.path.abspath(out_dir)}\n")
            f.write("train: train/images\nval: val/images\ntest: test/images\n\n")
            f.write(f"nc: 1\nnames: ['{cat['class_name']}']\n")

        n = {s: len(v) for s, v in splits.items()}
        print(f"\n=== {cat['key']} ({cat['class_name']}) ===")
        print(f"  样本: {n['train']} train / {n['val']} val / {n['test']} test  (共 {len(items)})")
        print(f"  检测框总数: {total_boxes}")
        if zero_box:
            print(f"  [警告] 掩膜无有效连通域(面积<{MIN_BOX_AREA}px): {len(zero_box)} 张")
            print("         " + " ".join(zero_box[:10]))
        print(f"  输出: {out_dir}")


if __name__ == "__main__":
    main()
