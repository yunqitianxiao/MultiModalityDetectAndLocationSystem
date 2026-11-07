#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真配对数据 -> 轨道扣件 3 类检测数据集 (断裂/丢失/松动歪斜)
================================================================
data_gen 管线输出 8 类 (sim id: 0裂缝 1渗漏 2扣件断裂 3扣件缺失 4扣件松动
                9=0..7 对应 defect_placement.CLASSES 顺序)。
本脚本抽成轨道扣件 3 类:
  - 只保留 扣件断裂(2)/扣件缺失(3)/扣件松动(4), 重映射 2->0 3->1 4->2
  - 无任何扣件框的帧默认丢弃(缺省), 或 --keep-neg 保留作负样本
  - train/val 按 --val-frac 随机切分(seed 固定)
  - 写 images/{train,val} + labels/{train,val} + data.yaml(名同 轨道3类)

用途: 把 output_track_s* 这些仿真批次合成一份域内轨道扣件3类训练集,
     与 RFDD(轨道_3cls) 混合训练, 提高赛方扣件高清图上的检出率。

用法:
  python3 data_gen/build_track_3cls_from_sim.py \
      data_gen/output_track_s0 data_gen/output_track_s1 ... \
      --out data1/轨道_3cls_sim --val-frac 0.15
"""
import argparse
import glob
import os
import random
import shutil

# 仿真全局类号 -> 轨道扣件3类局号 (对齐 轨道_3cls 类序 0断裂/1丢失/2松动歪斜)
SIM_TRACK = {2: 0, 3: 1, 4: 2}        # 断裂2->0, 缺失3->1, 松动4->2
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SUB = ("images", "labels")


def read_boxes(txt):
    boxes = []
    if not os.path.isfile(txt):
        return boxes
    for line in open(txt, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        p = line.split()
        if len(p) != 5:
            continue
        boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return boxes


def find_img(img_dir, stem):
    for ext in EXTS:
        c = os.path.join(img_dir, stem + ext)
        if os.path.exists(c):
            return c
    return None


def collect_src(src, keep_neg):
    img_dir = os.path.join(src, "images")
    lbl_dir = os.path.join(src, "labels")
    if not os.path.isdir(lbl_dir):
        return []
    items = []
    for txt in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(txt))[0]
        img = find_img(img_dir, stem)
        if img is None:
            continue
        rb = []
        for cls, cx, cy, w, h in read_boxes(txt):
            if cls in SIM_TRACK:
                rb.append((SIM_TRACK[cls], cx, cy, w, h))
        if rb:
            items.append((img, rb, stem))
        elif keep_neg and read_boxes(txt):
            items.append((img, [], stem))
    return items


def main():
    ap = argparse.ArgumentParser(description="仿真 8 类 -> 轨道扣件 3 类检测集")
    ap.add_argument("sources", nargs="+", help="仿真输出目录(含 images/labels)")
    ap.add_argument("--out", required=True, help="输出 轨道3类 数据集目录")
    ap.add_argument("--val-frac", type=float, default=0.15, help="验证集比例(默认 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-neg", action="store_true",
                    help="保留含非扣件框(无扣件框)的帧作背景负样本(默认丢弃)")
    args = ap.parse_args()

    items = []
    per_src = {}
    for src in args.sources:
        it = collect_src(src, args.keep_neg)
        per_src[os.path.basename(src.rstrip("/"))] = len(it)
        items += it
    if not items:
        print("❌ 无有效样本(检查仿真输出目录是否含 images/labels)")
        return

    seen = set()
    uniq = []
    for it in items:
        if it[0] not in seen:
            seen.add(it[0])
            uniq.append(it)
    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    n_val = max(1, round(len(uniq) * args.val_frac))
    val, train = uniq[:n_val], uniq[n_val:]

    out = args.out
    for sp in ("train", "val"):
        for sub in SUB:
            os.makedirs(os.path.join(out, sub, sp), exist_ok=True)
    gidx = 0
    n_neg = 0
    boxcnt = {0: 0, 1: 0, 2: 0}
    for sp, lst in (("train", train), ("val", val)):
        for img, boxes, _ in lst:
            stem = f"trk_{gidx:06d}"
            ext = os.path.splitext(img)[1].lower()
            shutil.copy2(img, os.path.join(out, "images", sp, stem + ext))
            if boxes:
                with open(os.path.join(out, "labels", sp, stem + ".txt"), "w") as f:
                    for b in boxes:
                        f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                        boxcnt[b[0]] += 1
            else:
                n_neg += 1
                open(os.path.join(out, "labels", sp, stem + ".txt"), "w").close()
            gidx += 1

    yaml = ("# 由 仿真配对数据 -> 轨道扣件3类 (build_track_3cls_from_sim.py)\n"
            "path: " + os.path.abspath(out) + "\n"
            "train: images/train\nval: images/val\n"
            "names:\n  0: 断裂\n  1: 丢失\n  2: 松动歪斜\n")
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)

    print("✅ 轨道扣件3类仿真集已生成 ->", out)
    print(f"   train {len(train)} / val {len(val)}  (共 {len(uniq)} 帧)")
    print(f"   框分布: 断裂 {boxcnt[0]} / 丢失 {boxcnt[1]} / 松动歪斜 {boxcnt[2]}")
    for src, cnt in per_src.items():
        print(f"      {src:28s} {cnt:4d} 帧")


if __name__ == "__main__":
    main()
