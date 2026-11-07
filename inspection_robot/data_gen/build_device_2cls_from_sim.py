#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真配对数据 -> 设备 2 类检测数据集 (异物入侵 / 管线支架松脱)
================================================================
data_gen 管线输出 8 类 (0裂缝 1渗漏 2/3/4扣件 5异物 6掉块 7支架)。
本脚本抽成设备 2 类:
  - 只保留 异物(5)/支架(7), 重映射 5->0 7->1
  - 无任何设备框的帧默认丢弃(缺省), 或 --keep-neg 保留作负样本
  - train/val 按 --val-frac 随机切分(seed 固定); 帧重编号 sim_000000 避免与真图 stem 冲突
  - test 目录由 split_bing_device.py 已生成(病集留测真图), 本脚本只填 train/val 并统一写 data.yaml

用法:
  python3 data_gen/build_device_2cls_from_sim.py \
      data_gen/output_big_s0 data_gen/output_big_s1 data_gen/output_big_s2 data_gen/output_big_s3 \
      --out /home/ylf/workspace/pro/3/设备_2cls_sim --val-frac 0.15
"""
import argparse
import glob
import os
import random
import shutil

SIM_DEVICE = {5: 0, 7: 1}           # 异物(5)->0, 支架(7)->1
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
CLASS_NAME = {0: "异物入侵", 1: "管线支架松脱"}


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
            if cls in SIM_DEVICE:
                rb.append((SIM_DEVICE[cls], cx, cy, w, h))
        if rb:
            items.append((img, rb))
        elif keep_neg and read_boxes(txt):
            items.append((img, []))
    return items


def main():
    ap = argparse.ArgumentParser(description="仿真 8 类 -> 设备 2 类检测集")
    ap.add_argument("sources", nargs="+", help="仿真输出目录(含 images/labels)")
    ap.add_argument("--out", required=True, help="输出 设备2类 数据集目录(含已生成 test)")
    ap.add_argument("--val-frac", type=float, default=0.15, help="验证集比例(默认 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-neg", action="store_true",
                    help="保留含非设备框(无设备框)的帧作背景负样本(默认丢弃)")
    args = ap.parse_args()

    items, per_src = [], {}
    for src in args.sources:
        it = collect_src(src, args.keep_neg)
        per_src[os.path.basename(src.rstrip("/"))] = len(it)
        items += it
        print(f"  {os.path.basename(src.rstrip('/')):28s} {len(it):4d} 帧(含设备框) -> "
              f"[{'负' if args.keep_neg else '忽略无框'}]")
    if not items:
        print("❌ 无有效样本(检查仿真输出目录是否含 images/labels)")
        return

    seen, uniq = set(), []
    for img, boxes in items:
        if img not in seen:
            seen.add(img)
            uniq.append((img, boxes))
    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    n_val = max(1, round(len(uniq) * args.val_frac))
    val, train = uniq[:n_val], uniq[n_val:]

    out = args.out
    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, sub, sp), exist_ok=True)
    gidx, n_neg, boxcnt = 0, 0, {0: 0, 1: 0}
    for sp, lst in (("train", train), ("val", val)):
        for img, boxes in lst:
            stem = f"sim_{gidx:06d}"
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

    t_img = os.path.join(out, "images", "test")
    t_lab = os.path.join(out, "labels", "test")
    n_test = len(glob.glob(os.path.join(t_img, "*"))) if os.path.isdir(t_img) else 0
    t_box = sum(len(read_boxes(t)) for t in glob.glob(os.path.join(t_lab, "*.txt")))

    yaml = ("# 设备2类仿真数据集 (merge sim 多seed + 病集留测真图test)\n"
            "path: " + os.path.abspath(out) + "\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: 异物入侵\n  1: 管线支架松脱\n")
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)

    print("✅ 设备2类仿真集已生成 ->", out)
    print(f"   train {len(train)} / val {len(val)} / test {n_test}  (共 {len(uniq)+n_test} 图; 仿真 {len(uniq)} 帧)")
    print(f"   train+val 框分布: 异物入侵 {boxcnt[0]} / 管线支架松脱 {boxcnt[1]}")
    print(f"   test 框分布: {t_box} 框")
    if args.keep_neg:
        print(f"   负样本(无设备框) {n_neg} 帧")


if __name__ == "__main__":
    main()
