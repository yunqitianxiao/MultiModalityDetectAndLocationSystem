#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真配对数据 -> 轨道(扣件)3类检测数据集 (断裂/丢失/松动歪斜)
================================================================
data_gen 管线输出的是 8 类 (0裂缝 1渗漏 2/3/4扣件 5异物 6掉块 7支架)。
本脚本把一份或多份仿真输出抽成轨道 3 类:
  - 只保留扣件(2/3/4), 重映射 {2:0(断裂), 3:1(丢失), 4:2(松动歪斜)} (对齐 轨道_3cls 类序)
  - 其余类框丢弃; 无任何扣件框的帧默认丢弃(--keep-neg 可保留作背景负样本)
  - test 目录由 split_bing_rail.py 已生成(病集留测真图), 本脚本只填 train/val 并统一写 data.yaml
  - train/val 按 --val-frac 随机切分(seed 固定, 复现); 帧重编号 sim_000000.jpg 避免与真图 stem 冲突

用法:
  python3 data_gen/build_rail_3cls_from_sim.py \
      data_gen/output_rail_s0 data_gen/output_rail_s1 data_gen/output_rail_s2 \
      --out /home/ylf/workspace/pro/2/data1/轨道_3cls_sim --val-frac 0.15
"""
import argparse
import glob
import os
import random
import shutil

# 仿真全局扣件类号 -> 轨道3类局号
SIM_LOCAL = {2: 0, 3: 1, 4: 2}        # broken->0, missing->1, loose->2
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
CLASS_NAME = {0: "断裂", 1: "丢失", 2: "松动歪斜"}


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
    """收集一份仿真源里所有"有图且含扣件框(或 keep_neg)"的帧 -> list[(img, remapped_boxes)]。"""
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
            if cls in SIM_LOCAL:
                rb.append((SIM_LOCAL[cls], cx, cy, w, h))
        if rb:
            items.append((img, rb))
        elif keep_neg and read_boxes(txt):   # 只含非扣件框的负样本(可选保留, 空标签)
            items.append((img, []))
    return items


def main():
    ap = argparse.ArgumentParser(description="仿真 8 类 -> 轨道 3 类检测集")
    ap.add_argument("sources", nargs="+", help="仿真输出目录(含 images/labels)")
    ap.add_argument("--out", required=True, help="输出 轨道3类 数据集目录(含已生成 test)")
    ap.add_argument("--val-frac", type=float, default=0.15, help="验证集比例(默认 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-neg", action="store_true",
                    help="保留含非扣件框(无扣件框)的帧作背景负样本(默认丢弃)")
    args = ap.parse_args()

    items = []
    per_src = {}
    for src in args.sources:
        it = collect_src(src, args.keep_neg)
        n = len(it)
        per_src[os.path.basename(src.rstrip("/"))] = n
        items += it
        print(f"  {os.path.basename(src.rstrip('/')):28s} {n:4d} 帧(含扣件框) -> [{'负' if args.keep_neg else '忽略无框'}]")
    if not items:
        print("❌ 无有效样本(检查仿真输出目录是否含 images/labels)")
        return

    # 去重(同一帧只留一次) + 固定随机划分 train/val
    seen = set()
    uniq = []
    for img, boxes in items:
        if img not in seen:
            seen.add(img)
            uniq.append((img, boxes))
    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    n_val = max(1, round(len(uniq) * args.val_frac))
    val = uniq[:n_val]
    train = uniq[n_val:]

    # 写 train/val (帧重编号 sim_000000)
    out = args.out
    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, sub, sp), exist_ok=True)
    gidx = 0
    boxcnt = {0: 0, 1: 0, 2: 0}
    n_neg = 0
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

    # 统计 test(已由 split_bing_rail.py 生成)
    t_img = os.path.join(out, "images", "test")
    t_lab = os.path.join(out, "labels", "test")
    n_test = len(glob.glob(os.path.join(t_img, "*"))) if os.path.isdir(t_img) else 0
    t_box = 0
    for txt in glob.glob(os.path.join(t_lab, "*.txt")):
        t_box += len(read_boxes(txt))

    # 最终 data.yaml
    yaml = ("# 轨道扣件3类仿真数据集 (merge sim 多seed + 病集留测真图test)\n"
            "path: " + os.path.abspath(out) + "\n"
            "train: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: 断裂\n  1: 丢失\n  2: 松动歪斜\n")
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)

    print("✅ 轨道3类仿真集已生成 ->", out)
    print(f"   train {len(train)} / val {len(val)} / test {n_test}  (共 {len(uniq)+n_test} 图; 仿真 {len(uniq)} 帧)")
    print(f"   train+val 框分布: 断裂 {boxcnt[0]} / 丢失 {boxcnt[1]} / 松动歪斜 {boxcnt[2]}")
    print(f"   test 框分布: {t_box} 框")
    if args.keep_neg:
        print(f"   负样本(无扣件框) {n_neg} 帧")


if __name__ == "__main__":
    main()
