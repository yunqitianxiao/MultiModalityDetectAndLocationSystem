#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真配对数据 -> 土建 3 类检测数据集 (裂缝/渗漏水/管片掉块)
================================================================
data_gen 管线输出的是 8 类 (0裂缝 1渗漏 2/3/4扣件 5异物 6掉块 7支架)。
本脚本把一份或多份仿真输出抽成土建 3 类:
  - 只保留裂缝(0)/渗漏水(1)/掉块(6,len=6), 重映射 6->2 (对齐 土建_病害_yolo_3cls 类序)
  - 无任何土建框的帧默认丢弃(缺省), 或 --keep-neg 保留作负样本
  - train/val 按 --val-frac 随机切分(seed 固定, 复现)
  - 写 images/{train,val} + labels/{train,val} + data.yaml(名同 土建3类)

用途: 把 output_data1_bing*, output_big_s* 这些仿真批次, 合成一份域内土建3类训练集,
     补掉当前 土建_病害_yolo_3cls 里 渗漏(858)/掉块(249) 严重不足的两类。

用法:
  python3 data_gen/build_tujian_3cls_from_sim.py \
      data_gen/output_data1_bing data_gen/output_data1_bing_s0 data_gen/output_big_s0 ... \
      --out data1/土建_3cls_sim --val-frac 0.15

单类裂缝 (仿真已是 nc=1, 类 0=裂缝):
  python3 data_gen/build_tujian_3cls_from_sim.py data_gen/output_crack_big \
      --out data_gen/crack_1cls_sim --remap 0:0 --names crack --val-frac 0.15
"""
import argparse
import glob
import os
import random
import shutil

# 仿真全局类号 -> 局部类号 (默认土建3类; 可用 --remap 覆盖, 如单类裂缝 "0:0")
SIM_LOCAL = {0: 0, 1: 1, 6: 2}        # 裂缝 0->0, 渗漏 1->1, 掉块 6->2
LOCAL_NAMES = ["裂缝", "渗漏水", "管片掉块"]
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
SUBDIRS = ("images", "labels")


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


def collect_src(src):
    """收集一份仿真源里所有"有图且含土建框(或 keep_neg)"的帧 -> list[(img, boxes)]。"""
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
        if rb:                              # 含土建框 -> 保留
            items.append((img, rb, stem))
        elif keep_empty or (keep_neg and read_boxes(txt)):
            # keep_empty: 保留"完全无框"的帧作背景负样本。单类(只 remap 一类)时
            # keep_neg 恒不生效 —— 它要求标签里含非目标类框, 而单类下不存在这种帧,
            # 于是训练/验证集 100% 是正样本, val 指标不惩罚假阳性。
            items.append((img, [], stem))
    return items


def main():
    global keep_neg, keep_empty
    ap = argparse.ArgumentParser(description="仿真 8 类 -> 土建 3 类检测集")
    ap.add_argument("sources", nargs="+", help="仿真输出目录(含 images/labels)")
    ap.add_argument("--out", required=True, help="输出 土建3类 数据集目录")
    ap.add_argument("--val-frac", type=float, default=0.15, help="验证集比例(默认 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-neg", action="store_true",
                    help="保留含非土建框(无土建框)的帧作背景负样本(默认丢弃)")
    ap.add_argument("--keep-empty", action="store_true",
                    help="保留完全无框的帧作背景负样本(单类场景必需; --keep-neg 在单类下失效)")
    ap.add_argument("--remap", default="", help="覆盖类映射, 如 '0:0'(单类裂缝); 默认 0:0,1:1,6:2")
    ap.add_argument("--names", default="", help="覆盖局部类名(逗号分隔), 如 'crack'; 默认裂缝,渗漏水,管片掉块")
    args = ap.parse_args()
    keep_neg = args.keep_neg
    keep_empty = args.keep_empty
    global SIM_LOCAL, LOCAL_NAMES
    if args.remap:
        SIM_LOCAL = {int(k): int(v) for k, v in (kv.split(":") for kv in args.remap.split(","))}
    if args.names:
        LOCAL_NAMES = [s.strip() for s in args.names.split(",")]

    items = []
    per_src = {}
    for src in args.sources:
        it = collect_src(src)
        per_src[os.path.basename(src.rstrip("/"))] = len(it)
        items += it
    if not items:
        print("❌ 无有效样本(检查仿真输出目录是否含 images/labels)")
        return

    # 固定随机划分 train/val, 去重(同一路径只留一次)
    seen = set()
    uniq = []
    for it in items:
        if it[0] not in seen:
            seen.add(it[0])
            uniq.append(it)
    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    n_val = max(1, round(len(uniq) * args.val_frac))
    val = uniq[:n_val]
    train = uniq[n_val:]

    # 写目录
    out = args.out
    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, sub, sp), exist_ok=True)
    # 全新名字: sim_000000.jpg 递增
    gidx = 0
    n_neg = 0
    boxcnt = {i: 0 for i in sorted(set(SIM_LOCAL.values()))}
    for sp, lst in (("train", train), ("val", val)):
        for img, boxes, _ in lst:
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

    # data.yaml
    yaml = ("# 由 仿真配对数据 -> 检测集 (build_tujian_3cls_from_sim.py)\n"
            "path: " + os.path.abspath(out) + "\n"
            "train: images/train\nval: images/val\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(LOCAL_NAMES)))
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)

    print("✅ 仿真检测集已生成 ->", out)
    print(f"   train {len(train)} / val {len(val)}  (共 {len(uniq)} 帧)")
    print("   框分布: " + " / ".join(f"{LOCAL_NAMES[i] if i < len(LOCAL_NAMES) else i} {c}"
                                     for i, c in boxcnt.items()))
    if keep_neg:
        print(f"   负样本(无土建框) {n_neg} 帧")
    else:
        print(f"   丢弃纯非土建帧: 每个 src 见下")
    for src, cnt in per_src.items():
        print(f"      {src:28s} {cnt:4d} 帧")


if __name__ == "__main__":
    main()
