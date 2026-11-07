#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨道扣件3类 训练集合并: RFDD(轨道_3cls) + 仿真(轨道_3cls_sim) -> 干净混训集
================================================================
RFDD 是独立公开数据集(2048x2021, 域匹配), 仿真用赛方纹理(域内), 两者零重叠、零泄漏。
合并为单一 data.yaml 供 ultralytics train (类序一致 0断裂/1丢失/2松动歪斜)。

用法:
  python3 data_gen/build_track_train_merge.py \
      --rfd data1/轨道_3cls --sim data1/轨道_3cls_sim \
      --out data1/轨道_3cls_train_mix
"""
import argparse
import glob
import os
import shutil

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_exclude(path):
    """读留测 stem 列表(每行一个) -> set。源图名可能带 bing_ 前缀, 匹配时统一去掉。"""
    if not path or not os.path.isfile(path):
        return set()
    return {ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()}


def find_img(d, split, stem):
    for e in EXTS:
        p = os.path.join(d, "images", split, stem + e)
        if os.path.isfile(p):
            return p
    return None


def find_lbl(d, split, stem):
    p = os.path.join(d, "labels", split, stem + ".txt")
    return p if os.path.isfile(p) else None


def copy_split(src_root, split, prefix, seen, exclude):
    """把 src_root 的 images/<split> + labels/<split> 合成到 out 的同 split。
    源 stem 去掉 bing_ 前缀后若在 exclude(留测图) 里 -> 跳过, 杜绝泄漏。"""
    o_img = os.path.join(args.out, "images", split)
    o_lbl = os.path.join(args.out, "labels", split)
    os.makedirs(o_img, exist_ok=True)
    os.makedirs(o_lbl, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(src_root, "images", split, "*"))):
        stem0 = os.path.splitext(os.path.basename(p))[0]
        base = stem0[5:] if stem0.startswith("bing_") else stem0   # 去 bing_ 前缀
        ext = os.path.splitext(p)[1].lower()
        if base in exclude:
            continue
        lbl_p = find_lbl(src_root, split, stem0)
        if lbl_p is None:
            continue
        lbl = open(lbl_p).read().strip()
        stem = f"{prefix}_{stem0}"
        if stem in seen:
            continue
        seen.add(stem)
        shutil.copy2(p, os.path.join(o_img, stem + ext))
        with open(os.path.join(o_lbl, stem + ".txt"), "w") as f:
            f.write(lbl + ("\n" if lbl else ""))
        n += 1
    return n


def main():
    global args
    ap = argparse.ArgumentParser(description="轨道扣件3类 混训集(RFDD+仿真)")
    ap.add_argument("--rfd", required=True, help="RFDD 轨道_3cls 目录")
    ap.add_argument("--sim", required=True, help="仿真 轨道_3cls_sim 目录")
    ap.add_argument("--out", required=True, help="输出合并训练集")
    ap.add_argument("--exclude-stems", default=None, help="留测 stem 列表文件(每行一个), 源图带bing_也剔除")
    args = ap.parse_args()

    exclude = load_exclude(args.exclude_stems)
    seen = set()
    n_tr = copy_split(args.rfd, "train", "rfd", seen, exclude) + copy_split(args.sim, "train", "sim", seen, exclude)
    n_va = copy_split(args.rfd, "val", "rfd", seen, exclude) + copy_split(args.sim, "val", "sim", seen, exclude)
    print(f"   (剔除留测图 {len(exclude)} 张 if 存在于源)" if exclude else "")

    yaml = ("# 轨道扣件3类 混训集(RFDD+仿真)  build_track_train_merge.py\n"
            "path: " + os.path.abspath(args.out) + "\n"
            "train: images/train\nval: images/val\n"
            "names:\n  0: 断裂\n  1: 丢失\n  2: 松动歪斜\n")
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)
    print(f"✅ 轨道混训集 -> {args.out}  (train {n_tr} / val {n_va})")


if __name__ == "__main__":
    main()
