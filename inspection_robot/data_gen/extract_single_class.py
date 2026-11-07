#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从多类配对数据集里抽出一个类的子集, 重编号为 0, 生成单类数据集。

多类源(output_big / output_rail_s*)里某个类的标签就是现成的单类受控实验数据,
不必重跑仿真录制。其他类的病害仍留在图像里, 成为未标注的干扰物
(对 A/B 两档同等影响, 不破坏对比, 但会让绝对 AP 低于纯净单类数据集)。

用法:
    python3 data_gen/extract_single_class.py --class-id 2 --out data_gen/output_rail_broken_a \\
        data_gen/output_rail_s0
"""
import argparse
import glob
import os
import shutil

SUBDIRS = ["images", "labels", "clouds", "attrmaps", "poses"]
EXT = {"images": ".jpg", "labels": ".txt", "clouds": ".npy",
       "attrmaps": ".npz", "poses": ".json"}


def valid_frames(root):
    sets = []
    for sub in SUBDIRS:
        d = os.path.join(root, sub)
        s = set()
        if os.path.isdir(d):
            ext = EXT[sub]
            for f in os.listdir(d):
                stem, e = os.path.splitext(f)
                if e == ext:
                    s.add(stem)
        sets.append(s)
    return sorted(set.intersection(*sets)) if sets else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-id", type=int, required=True, help="源数据里该类的 id")
    ap.add_argument("--class-name", default="defect")
    ap.add_argument("--out", required=True)
    ap.add_argument("sources", nargs="+")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for sub in SUBDIRS:
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    g = 0
    n_box = 0
    empty = 0
    for src in args.sources:
        frames = valid_frames(src)
        print(f"  {os.path.basename(src)}: {len(frames)} 帧五路齐全")
        for stem in frames:
            for sub in SUBDIRS:
                ext = EXT[sub]
                shutil.copy2(os.path.join(src, sub, stem + ext),
                             os.path.join(args.out, sub, f"{g:06d}" + ext))
            src_lbl = os.path.join(src, "labels", stem + ".txt")
            keep = []
            for line in open(src_lbl):
                t = line.split()
                if t and int(t[0]) == args.class_id:
                    keep.append("0 " + " ".join(t[1:]))
            if not keep:
                empty += 1
            n_box += len(keep)
            with open(os.path.join(args.out, "labels", f"{g:06d}.txt"), "w") as f:
                f.write("\n".join(keep) + ("\n" if keep else ""))
            g += 1

    n = g
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n"
                "train: images\nval: images\ntest: images\n"
                "nc: 1\nnames:\n  0: " + args.class_name + "\n")
    for meta in ("calibration.json", "metadata.json"):
        sp = os.path.join(args.sources[0], meta)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(args.out, meta))
    print(f"✅ {n} 帧 | 框 {n_box} | 空帧 {empty} ({empty/max(n,1):.0%}) | "
          f"每帧 {n_box/max(n,1):.2f} -> {args.out}")


if __name__ == "__main__":
    main()
