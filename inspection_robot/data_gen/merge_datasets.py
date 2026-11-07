#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多 seed 配对数据集合并 -> 一个大数据集 (帧全局重编号 + 重建 data.yaml)
================================================================
多个 run_data_gen.sh seed 跑出 output_data1_bing_s0..s9, 每份都有独立的
images/labels/clouds/attrmaps/poses + defects.json。本脚本把它们 + 原始
output_data1_bing 串起来, 帧号按源(字典序)依次重编号 000000..N, 使训练能
在一份更大的、跨不同缺陷世界的集合上进行(独立场景数翻倍 -> 召回).

规则:
  - 每源只取"五路齐全"的帧(images/labels/clouds/attrmaps/poses 都存在的帧号),
    自动排掉单路缺帧(如某源 attrmaps 多出的一帧无图).
  - data.yaml 重写 path(指向合并目录), nc/names 沿用第一源.
  - calibration.json / metadata.json 从第一源拷, 各源标定一致才有效.

用法:
    python3 data_gen/merge_datasets.py --out data_gen/output_data1_bing_big \
        data_gen/output_data1_bing data_gen/output_data1_bing_s0 data_gen/output_data1_bing_s1 ...
"""
import argparse
import json
import os
import shutil
import sys

SUBDIRS = ["images", "labels", "clouds", "attrmaps", "poses"]
EXT = {"images": ".jpg", "labels": ".txt", "clouds": ".npy",
       "attrmaps": ".npz", "poses": ".json"}


def valid_frames(root: str) -> list[str]:
    """返回 root 下五路齐全的帧号 stem 列表(排序)。"""
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
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def main() -> None:
    ap = argparse.ArgumentParser(description="合并多 seed 配对数据集")
    ap.add_argument("--out", required=True, help="合并输出目录")
    ap.add_argument("sources", nargs="+", help="源数据集目录(按字典序遍历)")
    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)
    for sub in SUBDIRS:
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    g = 0                               # 全局帧号
    per_src = {}
    for src in args.sources:
        frames = valid_frames(src)
        per_src[src] = len(frames)
        for stem in frames:
            for sub in SUBDIRS:
                ext = EXT[sub]
                sp = os.path.join(src, sub, stem + ext)
                dp = os.path.join(out, sub, f"{g:06d}" + ext)
                shutil.copy2(sp, dp)
            g += 1
    n = g
    print(f"✅ 合并完成: {n} 帧 <- {len(args.sources)} 源")
    for src, cnt in per_src.items():
        print(f"   {os.path.basename(src):24s} {cnt:4d} 帧")
    print(f"   共计 {n} 帧 -> {out}")

    # ---- 用第一源重建 data.yaml (正确解析 names: 多行块) ----
    src0 = args.sources[0]
    yaml_path = os.path.join(src0, "data.yaml")
    nc = 8
    names_block = []                     # [(index, name)]
    if os.path.isfile(yaml_path):
        in_names = False
        for line in open(yaml_path, encoding="utf-8"):
            raw = line.rstrip("\n")
            if raw.startswith("nc:"):
                nc = int(raw.split(":", 1)[1].strip())
            elif raw.startswith("names:"):
                inline = raw.split(":", 1)[1].strip()
                if inline:               # 行内式 names: [..] / {..}
                    names_block = [(i, n.strip().strip("'\" "))
                                   for i, n in enumerate(inline.strip("[]{}").split(","))
                                   if n.strip()]
                    in_names = False
                else:                    # 多行块, 待收集下面的缩进行
                    in_names = True
            elif in_names and raw.startswith((" ", "\t")):
                parts = raw.split(":", 1)
                if len(parts) == 2:
                    names_block.append((int(parts[0].strip()), parts[1].strip()))
            else:
                in_names = False
    names_ordered = [n for _, n in sorted(names_block, key=lambda kv: kv[0])]
    if not names_ordered:                # 兜底: 有 nc 则用占位名
        names_ordered = [str(i) for i in range(nc)]
    lines = [
        "path: " + os.path.abspath(out),
        "train: images",
        "val: images",
        "test: images",
        f"nc: {nc}",
        "names:",
    ]
    for i, nm in enumerate(names_ordered):
        lines.append(f"  {i}: {nm}")
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---- 拷贝标定/元数据 ----
    for meta in ("calibration.json", "metadata.json"):
        sp = os.path.join(src0, meta)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(out, meta))
    print(f"✅ data.yaml 已重建 (nc={nc}), calibration/metadata 取自 {os.path.basename(src0)}")


if __name__ == "__main__":
    main()
