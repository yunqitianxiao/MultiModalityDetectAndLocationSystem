#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
病集轨扣件图分层切分 -> 裁贴图池 + 干净测试集 (杜绝泄漏)
================================================================
病集扣件(25张)按类分层切成两份:
  - TEXTURE 池 (~2/3): 进仿真训练(裁贴图), 另 RFDD 提供大比例训练
  - TEST 池   (~1/3): 单独留作测试, 不裁贴图、不进仿真; 三类都出现

类别: 病集 0=扣件松动 / 1=扣件断裂 / 2=扣件缺失
映射到轨道 YOLO 类序(0断裂/1丢失/2松动歪斜): 病集0松动->2 病集1断裂->0 病集2缺失->1

产物:
  --stems 输出  track_test_stems.txt   (留测stem, 供 prepare_textures --skip-stems)
  --out 输出    轨道_3cls_test/        images/test + labels/test(重映射) + data.yaml

用法:
  python3 data_gen/split_bing_guijian.py --frac 0.33 \
      --stems data_gen/track_test_stems.txt --out data1/轨道_3cls_test
"""
import argparse
import glob
import os
import random
import shutil
import sys

BING = os.path.expanduser("~/workspace/pro/2 (1)/病害训练集+正常状态/病害训练集")
# 病集类 -> 轨道YOLO类序
BING_TRACK = {0: 2, 1: 0, 2: 1}          # 松动->松动歪斜2, 断裂->断裂0, 缺失->丢失1
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG", ".JPEG")


def find_img(stem):
    for e in EXTS:
        p = os.path.join(BING, "images", stem + e)
        if os.path.isfile(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.33, help="每类留测比例(默认 0.33)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stems", required=True, help="输出留测 stem 列表(track_test_stems.txt)")
    ap.add_argument("--out", required=True, help="测试集输出目录")
    args = ap.parse_args()

    # 1) 按类收集病集扣件图 (单类图)
    by_cls = {0: [], 1: [], 2: []}        # 病集类 -> [stem]
    for lp in sorted(glob.glob(os.path.join(BING, "labels", "*.txt"))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        name = None
        for line in open(lp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            c = int(line.split()[0])
            if c in BING_TRACK:
                name = c
                break
        if name is None or find_img(stem) is None:
            continue
        by_cls[name].append(stem)
    total = sum(len(v) for v in by_cls.values())

    # 2) 每类按比例留测(至少1), 确定性打乱
    rng = random.Random(args.seed)
    test, pool = [], []
    test_stems = []
    for c, stems in sorted(by_cls.items()):
        stems = list(stems)
        rng.shuffle(stems)
        n = max(1, round(len(stems) * args.frac))
        n = min(n, len(stems))
        for s in stems[:n]:
            test_stems.append(s)
        pool += stems[n:]
    test_stems = sorted(set(test_stems))
    pool = sorted(set(pool))

    # 3) 打印
    print("=" * 60)
    names = {0: "扣件松动", 1: "扣件断裂", 2: "扣件缺失"}
    print(f"病集扣件图共 {total} -> 留测 {len(test_stems)} / 贴图池 {len(pool)}")
    for c in sorted(by_cls):
        nt = sum(1 for s in test_stems if s in by_cls[c])
        print(f"  {names[c]:8s} 共{len(by_cls[c])} -> 留测 {nt} / 池 {len(by_cls[c])-nt}")
    print("\n[TEST 留测 stems]")
    print("  " + " ".join(test_stems))

    # 4) 写留测列表
    with open(args.stems, "w", encoding="utf-8") as f:
        f.write("\n".join(test_stems) + "\n")
    print(f"\n✅ 留测 stems -> {args.stems}")

    # 5) 复制干净测试集 (原图 + 重映射框)
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(args.out, "test", sub), exist_ok=True)
    n_box = 0
    for stem in test_stems:
        src = find_img(stem)
        shutil.copy2(src, os.path.join(args.out, "test", "images", os.path.basename(src)))
        lines = []
        for line in open(os.path.join(BING, "labels", stem + ".txt"), encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if int(p[0]) in BING_TRACK:
                lines.append(f"{BING_TRACK[int(p[0])]} {p[1]} {p[2]} {p[3]} {p[4]}")
        with open(os.path.join(args.out, "test", "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_box += len(lines)
    yaml = ("# 病集轨扣件干净测试集 (split_bing_guijian.py, 与仿真训练贴图零重叠)\n"
            "path: " + os.path.abspath(args.out) + "\n"
            "train: test/images\nval: test/images\ntest: test/images\n"
            "names:\n  0: 断裂\n  1: 丢失\n  2: 松动歪斜\n")
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)
    print(f"✅ 轨道测试集 -> {args.out}  ({len(test_stems)} 图, {n_box} 框, 重映射 0断裂/1丢失/2松动)")


if __name__ == "__main__":
    main()
