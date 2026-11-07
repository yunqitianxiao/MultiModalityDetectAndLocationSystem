#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
病集土建图分层切分 -> 裁贴图池 + 干净测试集 (杜绝泄漏)
================================================================
目的: 仿真训练要"全真实考"竞赛图, 但贴图本来就是从竞赛病集裁的。
    若测试图恰好也被裁过贴图进仿真训练, 测出的 recall 虚高 -> 不可信。
本脚本把病集里 32 张土建图(裂缝/渗漏/掉块)按类别分层切成两份:
  - TEXTURE 池 (~2/3): 喂给 prepare_textures --source bing 裁贴图 -> 进仿真训练
  - TEST 池   (~1/3): 单独留作测试, 不裁贴图、不进仿真; 三类都保证出现

产物:
  --stems 输出   tujian_test_stems.txt   (每行一个被跳过(留测)的图 stem)
  --out 输出     土建_3cls_test/         images/test + labels/test(重映射 6->0/7->1/5->2) + data.yaml
  同时打印切分结果供人工抽查(每图含哪几类)。

类别映射(官方 id -> 土建3类): 6=裂缝->0  7=渗漏水->1  5=管片掉块->2
用法:
  python3 data_gen/split_bing_tujian.py --n-test 10 \
      --stems data_gen/tujian_test_stems.txt --out data1/土建_3cls_test
"""
import argparse
import glob
import os
import random
import shutil
import sys

BING = os.path.expanduser("~/workspace/pro/2 (1)/病害训练集+正常状态/病害训练集")
GLOBAL_TO_LOCAL = {6: 0, 7: 1, 5: 2}        # crack/leakage/spalling -> 0/1/2
EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG")


def find_img(images_dir, stem):
    for e in EXTS:
        p = os.path.join(images_dir, stem + e)
        if os.path.isfile(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=10, help="留作测试的图数(默认 10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stems", default=None, help="输出 holdout stem 列表(.txt)")
    ap.add_argument("--out", default=None, help="测试集输出目录(缺省不复制图)")
    args = ap.parse_args()

    # 1) 收集 32 张土建图 + 每图含的类(裂缝/渗漏/掉块)
    imgs = []          # (stem, set(全局类id))
    for lp in sorted(glob.glob(os.path.join(BING, "labels", "*.txt"))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        clses = set()
        for line in open(lp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            c = int(line.split()[0])
            if c in GLOBAL_TO_LOCAL:
                clses.add(c)
        if not clses:            # 非土建图
            continue
        if find_img(os.path.join(BING, "images"), stem) is None:
            continue
        imgs.append((stem, clses))

    if len(imgs) < args.n_test:
        print(f"❌ 土建图仅 {len(imgs)} 张 < 需留测 {args.n_test}", file=sys.stderr)
        return

    # 2) 贪心分层选 test: 每步挑"最补最缺类别"的图, 保证三类都进 test 且均衡
    rng = random.Random(args.seed)
    imgs = list(imgs)
    rng.shuffle(imgs)
    test, pool = [], list(imgs)
    tc = {5: 0, 6: 0, 7: 0}
    while len(test) < args.n_test and pool:
        def score(it):
            _, cs = it
            # 对每类, 若 test 里它最少, 则该图含此类则加分; 得分=降序(最少类优先)
            least = min(tc.values())
            return (sum(1 for c in cs if tc[c] == least),
                    sum(1 for c in cs if tc[c] < least + 2))
        best = max(pool, key=score)
        test.append(best)
        pool.remove(best)
        for c in best[1]:
            tc[c] += 1

    test_stems = {stem for stem, _ in test}
    tex_pool = [stem for stem, _ in pool]

    # 3) 打印切分结果
    print("=" * 60)
    print(f"土建图共 {len(imgs)} 张 -> TEST {len(test)} / 贴图池 {len(tex_pool)}")
    print("\n[TEST 留测]  含类集分布:")
    for stem, cs in sorted(test):
        print(f"  {stem:32s} {sorted(GLOBAL_TO_LOCAL[c] for c in cs)}")
    print(f"\n[TEST 各类别覆盖]  裂缝{sum(1 for _,c in test if 6 in c)}  渗漏{sum(1 for _,c in test if 7 in c)}  掉块{sum(1 for _,c in test if 5 in c)}")
    print(f"[贴图池 各类别覆盖]  裂缝{sum(1 for _,c in pool if 6 in c)}  渗漏{sum(1 for _,c in pool if 7 in c)}  掉块{sum(1 for _,c in pool if 5 in c)}")

    # 4) 写 holdout stems
    if args.stems:
        with open(args.stems, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(test_stems)) + "\n")
        print(f"\n✅ holdout stems -> {args.stems}")

    # 5) 复制干净测试集 (原图 + 重映射框)
    if args.out:
        img_dir = os.path.join(BING, "images")
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.out, "test", sub), exist_ok=True)
        n_box = 0
        for stem, cs in test:
            src = find_img(img_dir, stem)
            if src is None:
                continue
            shutil.copy2(src, os.path.join(args.out, "test", "images", os.path.basename(src)))
            lines = []
            lp = os.path.join(BING, "labels", stem + ".txt")
            for line in open(lp, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                p = line.split()
                if int(p[0]) in GLOBAL_TO_LOCAL:
                    lines.append(f"{GLOBAL_TO_LOCAL[int(p[0])]} {p[1]} {p[2]} {p[3]} {p[4]}")
            with open(os.path.join(args.out, "test", "labels", stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n_box += len(lines)
        yaml = ("# 病集土建干净测试集 (split_bing_tujian.py, 与仿真训练贴图零重叠)\n"
                "path: " + os.path.abspath(args.out) + "\n"
                "train: test/images\nval: test/images\ntest: test/images\n"
                "names:\n  0: 裂缝\n  1: 渗漏水\n  2: 管片掉块\n")
        with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml)
        print(f"✅ 测试集 -> {args.out}  ({sum(1 for _ in test)} 图, {n_box} 框, 已重映射 0/1/2)")


if __name__ == "__main__":
    import sys
    main()
