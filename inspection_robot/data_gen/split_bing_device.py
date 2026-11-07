#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
病集设备图分层切分 -> 留测(不参与生成) + 贴图池 (杜绝泄漏)
================================================================
仿 split_bing_tujian/rail.py: 把病集设备(异物入侵/管线支架松脱)图按类分层抽 --test-frac 作"留测",
其余进贴图池喂 prepare_textures --source bing --skip-stems 裁贴图 -> 进仿真训练。
保证: 留测图不裁贴图, 仿真训练贴图与测试真图零重叠(否则 recall 虚高)。

类映射(病集全局 -> 设备局部):
   病集全局: 3=异物入侵(yiwu)  4=管线支架松脱(guanxiansongtuo)
   设备局部: 0=异物入侵  1=管线支架松脱
   => BING_TO_LOCAL = {3:0, 4:1}

产物:
  --stems   device_test_stems.txt   每行一个留测 stem (喂 run_data_gen.sh 的 SKIP_STEMS_FILE)
  --out     真图测试集 <out>/images/test + <out>/labels/test (重映射 0/1)
  同时打印留测分布供人工抽查

用法:
  python3 data_gen/split_bing_device.py --test-frac 0.33 \
      --stems data_gen/device_test_stems.txt --out /home/ylf/workspace/pro/3/设备_2cls_sim
"""
import argparse
import glob
import os
import random
import shutil

BING = os.path.expanduser("~/workspace/pro/2 (1)/病害训练集+正常状态/病害训练集")
BING_TO_LOCAL = {3: 0, 4: 1}        # 异物->0, 支架松脱->1
DEVICE_GLOBAL = (3, 4)
EXTS = (".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG")
CLASS_NAME = {0: "异物入侵", 1: "管线支架松脱"}


def find_img(images_dir, stem):
    for e in EXTS:
        p = os.path.join(images_dir, stem + e)
        if os.path.isfile(p):
            return p
    return None


def num_test(count, frac):
    return min(max(1, round(count * frac)), count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.33, help="每类留测比例(默认 0.33)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stems", default=None, help="输出 holdout stem 列表(.txt)")
    ap.add_argument("--out", default=None, help="最终数据集根目录(写 images/test + labels/test)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    imgs_by_stem, by_class = {}, {c: [] for c in DEVICE_GLOBAL}
    for lp in sorted(glob.glob(os.path.join(BING, "labels", "*.txt"))):
        stem = os.path.splitext(os.path.basename(lp))[0]
        clses = set()
        for line in open(lp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if len(p) == 5 and int(p[0]) in DEVICE_GLOBAL:
                clses.add(int(p[0]))
        if not clses:
            continue
        if find_img(os.path.join(BING, "images"), stem) is None:
            continue
        imgs_by_stem[stem] = clses
        for c in clses:
            by_class[c].append(stem)
    if not imgs_by_stem:
        print("❌ 未找到任何设备图", file=os.sys.stderr)
        return
    by_class = {c: sorted(set(v)) for c, v in by_class.items()}
    print(f"设备图共 {len(imgs_by_stem)} 张, 各类分布: "
          + ", ".join(f"id{c}={len(by_class[c])}" for c in DEVICE_GLOBAL))

    test_stems, seen = [], set()
    for c in DEVICE_GLOBAL:
        cand = [s for s in by_class[c] if s not in seen]
        rng.shuffle(cand)
        n = num_test(len(by_class[c]), args.test_frac)
        for s in cand[:n]:
            test_stems.append(s)
            seen.add(s)
    test_stems = sorted(set(test_stems))
    tex_pool = sorted(set(imgs_by_stem) - set(test_stems))

    print("=" * 60)
    print(f"设备图 {len(imgs_by_stem)} 张 -> TEST {len(test_stems)} / 贴图池 {len(tex_pool)}")
    print("[TEST 留测]")
    for stem in test_stems:
        cs = imgs_by_stem[stem]
        print(f"  {stem:8s} 含类 {sorted(cs)} -> 局部 {sorted(BING_TO_LOCAL[c] for c in cs)}")

    def _cov(stems):
        cov = {0: 0, 1: 0}
        for s in stems:
            for c in imgs_by_stem[s]:
                cov[BING_TO_LOCAL[c]] += 1
        return cov

    tc, pc = _cov(test_stems), _cov(tex_pool)
    print("[TEST 各类覆盖](局部) " + "  ".join(f"{CLASS_NAME[k]}={tc[k]}" for k in (0, 1)))
    print("[贴图池 各类覆盖](局部) " + "  ".join(f"{CLASS_NAME[k]}={pc[k]}" for k in (0, 1)))

    if args.stems:
        with open(args.stems, "w", encoding="utf-8") as f:
            f.write("\n".join(test_stems) + "\n")
        print(f"\n✅ holdout stems -> {args.stems}")

    if args.out:
        img_dir = os.path.join(BING, "images")
        t_img = os.path.join(args.out, "images", "test")
        t_lab = os.path.join(args.out, "labels", "test")
        os.makedirs(t_img, exist_ok=True)
        os.makedirs(t_lab, exist_ok=True)
        n_box = 0
        for stem in test_stems:
            src = find_img(img_dir, stem)
            if src is None:
                continue
            shutil.copy2(src, os.path.join(t_img, os.path.basename(src)))
            lines = []
            lp = os.path.join(BING, "labels", stem + ".txt")
            for line in open(lp, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                p = line.split()
                if int(p[0]) in BING_TO_LOCAL:
                    lines.append(f"{BING_TO_LOCAL[int(p[0])]} {p[1]} {p[2]} {p[3]} {p[4]}")
            with open(os.path.join(t_lab, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n_box += len(lines)
        print(f"✅ 测试集(真图) -> {args.out}/test ({len(test_stems)} 图, {n_box} 框, 重映射 0/1)")


if __name__ == "__main__":
    main()
