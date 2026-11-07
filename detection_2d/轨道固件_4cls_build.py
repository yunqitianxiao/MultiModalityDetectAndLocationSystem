#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨道固件(扣件) 6类 -> 4类 均衡数据集构建
旧6类: 0变形 1断裂 2丢失 3反装 4正常 5移位
新4类: 0正常 1断裂 2丢失 3松动歪斜
  正常      = 旧4
  断裂      = 旧1
  丢失      = 旧2
  松动歪斜  = 旧0(变形) + 旧3(反装) + 旧5(移位)
均衡策略: 保留全部缺陷框(断裂199+丢失201+松动歪斜600=1000)，
         正常框(旧4)从全部里抽稀到 NORMAL_TARGET(默认1000)，与缺陷1:1。
划分: 按缺陷类别分层 train/val/test。
输出: 4类label + data.yaml 用的路径 + split清单(图片文件名->split)。
"""
import os, random, shutil, collections, glob

BASE = "/home/ylf/workspace/pro/2/data1/轨道固件"
STAGE = "/home/ylf/workspace/pro/2/data1/轨道固件_4cls_stage"
NORMAL_TARGET = 1000          # 正常框目标(与缺陷总数1:1)
SEED = 42
R = random.Random(SEED)

# 旧 -> 新
REMAP = {0: 3, 1: 1, 2: 2, 3: 3, 4: 0, 5: 3}
DEFECT_OLD = {1: "断裂", 2: "丢失", 0: "松动歪斜", 3: "松动歪斜", 5: "松动歪斜"}

def collect(label_dir):
    """返回 list[(label_path, image_path, stem)]"""
    out = []
    for lp in glob.glob(os.path.join(label_dir, "*.txt")):
        stem = os.path.splitext(os.path.basename(lp))[0]
        # 图片同名，扩展名可能 png/jpg
        ip = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(os.path.dirname(label_dir), "images", stem + ext)
            if os.path.exists(cand):
                ip = cand; break
        if ip is None:
            raise FileNotFoundError(f"找不到图片: {stem}")
        out.append((lp, ip, stem))
    return out

def read_boxes(lp):
    boxes = []
    with open(lp) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            p = line.split()
            if len(p) != 5: continue
            boxes.append((int(p[0]), p[1], p[2], p[3], p[4]))
    return boxes

def write_label(path, boxes):
    with open(path, "w") as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]} {b[2]} {b[3]} {b[4]}\n")

def main():
    os.makedirs(STAGE, exist_ok=True)

    # 1. 收集所有样本
    samples = collect(os.path.join(BASE, "test/labels")) + \
              collect(os.path.join(BASE, "train&val/YOLO/labels"))
    print(f"总样本: {len(samples)}")

    # 2. 重映射 + 收集正常框与缺陷框
    all_boxes = {}          # stem -> [(newcls, x,y,w,h)]
    normal_spots = []       # (stem, box)  待抽稀的正常框
    defect_boxes = []       # (stem, box)  全部缺陷框
    for lp, ip, stem in samples:
        bs = read_boxes(lp)
        newb = []
        for olds in bs:
            nc = REMAP[olds[0]]
            newb.append((nc, olds[1], olds[2], olds[3], olds[4]))
            if nc == 0:
                normal_spots.append((stem, newb[-1]))
            else:
                defect_boxes.append((stem, newb[-1]))
        all_boxes[stem] = newb

    n_def = len(defect_boxes)
    print(f"缺陷框总数: {n_def}  |  正常框总数: {len(normal_spots)}")

    # 3. 抽稀正常框到 NORMAL_TARGET
    R.shuffle(normal_spots)
    keep_normal = normal_spots[:NORMAL_TARGET]
    keep_set = collections.defaultdict(list)   # stem -> [normal boxes]
    for stem, b in keep_normal:
        keep_set[stem].append(b)
    print(f"抽稀后正常框: {len(keep_normal)}  (目标 {NORMAL_TARGET})")

    # 4. 组装每张图的最终框(缺陷框全保留 + 抽稀后的正常框)，并按缺陷类别分组
    group_images = collections.defaultdict(list)   # defect_class -> [stem]
    final = {}
    for stem in all_boxes:
        final_boxes = [b for b in all_boxes[stem] if b[0] != 0] + keep_set.get(stem, [])
        final[stem] = final_boxes
        # 该图属于哪个缺陷组(取第一个缺陷)，无缺陷则纯正常
        cls = None
        for b in all_boxes[stem]:
            if b[0] != 0:
                cls = {1: "断裂", 2: "丢失", 3: "松动歪斜"}[b[0]]; break
        group_images[cls or "正常"].append(stem)

    for k, v in group_images.items():
        print(f"  组[{k}]: {len(v)} 图")

    # 5. 分层划分 train/val/test
    split_of = {}
    for g, stems in group_images.items():
        R.shuffle(stems)
        n = len(stems)
        n_tr = int(n * 0.8); n_va = int(n * 0.1)
        for i, s in enumerate(stems):
            if i < n_tr: split_of[s] = "train"
            elif i < n_tr + n_va: split_of[s] = "val"
            else: split_of[s] = "test"

    # 6. 写4类label到临时目录(按split) + 生成整理manifest
    img_dir = os.path.join(STAGE, "images")
    lab_dir = os.path.join(STAGE, "labels")
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(img_dir, sp), exist_ok=True)
        os.makedirs(os.path.join(lab_dir, sp), exist_ok=True)

    img_split = map  # placeholder
    with open(os.path.join(STAGE, "_split_manifest.txt"), "w") as mf:
        for stem, sp in split_of.items():
            mf.write(f"{sp}\t{stem}\n")
            # 写标签(轻量)
            write_label(os.path.join(lab_dir, sp, stem + ".txt"), final[stem])

    # 7. 统计每split每类 框数/图数
    print("\n=== 各split类别实例计数 ===")
    for sp in ("train", "val", "test"):
        cnt = collections.Counter()
        icnt = collections.Counter()
        for stem, s in split_of.items():
            if s != sp: continue
            boxes = final[stem]
            has = set(b[0] for b in boxes)
            for h in has: icnt[h] += 1
            for b in boxes: cnt[b[0]] += 1
        print(f"  {sp}: 图{len([1 for s in split_of.values() if s==sp])} | "
              f"实例 正常{cnt[0]} 断裂{cnt[1]} 丢失{cnt[2]} 松动歪斜{cnt[3]}")

    print("\n=== 全局实例计数 ===")
    tot = collections.Counter()
    for stem in final:
        for b in final[stem]: tot[b[0]] += 1
    print(f"  正常{tot[0]} 断裂{tot[1]} 丢失{tot[2]} 松动歪斜{tot[3]}  (总{sum(tot.values())})")

    with open(os.path.join(STAGE, "_class_dist.txt"), "w") as f:
        f.write(f"normal={tot[0]} fracture={tot[1]} lost={tot[2]} loose={tot[3]}\n")

    print(f"\n完成。标签与split清单写入: {STAGE}")

if __name__ == "__main__":
    main()
