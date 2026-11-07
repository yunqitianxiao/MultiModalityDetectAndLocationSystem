#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
土建 A+B: 仅赛方病集 · 保分辨率(cap=1920) + 增强补量
  A 保分辨率: 生成 cap=1920 (旧 1280 丢细节); 训练时配 --imgsz 1280 (细裂缝才有像素)
  B 增强补量 + 类均衡: 翻转/亮度抖动(盒安全), 掉块×4 渗漏×3 裂缝×2 过采样
  test: 10 张留测原样(cap 1920), 不增强; 局部类 0裂缝1渗漏水2管片掉块
=================================================================
"""
import os, glob, shutil, random
import cv2, numpy as np

random.seed(42); np.random.seed(42)
_HERE = os.path.dirname(os.path.abspath(__file__))
PRO3 = os.environ.get("PROJECT_ROOT", _HERE)
BD = os.environ.get("DETECTION_2D_DATA", os.path.join(_HERE, "..", "detection_2d", "病害训练集+正常状态", "病害训练集"))
IMG, LBL = os.path.join(BD, "images"), os.path.join(BD, "labels")
OUT = os.path.join(PRO3, "土建_3cls_comp_aug")
CAP = 1920
TEST = {"0506","0507","0510","0601","0602","0606","0610","0701","0704","0705"}
MAP  = {6:0, 7:1, 5:2}   # 全局->本地(0裂缝/1渗漏/2掉块)
LOCAL_NAMES = ["裂缝","渗漏水","管片掉块"]

def labels_of(base):
    p = os.path.join(LBL, base + ".txt")
    out = []
    if os.path.exists(p):
        for ln in open(p):
            t = ln.split()
            g = int(t[0])
            if g in MAP:
                out.append((MAP[g], float(t[1]), float(t[2]), float(t[3]), float(t[4])))
    return out

def cap(im):
    H, W = im.shape[:2]; m = max(H, W)
    if m > CAP:
        s = CAP / m
        return cv2.resize(im, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
    return im

def variants(im, boxes, n):
    """n 个盒安全变体: 奇数索引做水平翻转, 全部做亮度/对比度抖动."""
    res = []
    for k in range(n):
        img, bx = im, [list(b) for b in boxes]
        if k % 2 == 1:
            img = cv2.flip(img, 1)
            bx = [(c, 1-x, y, w, h) for (c, x, y, w, h) in bx]
        a = random.uniform(0.85, 1.15); br = random.uniform(-18, 18)
        img = cv2.convertScaleAbs(img, alpha=a, beta=br)
        res.append((img, bx))
    return res

def count_of(boxes, cls):
    return sum(1 for b in boxes if b[0] == cls)

if os.path.exists(OUT):
    shutil.rmtree(OUT)
for split in ("train", "test"):
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)

# 土建所有病集图
bases = set()
for f in glob.glob(os.path.join(LBL, "*.txt")):
    base = os.path.basename(f).replace(".txt", "")
    if labels_of(base):
        bases.add(base)

n_tr = n_te = 0
cls_count = {0:0, 1:0, 2:0}
for base in sorted(bases, key=lambda x:(len(x), x)):
    if base in TEST:
        split = "test"
    else:
        split = "train"
    src = glob.glob(os.path.join(IMG, base + ".*"))
    if not src: continue
    im = cv2.imread(src[0])
    if im is None: continue
    im = cap(im)
    boxes = labels_of(base)
    if split == "test":
        cv2.imwrite(f"{OUT}/images/test/{base}.jpg", im)
        with open(f"{OUT}/labels/test/{base}.txt", "w") as f:
            f.write("\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c,x,y,w,h in boxes))
        n_te += 1
        continue
    # train: 确定过采样倍数
    if count_of(boxes, 2) > 0:   K = 4   # 掉块
    elif count_of(boxes, 1) > 0: K = 3   # 渗漏
    else: K = 2                        # 裂缝
    for vi, (img, bx) in enumerate(variants(im, boxes, K)):
        stem = f"{base}_v{vi}"
        cv2.imwrite(f"{OUT}/images/train/{stem}.jpg", img)
        with open(f"{OUT}/labels/train/{stem}.txt", "w") as f:
            f.write("\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c,x,y,w,h in bx))
        for c, *_ in bx: cls_count[c] += 1
        n_tr += 1

with open(os.path.join(OUT, "data.yaml"), "w") as f:
    f.write("# 土建仅赛方病集·保分辨率1920+增强过采样; test=病集留测真图零重叠\n"
            "# val=test(数据小); 训练请配 --imgsz 1280\n"
            f"path: {OUT}\n"
            f"train: images/train\nval: images/test\ntest: images/test\n"
            f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(LOCAL_NAMES)))

print(f"[土建_3cls_comp_aug] train图={n_tr} test图={n_te}")
print(f"  本地类实例分布: 裂缝{cls_count[0]} 渗漏水{cls_count[1]} 管片掉块{cls_count[2]}")
