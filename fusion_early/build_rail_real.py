#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨道分支 · 真实数据重建(替代 cut-paste) —— 目标取景 = 单扣件特写(贴满)
=====================================================================
数据源:
  A) Railway Fastener Defect Dataset (RFDD) 1250张 6类俯视双轨图
     逐扣件裁成单扣特写, 使扣件占满画面 ~60%  (match 赛方测试单扣占帧)
     类别映射(全局RFDD -> 本地轨道3类): 本地类序 0:断裂 1:丢失 2:松动歪斜
       1(DL 断裂)->0 断裂   (高置信 duanLie)
       2(DS 丢失)->1 丢失   (高置信 diuShi)
       0(BX 变形/歪斜)->2 松动歪斜 (中置信; 同框核对见下)
       4(N  正常)-> 负样本(无框, 抑制误检); 从中抽样控制量
       3(FZ 翻转)/5(YW 异物)-> 舍弃 (映射不确定, 防毒化)
  B) 赛方病集 轨道图(歪斜10/断裂10/缺失5) 原生全幅=贴满, 全局类已对好
       全局类 -> 本地: 1(断裂)->0, 2(缺失/丢失)->1, 0(歪斜)->2
       排除 5 张留测(0004/0008/0104/0106/0201) 作为 test(零重叠)

输出: pro/3/轨道_3cls_real/{images,labels}/{train,val,test}
=====================================================================
"""
import os, glob, random, shutil
import cv2, numpy as np
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED)

PRO3     = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
OUT      = os.path.join(PRO3, "轨道_3cls_real")
_DET_ROOT = os.environ.get("DETECTION_2D_ROOT", os.path.join(PRO3, "..", "detection_2d"))
RFDD     = os.path.join(_DET_ROOT, "铁轨紧固件损坏检测数据集 YOLO格式", "train&val", "YOLO")
COMP_IMG = os.path.join(_DET_ROOT, "病害训练集+正常状态", "病害训练集", "images")
COMP_LBL = os.path.join(_DET_ROOT, "病害训练集+正常状态", "病害训练集", "labels")
TEST_BASE = {"0004", "0008", "0104", "0106", "0201"}   # 留测(从病集排除)

# RFDD 全局类 -> (本地类 or None=负样本 or None)
# 本地: 0=断裂, 1=丢失, 2=松动歪斜
RFDD_TO_LOCAL = {0: 2, 1: 0, 2: 1}          # BX/变形->松动歪斜, DL->断裂, DS->丢失
RFDD_NORMAL   = 4                            # N 正常 -> 负样本
RFDD_DROP     = {3, 5}                       # FZ/YW 舍弃
COMP_TO_LOCAL = {1: 0, 2: 1, 0: 2}           # 断裂->0, 缺失/丢失->1, 歪斜->2

def norm_to_pixel(cx, cy, w, h, W, H):
    return (cx - w/2) * W, (cy - h/2) * H, w * W, h * H

def crop_center_square(img, cpx, cpy, mdim, fill=0.60, size=640):
    """以扣件中心裁方形, 使扣件占画面 fill 比例; 越界用边缘垫. 返回 size×size BGR."""
    H, W = img.shape[:2]
    side = int(mdim / fill)
    x0, y0 = int(cpx - side/2), int(cpy - side/2)
    # 裁取(含越界垫)
    x1, y1 = x0 + side, y0 + side
    patch = np.full((side, side, 3), 30, np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    if sx1 > sx0 and sy1 > sy0:
        sub = img[sy0:sy1, sx0:sx1]
        patch[sy0-y0:sy0-y0+sub.shape[0], sx0-x0:sx0-x0+sub.shape[1]] = sub
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)

def load_rfdd(defects, normals, max_neg):
    """扫 RFDD, 生成 特写 crop(缺陷) 与 crop(正常负样本)."""
    n_inst = {"defect": 0, "normal": 0, "drop": 0}
    for img_path in sorted(glob.glob(os.path.join(RFDD, "images", "*"))):
        img = cv2.imread(img_path)
        if img is None: continue
        H, W = img.shape[:2]
        lbl = os.path.join(RFDD, "labels", os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        if not os.path.exists(lbl): continue
        boxes = [ln.split() for ln in open(lbl) if ln.strip()]
        for b in boxes:
            cls = int(b[0]); cx, cy, w, h = map(float, b[1:])
            mdim = max(w*W, h*H)
            px, py = cx*W, cy*H
            if cls in RFDD_TO_LOCAL:
                crop = crop_center_square(img, px, py, mdim)
                defects[RFDD_TO_LOCAL[cls]].append(crop); n_inst["defect"] += 1
            elif cls == RFDD_NORMAL:
                if len(normals) < max_neg:
                    normals.append(crop_center_square(img, px, py, mdim))
                    n_inst["normal"] += 1
            elif cls in RFDD_DROP:
                n_inst["drop"] += 1
    return n_inst

def load_competition():
    """赛方病集轨道图, 排除留测 -> train 全幅贴满. 返回 {local_cls:[(bgr, box)]}."""
    rail = {0: [], 1: [], 2: []}
    for img_path in sorted(glob.glob(os.path.join(COMP_IMG, "*"))):
        base = os.path.splitext(os.path.basename(img_path))[0]
        if base in TEST_BASE: continue
        lbl = os.path.join(COMP_LBL, base + ".txt")
        if not os.path.exists(lbl): continue
        img = cv2.imread(img_path)
        if img is None: continue
        H, W = img.shape[:2]
        boxes = [ln.split() for ln in open(lbl) if ln.strip()]
        mapped = [(COMP_TO_LOCAL[int(b[0])], b) for b in boxes if int(b[0]) in COMP_TO_LOCAL]
        if not mapped: continue
        # 缩放至最长边<=1500 保持比例
        old = (W, H)
        if max(W, H) > 1500:
            s = 1500 / max(W, H); img = cv2.resize(img, (int(W*s), int(H*s))); H, W = img.shape[:2]
        for cls, b in mapped:
            rail[cls].append((img, [float(x) for x in b[1:]], (W, H), base))
    return rail

def write_one(img, cls, out_img, out_lbl, empty_lbl=False):
    p = out_img + ".jpg"
    cv2.imwrite(p, img)
    if not empty_lbl:
        with open(out_lbl + ".txt", "w") as f:
            f.write(f"{cls} 0.5 0.5 0.5 0.5\n")
    return p

def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for sp in ("train", "val", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(OUT, sub, sp), exist_ok=True)

    # ---- A) RFDD 单扣特写 ----
    defects = {0: [], 1: [], 2: []}; normals = []
    n = load_rfdd(defects, normals, max_neg=900)
    print(f"[RFDD] 缺陷crop: 断裂{len(defects[0])} 丢失{len(defects[1])} 松动歪斜{len(defects[2])} "
          f"| 正常负sample(用于负样本){len(normals)} | 舍弃FZ/YW {n['drop']}")
    # ---- B) 赛方病集全幅 ----
    comp = load_competition()

    # ---- 汇入 train/val ----
    all_items = []   # (img, cls, is_neg(skip label), tag)
    for cls in (0, 1, 2):
        for crop in defects[cls]:
            all_items.append((crop, cls, False, "rfdd"))
    for crop in normals:
        all_items.append((crop, -1, True, "rfdd_neg"))
    for cls in (0, 1, 2):
        for (img, bbox, dim, base) in comp[cls]:
            all_items.append((img, cls, False, "comp"))
    random.shuffle(all_items)
    n_val = int(0.15 * len(all_items))
    val_pool, train_pool = all_items[:n_val], all_items[n_val:]

    def dump(pool, split):
        for i, (img, cls, is_neg, tag) in enumerate(pool):
            stem = f"{tag}_{cls}_{i:05d}"
            write_one(img, cls if not is_neg else 0,
                      os.path.join(OUT, "images", split, stem),
                      os.path.join(OUT, "labels", split, stem),
                      empty_lbl=is_neg)
            # 补偿: RFDD裁剪补一张水平翻转(特写池量)
            if tag == "rfdd" and not is_neg:
                fl = cv2.flip(img, 1)
                stem2 = f"{tag}_f_{cls}_{i:05d}"
                write_one(fl, cls, os.path.join(OUT, "images", split, stem2),
                          os.path.join(OUT, "labels", split, stem2))
    dump(val_pool, "val"); dump(train_pool, "train")

    # ---- test: 病集留测5张(真图+映射标签) ----
    n_test = 0
    for base in sorted(TEST_BASE):
        img_path = glob.glob(os.path.join(COMP_IMG, base + ".*"))
        lbl_path = os.path.join(COMP_LBL, base + ".txt")
        if not (img_path and os.path.exists(lbl_path)): continue
        img = cv2.imread(img_path[0])
        if img is None: continue
        lines = []
        for ln in open(lbl_path):
            p = ln.split(); g = int(p[0])
            if g in COMP_TO_LOCAL:
                lines.append(f"{COMP_TO_LOCAL[g]} " + " ".join(p[1:]))
        p = os.path.join(OUT, "images", "test", base + ".jpg")
        cv2.imwrite(p, img)
        with open(os.path.join(OUT, "labels", "test", base + ".txt"), "w") as f:
            f.write("\n".join(lines))
        n_test += 1

    # ---- data.yaml ----
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"""# 轨道3类 真实数据 (RFDD单扣特写+赛方病集全幅; test=病集留测真图零重叠)
path: {OUT}
train: images/train
val: images/val
names:
  0: 断裂
  1: 丢失
  2: 松动歪斜
test: images/test
""")
    def cnt(pool):
        c = {"断裂":0,"丢失":0,"松动歪斜":0,"正常":0}
        for img,cls,neg,tag in pool:
            c["正常" if neg else ["断裂","丢失","松动歪斜"][cls]] += 1
        return c
    print("[汇总] train:", cnt(train_pool), " val:", cnt(val_pool), " test:", n_test)
    print("OUT =", OUT)

if __name__ == "__main__":
    main()
