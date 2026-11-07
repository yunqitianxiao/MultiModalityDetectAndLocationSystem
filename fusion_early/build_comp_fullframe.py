#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅赛方病集 · 原生全幅贴满(不用任何外部真实库) —— 三支一次生成
=================================================================
原则(用户):
  · 只用赛方病集图(同分布), 不用 RFDD/土建_三病害/设备_yolo 等外部库
  · 病集图原生全幅 = 贴满(不裁剪不合成; 只把>1280的缩到1280)
  · 病集不全会用: 留一部分当 test(留测), 其余当 train
划分(已定):
  轨道 25 -> train20/test5  局部类 0:断裂 1:丢失 2:松动歪斜
  土建 32 -> train22/test10 局部类 0:裂缝 1:渗漏水 2:管片掉块
  设备 17 -> train12/test5  局部类 0:异物入侵 1:管线支架松脱
全局类: 0歪斜 1断裂 2缺失 3异物 4管线松脱 5管片破损 6裂缝 7渗漏水
=================================================================
"""
import os, glob, shutil
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
PRO3 = os.environ.get("PROJECT_ROOT", _HERE)
BD = os.environ.get("DETECTION_2D_DATA", os.path.join(_HERE, "..", "detection_2d", "病害训练集+正常状态", "病害训练集"))
IMG, LBL = os.path.join(BD, "images"), os.path.join(BD, "labels")
MAXDIM = 1280  # 最长边

BRANCHES = [
    {"name": "轨道_3cls", "cls": {0,1,2}, "map": {1:0, 2:1, 0:2},  # 0歪斜->2松动歪斜,1断裂->0,2缺失->1
     "names": ["断裂","丢失","松动歪斜"],
     "test": {"0004","0008","0104","0106","0201"}},
    {"name": "土建_3cls", "cls": {5,6,7}, "map": {6:0, 7:1, 5:2},  # 6裂缝->0,7渗漏->1,5掉块->2
     "names": ["裂缝","渗漏水","管片掉块"],
     "test": {"0506","0507","0510","0601","0602","0606","0610","0701","0704","0705"}},
    {"name": "设备_2cls", "cls": {3,4},  "map": {3:0, 4:1},        # 3异物->0,4管线松脱->1
     "names": ["异物入侵","管线支架松脱"],
     "test": {"0303","0307","0308","0310","0402"}},
]

def resize(im):
    H, W = im.shape[:2]
    m = max(H, W)
    if m > MAXDIM:
        s = MAXDIM / m
        return cv2.resize(im, (max(1,int(W*s)), max(1,int(H*s))), interpolation=cv2.INTER_AREA)
    return im

def labels_of(base):
    """返回 [(global_cls, cx,cy,w,h)]"""
    p = os.path.join(LBL, base + ".txt")
    if not os.path.exists(p): return []
    out = []
    for ln in open(p):
        t = ln.split()
        out.append((int(t[0]), float(t[1]), float(t[2]), float(t[3]), float(t[4])))
    return out

for br in BRANCHES:
    OUT = os.path.join(PRO3, br["name"] + "_comp")
    if os.path.exists(OUT): shutil.rmtree(OUT)
    for split in ("train", "test"):
        os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)
    # 收集该分支所有病集图
    bases = set()
    for f in glob.glob(os.path.join(LBL, "*.txt")):
        base = os.path.basename(f).replace(".txt", "")
        if any(c in br["cls"] for c, *_ in labels_of(base)):
            bases.add(base)
    n_tr = n_te = 0
    for base in sorted(bases, key=lambda x:(len(x), x)):
        split = "test" if base in br["test"] else "train"
        src = glob.glob(os.path.join(IMG, base + ".*"))
        if not src: continue
        im = cv2.imread(src[0])
        if im is None: continue
        im = resize(im)
        cv2.imwrite(f"{OUT}/images/{split}/{base}.jpg", im)
        lines = []
        for g, cx, cy, w, h in labels_of(base):
            if g in br["map"]:
                lines.append(f"{br['map'][g]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(f"{OUT}/labels/{split}/{base}.txt", "w") as f:
            f.write("\n".join(lines))
        if split == "test": n_te += 1
        else: n_tr += 1
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write("# 仅赛方病集·原生全幅贴满; test=病集留测真图零重叠\n"
                "# 数据极小无法再分val, 用 test 兼任 val(早停略偏, 以 test 报告为准)\n"
                f"path: {OUT}\n"
                f"train: images/train\nval: images/test\ntest: images/test\n"
                f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(br["names"])))

    print(f"[{br['name']}] train={n_tr} test={n_te} -> {OUT}")
    print(f"   tags: {br['names']}")
