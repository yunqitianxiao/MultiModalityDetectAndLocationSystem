#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实背景 cut-paste 检测集生成器（替换"灰盒子仿真"）
================================================================================
背景 = 真实隧道正常状态照片（正常状态/{轨道,土建,设备}），缺陷 = 病集按框裁出的
羽化 RGBA 贴图。二者都是真实照片域的像素，合成结果天然落在真实照片域，
从而避免纯仿真（干净灰盒+矩形贴片）在真实病检图上 R=0 的域鸿沟。

每支生成一份 train/val，保留数据集里已就位的"留测真实 test"（不碰）。
生成器只填 train/val + 重写 data.yaml。

缺陷贴图来源：本脚本先调用 prepare_from_bing（--skip-stems 留测列表）按病集
YOLO 框裁缺陷 -> 512x512 羽化 RGBA。这与 old 仿真共用同一套贴图逻辑。

类别映射（病集全局 -> 分支局部）：
  土建: {6->0 裂缝, 7->1 渗漏, 5->2 掉块}
  轨道: {1->0 断裂, 2->1 缺失, 0->2 松动歪斜}
  设备: {3->0 异物, 4->1 支架松脱}

用法:
  python3 data_gen/build_cp_cutpaste.py --branch 轨道 --preview 6 --out /tmp/preview_rail
  python3 data_gen/build_cp_cutpaste.py --branch 土建 --n 1200 \
     --out /home/ylf/workspace/pro/3/土建_3cls_sim
"""
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NZ = os.path.expanduser("~/workspace/pro/2 (1)/病害训练集+正常状态/正常状态")
BING = os.path.expanduser("~/workspace/pro/2 (1)/病害训练集+正常状态/病害训练集")
COMBINED_SKIP_23 = os.path.join(ROOT, "data_gen", "combined_skip_23.txt")

MAX_DIM = 1280        # 背景工作尺寸（最长边），保比例
VAL_FRAC = 0.15
SEED = 42

# 分支: 局部类 -> [局部类名, 病集全局id, (w范围), (h范围)]
# 关键：取景必须贴合留测真图("特写大缺陷" 轨道 / "整墙多缺陷" 土建 / "近景异物" 设备)，
#       否则模型学到"广角小贴片"，在特写测试图上 IoU 匹配不上 -> R=0。
#   zoom = 背景随机裁剪保留比例 (lo,hi)。越小=越特写(裁掉外围, 等效放大)。
#     轨道要特写(0.30~0.55)；土建整墙填满(0.70~1.0)；设备近景(0.55~0.85)。
#   k    = 每张缺陷数 (lo,hi)。土建整墙多个(4~8)；轨道/设备单/双(1~2)。
#   pos  = 归一化 (x范围, y范围)；第三元 True=侧带(避开中线)。
#   size 按"裁剪后帧"归一：轨道大缺陷居中；土建中缺陷散布；设备中缺陷近景。
BRANCHES = {
    "轨道": {
        "names": ["断裂", "丢失", "松动歪斜"],
        "classes": {
            0: ("rail_fastener_broken", {1}, (0.22, 0.45), (0.18, 0.40)),
            1: ("rail_fastener_missing", {2}, (0.22, 0.45), (0.18, 0.40)),
            2: ("rail_fastener_loose",   {0}, (0.20, 0.42), (0.16, 0.38)),
        },
        "zoom": (0.30, 0.55), "k": (1, 2),
        "pos": ((0.32, 0.72), (0.30, 0.72), False),   # 特写里扣件居中略偏
    },
    "土建": {
        "names": ["裂缝", "渗漏水", "管片掉块"],
        "classes": {
            0: ("crack",   {6}, (0.05, 0.20), (0.10, 0.42)),
            1: ("leakage", {7}, (0.12, 0.30), (0.12, 0.28)),
            2: ("spalling",{5}, (0.10, 0.30), (0.10, 0.28)),
        },
        "zoom": (0.70, 1.00), "k": (4, 8),
        "pos": ((0.08, 0.92), (0.08, 0.85), False),   # 整墙填满, 散布
    },
    "设备": {
        "names": ["异物入侵", "管线支架松脱"],
        "classes": {
            0: ("equip_intrusion",     {3}, (0.12, 0.32), (0.12, 0.30)),
            1: ("equip_bracket_loose",{4}, (0.06, 0.22), (0.08, 0.28)),
        },
        "zoom": (0.55, 0.85), "k": (1, 2),
        "pos": ((0.25, 0.80), (0.30, 0.80), False),   # 近景异物居中
    },
}
GLOBAL_TO_LOCAL = {  # 全局id -> (branch, local)
    0: ("轨道", 2), 1: ("轨道", 0), 2: ("轨道", 1),
    3: ("设备", 0), 4: ("设备", 1),
    5: ("土建", 2), 6: ("土建", 0), 7: ("土建", 1),
}


def load_skip_stems():
    if os.path.isfile(COMBINED_SKIP_23):
        return {ln.strip() for ln in open(COMBINED_SKIP_23, encoding="utf-8") if ln.strip()}
    return set()


def list_backgrounds(branch):
    fs = []
    for e in ("*.png", "*.jpg", "*.jpeg", "*.JPG"):
        fs += glob.glob(os.path.join(NZ, branch, e))
    return sorted(fs)


def _feather_alpha(bgr, p=4.0, feather=0.10):
    """给 3 通道 BGR 补一张羽化 alpha（圆角矩 p=4 超椭圆, 中心1->边缘0），返回 BGRA。
    prepare_textures 的增强变体只回写 BGR(丢 alpha) -> 硬方边；这里现场补羽化，去贴片感。"""
    h, w = bgr.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    dx = (xs - (w - 1) / 2.0) / (w / 2.0)
    dy = (ys - (h - 1) / 2.0) / (h / 2.0)
    d = (np.abs(dx) ** p + np.abs(dy) ** p) ** (1.0 / p)
    a = np.clip((1 - d) / feather, 0, 1).astype(np.float32)
    return np.dstack([bgr.astype(np.uint8), (a * 255).astype(np.uint8)])


def load_crops(branch, textures_dir):
    """{local: [BGRA numpy,...]}。只取本分支会用的类；3 通道补羽化。"""
    out = {}
    for local, (cls, *_rest) in BRANCHES[branch]["classes"].items():
        imgs = []
        for p in sorted(glob.glob(os.path.join(textures_dir, f"{cls}_*.png"))):
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            if im.ndim == 3 and im.shape[2] == 4:
                imgs.append(im)
            elif im.ndim == 3 and im.shape[2] == 3:
                imgs.append(_feather_alpha(im))
        out[local] = imgs
    return out


def bg_aug(bgr, rng, branch):
    """背景取景 + 增强：特写随机裁剪(按分支 zoom)、水平翻转、亮度/对比度抖动。
    特写裁剪在全分辨率上进行(先裁再 resize_max, 保住清晰度) -> 模拟 "相机贴近工件" 的取景。
    返回 (img_bgr, flip)。"""
    h, w = bgr.shape[:2]
    flip = rng.random() < 0.5
    if flip:
        bgr = cv2.flip(bgr, 1)
    # 特写随机中心裁剪: 保留比例 = zoom(lo,hi)。越小越特写(等效放大)。
    lo, hi = BRANCHES[branch]["zoom"]
    crop = rng.uniform(lo, hi)
    nw, nh = int(w * crop), int(h * crop)
    x0 = rng.randint(0, w - nw); y0 = rng.randint(0, h - nh)
    bgr = bgr[y0:y0 + nh, x0:x0 + nw]
    # 亮度/对比度
    bgr = cv2.convertScaleAbs(bgr, alpha=rng.uniform(0.85, 1.15), beta=rng.uniform(-12, 12))
    return bgr, flip


def resize_max(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    s = max_dim / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def paste_defect(bg, fg_rgba, cx, cy, dw_px, dh_px, rng):
    """把羽化 RGBA 缺陷贴到 bg(BGR)，中心在 (cx,cy) 像素，目标盒 (dw_px,dh_px)。
    小角度旋转 + alpha 混合；越界裁剪。返回 bg(原地修改)。"""
    fh, fw = fg_rgba.shape[:2]
    if rng.random() < 0.6:
        ang = rng.uniform(-6, 6)
        M = cv2.getRotationMatrix2D((fw / 2, fh / 2), ang, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(fh * sin + fw * cos), int(fh * cos + fw * sin)
        M[0, 2] += (nw - fw) / 2; M[1, 2] += (nh - fh) / 2
        fg_rgba = cv2.warpAffine(fg_rgba, M, (nw, nh), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
    fg = cv2.resize(fg_rgba, (max(1, int(dw_px)), max(1, int(dh_px))),
                    interpolation=cv2.INTER_AREA)
    fh, fw = fg.shape[:2]
    x0 = int(round(cx - fw / 2)); y0 = int(round(cy - fh / 2))
    x1 = x0 + fw; y1 = y0 + fh
    H, W = bg.shape[:2]
    # 求交集
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    # 前景对应子区
    fx0, fy0 = sx0 - x0, sy0 - y0
    fx1, fy1 = sx1 - x0, sy1 - y0
    fg_o = fg[fy0:fy1, fx0:fx1]
    if fg_o.shape[0] == 0 or fg_o.shape[1] == 0:
        return
    a = fg_o[..., 3:4].astype(np.float32) / 255.0
    rgb = fg_o[..., :3].astype(np.float32)
    bg_o = bg[sy0:sy1, sx0:sx1].astype(np.float32)
    bg[sy0:sy1, sx0:sx1] = (rgb * a + bg_o * (1 - a)).astype(np.uint8)


def pick_pos(branch, rng):
    """返回归一化 (cx, cy)。侧带类 x 落到左右带。"""
    cfg = BRANCHES[branch]["pos"]
    wx, wy = cfg[0], cfg[1]
    side = cfg[2] if len(cfg) > 2 else False
    if side:
        left = rng.random() < 0.5
        x = rng.uniform(*wx) if left else rng.uniform(1 - wx[1], 1 - wx[0])
    else:
        x = rng.uniform(*wx)
    y = rng.uniform(*wy)
    return x, y


def gen_one(bg, branch, crops, rng):
    """在一张背景上粘贴 k 个缺陷(按分支 k 范围)，返回 (box列表[(local,x,y,w,h)], bg_bgr)。"""
    boxes = []
    k_lo, k_hi = BRANCHES[branch]["k"]
    k = rng.randint(k_lo, k_hi)
    local_ids = list(BRANCHES[branch]["classes"].keys())
    H, W = bg.shape[:2]
    for _ in range(k):
        local = rng.choice(local_ids)
        cls, _gids, wr, hr = BRANCHES[branch]["classes"][local]
        pool = crops[local]
        if not pool:
            continue
        fg = rng.choice(pool)
        dw = rng.uniform(*wr) * W
        dh = rng.uniform(*hr) * H
        cx, cy = pick_pos(branch, rng)
        # 让盒尽量不越界太多：中心收缩
        cx = np.clip(cx, dw / (2 * W), 1 - dw / (2 * W))
        cy = np.clip(cy, dh / (2 * H), 1 - dh / (2 * H))
        paste_defect(bg, fg, cx * W, cy * H, dw, dh, rng)
        boxes.append((local, cx, cy, dw / W, dh / H))
    return boxes, bg


def write_label(path, boxes):
    with open(path, "w", encoding="utf-8") as f:
        for local, cx, cy, w, h in boxes:
            f.write(f"{local} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def build(branch, textures_dir, out, n_train, n_val, seed=SEED, preview=0):
    rng = random.Random(seed)
    skip = load_skip_stems()
    bgs = list_backgrounds(branch)
    if not bgs:
        sys.exit(f"{branch} 无正常背景图")
    crops = load_crops(branch, textures_dir)
    print(f"[{branch}] 背景 {len(bgs)} 张; 贴图 { {c: len(v) for c, v in crops.items()} }")

    def gen_images():
        """生成器：每个样本随机抽背景 -> 特写裁剪+增强 -> resize -> 贴缺陷。"""
        while True:
            bgp = rng.choice(bgs)
            bg = cv2.imread(bgp)
            if bg is None:
                continue
            bg, _flip = bg_aug(bg, rng, branch)   # 全分辨率特写裁剪+翻转+亮度
            bg = resize_max(bg)
            boxes, bg = gen_one(bg, branch, crops, rng)
            yield bg, boxes

    def dump(idxs, split):
        gen = gen_images()
        for gidx in idxs:
            bg, boxes = next(gen)
            stem = f"cp_{gidx:06d}"
            ext = ".jpg"
            cv2.imwrite(os.path.join(out, "images", split, stem + ext), bg)
            write_label(os.path.join(out, "labels", split, stem + ".txt"), boxes)

    if preview:
        # 只出 preview 张到 out/images，不切 split（看质量/贴片感）
        os.makedirs(os.path.join(out, "images"), exist_ok=True)
        gen = gen_images()
        for gidx in range(preview):
            bg, boxes = next(gen)
            stem = f"preview_{gidx:02d}"
            cv2.imwrite(os.path.join(out, "images", stem + ".jpg"), bg)
            write_label(os.path.join(out, "images", stem + ".txt"), boxes)
            print(f"  {stem}.jpg boxes={[ (b[0],round(float(b[1]),3),round(float(b[2]),3),round(float(b[3]),3),round(float(b[4]),3)) for b in boxes ]}")
        print(f"[{branch}] 预览 {preview} 张 -> {out}/images")
        return

    for sp in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, sub, sp), exist_ok=True)
    dump(range(n_train), "train")
    dump(range(n_train, n_train + n_val), "val")
    print(f"[{branch}] 完成: train {n_train} / val {n_val} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", required=True, choices=list(BRANCHES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000, help="train 图数")
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--textures", default=None, help="缺陷贴图目录(默认自动用 prepare_from_bing 生成)")
    ap.add_argument("--preview", type=int, default=0, help=">0 时只出 N 张预览到 out/images(不切 split)")
    ap.add_argument("--regen-textures", action="store_true",
                    help="重新用病集+skip-stems 裁贴图(否则用现成 textures)")
    a = ap.parse_args()

    if a.textures is None:
        a.textures = os.path.join(ROOT, "data_gen", "textures_cp")
    if a.regen_textures:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "prepare_textures", os.path.join(ROOT, "data_gen", "prepare_textures.py"))
        pt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pt)
        skip = load_skip_stems()
        os.makedirs(a.textures, exist_ok=True)
        counts = pt.prepare_from_bing(a.textures, size=512, skip_stems=skip)
        print(f"  裁缺陷贴图 -> {a.textures}: {counts}")

    n_val = max(1, round(a.n * a.val_frac))
    build(a.branch, a.textures, a.out, a.n, n_val, a.seed, preview=a.preview)


if __name__ == "__main__":
    main()
