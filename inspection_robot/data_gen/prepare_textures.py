# -*- coding: utf-8 -*-
"""
从病害图选近景 -> 512x512 贴图, 供世界文件做 albedo_map
================================================================
支持两种数据源 (--source):
  data   默认: 从 pro/2/data 各类目录挑图(原有 7 类)。
  crack1280  用户隧道裂缝数据集(单类"裂缝", YOLO 分割多边形, 512x512 近景),
         按多边形外接框裁裂缝区域 -> crack_*.png。空标签(无裂缝)跳过。
  data1  pro/2/data1 真实巡检照片, 6 类 (裂缝/渗漏水/设备异物/扣件断裂/缺失/松动)。
         - 目录多为嵌套 (images/{train,val}), 用递归 glob
         - 设备目录名无下划线 (设备异物入侵) 或单类 foreign_object (设备_病害_yolo)
         - 扣件三类: 轨道固件图片按 轨道固件_4cls_stage 标签(id 1断裂/2丢失/3松动, 0正常剔除) 筛

每类挑 N 张(默认5), cover-crop 成 512x512.
用法:
    python3 data_gen/prepare_textures.py [每类张数] [输出目录]            # data 源
    python3 data_gen/prepare_textures.py --source data1                  # data1 源 -> textures_data1
    python3 data_gen/prepare_textures.py --source data1 3 --out /tmp/tex # 自定义
"""
import argparse
import glob
import os
import random
import shutil
import sys

import cv2
import numpy as np

PRO_DATA = os.path.expanduser("~/workspace/pro/2/data")
DATA1 = os.path.expanduser("~/workspace/pro/2/data1")

# ---------------- 病集源(病害训练集+正常状态, --source bing) ----------------
# 全局 8 类真实缺陷照片; 每图单一类(19 图同框多个同类框), 图幅 189~5712px。
# 这里按 YOLO 框"裁缺陷"出贴图(而非整图), 一框一纹理一缺陷, 标签才对得上。
BING = os.path.expanduser("~/workspace/pro/2 (1)/病害训练集+正常状态/病害训练集")
# 全局类(class.txt) -> data1 管线局部类名
GLOBAL_TO_LOCAL = {
    0: "rail_fastener_loose",    # 扣件松动歪斜(轨道)
    1: "rail_fastener_broken",   # 扣件断裂(轨道)
    2: "rail_fastener_missing",  # 扣件缺失(轨道)
    3: "equip_intrusion",        # 异物入侵(设备)
    4: "equip_bracket_loose",    # 管线支架松脱(设备)
    5: "spalling",               # 管片破损掉块(土建)
    6: "crack",                  # 裂缝(土建)
    7: "leakage",                # 渗漏水(土建)
}
MIN_BOX = 48   # 框过小时以框心外扩到的最小窗口(px), 避免贴图过糊/空白

# ---------------- 裂缝数据集源(--source crack1280) ----------------
# 用户提供的隧道裂缝数据集(单类"裂缝"): train/valid 各 images+labels, 512x512 近景,
# YOLO 分割(多边形)标签, 裂缝只占图一小块(外接框面积中位 ~6%)。空标签 = 无裂缝背景。
# 裁法: 按多边形外接框外扩 CRACK_MARGIN 倍(留墙面上下文)再 cover_crop, 一框一纹理。
CRACK1280 = os.path.expanduser("~/datasets/tunnel_crack_20251119")
CRACK_MARGIN = 1.6            # 外接框外扩倍数
CRACK_MAX_DEFAULT = 200       # 不显式给张数时, crack1280 源生成的贴图上限

# ---------------- 墙色羽化(根治"贴纸感") ----------------
# 病害薄盒以 PBR albedo_map 贴在墙上。若裁出来是一张"带照片墙背景的硬矩形", 盒边
# 与仿真墙之间会形成直角亮度差 -> 一眼"贴纸"。这里把贴图边缘 RGB 混成【墙的 albedo】:
#   墙 albedo = 世界文件里 wall 的 <diffuse>0.55 0.55 0.58</diffuse>
#   (BGR≈148,140,140, 微偏蓝)。薄盒与墙同光同反射, 边上一旦等于墙 albedo, 渲染后同色
#   -> 盒边"隐形", 矩形轮廓消失, 只留中间真实缺陷渐变入墙。
# 机制上不依赖 gz 是否读贴图 alpha: 即便完全忽略 alpha, 边也已是墙色; 若读取则真透明。
WALL_BGR = (148, 140, 140)              # 0.55,0.55,0.58 (BGR)
WALL_CLASSES = {"crack", "leakage", "spalling", "equip_bracket_loose"}


def _feather_alpha(rgb_bgr, wall_bgr=None, feather=0.10, p=4.0):
    """给 512 缺陷贴图加羽化: 圆角矩形(p=4超椭圆) alpha 中心1->边缘0, 并(墙面类)混边缘RGB。
    wall_bgr 非空时: final = rgb*mask + wall*(1-mask) —— 边缘隐形; 纯 alpha 羽化兜底。
    p=2 是椭圆(会把缺陷打成圆斑, 破坏形态); p>=4 几乎矩形、仅圆角+软边, 保留缺陷真实形状。
    返回 BGRA(4通道 PNG, alpha 通道 = 羽化掩膜)。"""
    h, w = rgb_bgr.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    dx = (xs - (w - 1) / 2.0) / (w / 2.0)
    dy = (ys - (h - 1) / 2.0) / (h / 2.0)
    d = (np.abs(dx) ** p + np.abs(dy) ** p) ** (1.0 / p)   # p=4 -> 圆角矩形, 角≈1.19
    a = np.clip((1 - d) / feather, 0, 1).astype(np.float32)   # 1(中心)->0(边/角)
    a3 = a[..., None]
    out = rgb_bgr
    if wall_bgr is not None:
        out = rgb_bgr * a3 + np.array(wall_bgr, dtype=np.float32) * (1 - a3)
    rgba = np.dstack([out.astype(np.uint8), (a * 255).astype(np.uint8)])
    return rgba

# ---------------- 稀有类贴图增强 ----------------
# 病集某些缺陷框极少(如 equip_bracket_loose 仅 4 框、rail_fastener_missing 仅 5 框),
# 导致整个数据集里这几类外观种类太少、网络学不会。裁剪完成后对"低于目标张数"的类
# 用已有贴图做增强变体(翻转/小旋转/缩放/调色)补足, 让每类有足够多样的纹理库。
AUG_TARGET = 25          # 每类贴图目标张数(不足则增强补足)
AUG_NO_BOOST = {"crack", "equip_intrusion"}   # 已较充足/表现最优, 维持原数不上调
AUG_SEED = 42            # 确定性: 每次生成同一批变体, 不随运行漂移


def _aug_variants(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """对单张 512x512 贴图做一个随机增强变体(翻转/小旋转/缩放/调色), 返回等价 BGR。"""
    h, w = img.shape[:2]
    if rng.random() < 0.5:                              # 左右翻转
        img = cv2.flip(img, 1)
    ang = rng.uniform(-8, 8)                            # 小角度旋转 (BORDER_REFLECT 防黑边)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    z = rng.uniform(0.92, 1.08)                         # 轻微缩放(z<1 中心裁, z>1 放大裁)
    nw, nh = int(w * z), int(h * z)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = max(0, (nw - w) // 2); y0 = max(0, (nh - h) // 2)
    c = r[y0:y0 + h, x0:x0 + w]
    if c.shape[:2] != (h, w):
        c = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
    img = c
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.90, 1.10), beta=rng.uniform(-15, 15))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ch in (1, 2):                                   # 饱和度/明度抖动
        hsv[..., ch] = np.clip(hsv[..., ch].astype(np.int16) * rng.uniform(0.9, 1.1),
                               0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def augment_textures(out_dir: str, counts: dict) -> dict:
    """把 counts 中低于 AUG_TARGET 且不在 AUG_NO_BOOST 的类补足到目标张数。
    从已有贴图随机增强生成新变体(确定性), 结果整到 {类}_{k}.png 连续编号。
    返回 {类: 新增张数}。"""
    added = {}
    for local, base in sorted(counts.items()):
        if local in AUG_NO_BOOST or base >= AUG_TARGET:
            continue
        rng = random.Random(AUG_SEED * 1000 + sum(ord(c) for c in local))
        srcs = [cv2.imread(os.path.join(out_dir, f"{local}_{i}.png")) for i in range(1, base + 1)]
        srcs = [s for s in srcs if s is not None]
        if not srcs:
            continue
        for k in range(base + 1, AUG_TARGET + 1):
            v = _aug_variants(srcs[rng.randrange(len(srcs))], rng)
            cv2.imwrite(os.path.join(out_dir, f"{local}_{k}.png"), v)
        added[local] = AUG_TARGET - base
    return added

# ---------------- data 源类别(原 7 类) ----------------
CLASSES = {
    "crack":                {"dir": "土建_裂缝",           "ext_pref": [".png", ".jpg", ".jpeg"]},
    "leakage":              {"dir": "土建_渗漏水",         "ext_pref": [".jpg", ".jpeg", ".png"]},
    "spalling":             {"dir": "土建_管片破损掉块",   "ext_pref": [".jpg", ".jpeg", ".png"]},
    "rail_fastener_broken": {"dir": "轨道_扣件断裂",       "ext_pref": [".jpg", ".jpeg", ".png"]},
    "rail_fastener_missing": {"dir": "轨道_扣件缺失",      "ext_pref": [".jpg", ".jpeg", ".png"]},
    "rail_fastener_loose":   {"dir": "轨道_扣件松动歪斜",  "ext_pref": [".jpg", ".jpeg", ".png"]},
    "equip_intrusion":       {"dir": "设备_异物入侵",      "ext_pref": [".jpg", ".jpeg", ".png"]},
}
TARGET = 512

# ---------------- data1 源类别(6 类) ----------------
# 目录可 glob(relative), recursive 时对图片做递归匹配; 多目录取并集(去重)。
DATA1_CLASSES = {
    "crack":                {"globs": ["土建_裂缝/images/**/*.jpg", "土建_裂缝/images/**/*.png"]},
    "leakage":              {"globs": ["土建_渗漏水/*.jpg", "土建_渗漏水/*.png", "土建_渗漏水/*.jpeg"]},
    "equip_intrusion":      {"globs": ["设备_病害_yolo/images/**/*.jpg",
                                       "设备_病害_yolo/images/**/*.png",
                                       "设备异物入侵/images/**/*.jpg"]},
    # 扣件三类: 图来自 轨道固件, 按 4cls 标签 id 筛
    "rail_fastener_broken": {"rail": "broken"},
    "rail_fastener_missing": {"rail": "missing"},
    "rail_fastener_loose":   {"rail": "loose"},
}
# 扣件图/标签目录 (data1)
RAIL_IMG_GLOBS = ["轨道固件/train&val/YOLO/images/**/*.png", "轨道固件/test/images/**/*.png"]
RAIL_LBL_GLOBS = ["轨道固件_4cls_stage/labels/%s/**/*.txt" % split
                  for split in ("train", "val", "test")]
RAIL_ID_TO_CLS = {1: "rail_fastener_broken",
                  2: "rail_fastener_missing",
                  3: "rail_fastener_loose"}     # 正常(0) 剔除


def cover_crop(img: np.ndarray, size: int = TARGET) -> np.ndarray:
    """等比放大后中心裁剪到 size x size (cover-crop: 短边撑满, 长边裁掉溢出)。

    之前用 scale = size/max(w,h): 对宽图(w>h)会得到 nh<size 且 y0<0,
    负索引切片只剩几行 -> 贴图退化成 4/64 行高的白条(rail_fastener 类实测 4px)。
    改为 scale = size/min(w,h) 保证短边=size、长边>=size, 中心裁剪索引恒非负。"""
    h, w = img.shape[:2]
    scale = size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = max(0, (nw - size) // 2)
    y0 = max(0, (nh - size) // 2)
    return resized[y0:y0 + size, x0:x0 + size]


def _existing(paths):
    """去除重复且只保留真实存在的文件。"""
    seen, ok = set(), []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            ok.append(p)
    return ok


def _sorted_dedup(paths):
    return _existing(sorted(set(paths)))


def _rail_image_by_class() -> dict:
    """扫 轨道固件 图片 + 4cls 标签 -> {class_name: [图片路径]} (正常"剔除)。"""
    by = {}
    imgs = set()
    for g in RAIL_IMG_GLOBS:
        imgs |= set(glob.glob(os.path.join(DATA1, g), recursive=True))
    # 图片名(去扩展名) -> 路径
    stem2img = {}
    for p in imgs:
        stem2img.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    for g in RAIL_LBL_GLOBS:
        for txt in glob.glob(os.path.join(DATA1, g), recursive=True):
            stem = os.path.splitext(os.path.basename(txt))[0]
            cls = None
            try:
                for line in open(txt, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    c = int(line.split()[0])
                    if c in RAIL_ID_TO_CLS:
                        cls = RAIL_ID_TO_CLS[c]
                        break
            except (OSError, ValueError):
                continue
            if cls is None or stem not in stem2img:   # 正常(无病害)或图未匹配 -> 剔除
                continue
            by.setdefault(cls, []).append(stem2img[stem])
    # 每类去重
    return {k: _sorted_dedup(v) for k, v in by.items()}


def _find_img(images_dir: str, stem: str):
    for ext in (".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def _crop_box_window(img: np.ndarray, box, size: int = TARGET, min_box: int = MIN_BOX,
                     wall_bgr=None) -> np.ndarray:
    """按 YOLO 归一化框 [cx,cy,w,h] 裁出缺陷区域, 展到 size x size 后做墙色羽化。
    框过小/贴边时以框心扩到 min_box 窗口再 clamp, 保证非空白补丁。
    返回 BGRA(4 通道), 边羽化(墙面类混向墙色)。"""
    H, W = img.shape[:2]
    cx, cy, w, h = box
    w2, h2 = w / 2.0, h / 2.0
    x1, x2 = (cx - w2) * W, (cx + w2) * W
    y1, y2 = (cy - h2) * H, (cy + h2) * H
    if x2 - x1 < min_box:                    # 过小 -> 以框心外扩
        mx = (x1 + x2) / 2.0
        x1, x2 = mx - min_box / 2.0, mx + min_box / 2.0
    if y2 - y1 < min_box:
        my = (y1 + y2) / 2.0
        y1, y2 = my - min_box / 2.0, my + min_box / 2.0
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(W, x2)), int(min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return _feather_alpha(crop, wall_bgr=wall_bgr)


def prepare_from_bing(out_dir: str, size: int = TARGET, max_per: int = 0,
                      skip_stems: set = None) -> dict:
    """病集标签按框裁缺陷 -> {out_dir}/{local}_{k}.png, k=该局部类序号(从 1)。
    全局类经 GLOBAL_TO_LOCAL 映射; 每张病集图(的同框同类框)各成一纹理。
    返回 {local_class: n}。max_per>0 时每类最多 n 张(调试)。
    skip_stems: 被留作测试的图 stem 集合, 这些图不裁贴图(杜绝训练贴图与测试图重叠)。"""
    skip_stems = skip_stems or set()
    labels = sorted(glob.glob(os.path.join(BING, "labels", "*.txt")))
    shutil.rmtree(out_dir, ignore_errors=True)   # 重生成前清空(旧 6 类贴图不再残留)
    os.makedirs(out_dir, exist_ok=True)
    n_of: dict[str, int] = {}
    for lp in labels:
        stem = os.path.splitext(os.path.basename(lp))[0]
        if stem in skip_stems:                    # 留测图: 不裁贴图(防泄漏)
            continue
        imgp = _find_img(os.path.join(BING, "images"), stem)
        if imgp is None:
            continue
        img = cv2.imread(imgp)
        if img is None:
            continue
        for line in open(lp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            local = GLOBAL_TO_LOCAL.get(int(parts[0]))
            if local is None:
                continue
            if max_per and n_of.get(local, 0) >= max_per:
                continue
            wall_bgr = WALL_BGR if local in WALL_CLASSES else None
            tex = _crop_box_window(img, [float(x) for x in parts[1:5]], size, wall_bgr=wall_bgr)
            if tex is None:
                continue
            n_of[local] = n_of.get(local, 0) + 1
            cv2.imwrite(os.path.join(out_dir, f"{local}_{n_of[local]}.png"), tex)
    # 稀有类增强: 不足 AUG_TARGET 的类补足变体(crack/intrusion 维持原数)
    added = augment_textures(out_dir, n_of)
    for local, n_added in sorted(added.items()):
        n_of[local] = n_of.get(local, 0) + n_added
        print(f"  [增强] {local}: +{n_added} 张 (-> {n_of[local]})")
    return n_of


def _poly_to_bbox(parts):
    """YOLO seg 行 [cls, x1,y1,x2,y2,...] -> 归一化外接框 (cx,cy,w,h)。"""
    xy = [float(v) for v in parts[1:]]
    xs, ys = xy[0::2], xy[1::2]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def _crop_bbox_padded(img: np.ndarray, bbox, margin: float = CRACK_MARGIN,
                      min_side: int = MIN_BOX) -> np.ndarray:
    """按归一化外接框外扩 margin 倍裁矩形(夹到图内); 过小则以框心扩到 min_side。
    与 _crop_box_window 不同: 这里**不 reshape**, 交 cover_crop 保形缩放(裂缝是斜细线,
    强行拉成正方形会把裂缝撑肥)。"""
    H, W = img.shape[:2]
    cx, cy, w, h = bbox
    w, h = w * margin, h * margin
    x1, x2 = (cx - w / 2.0) * W, (cx + w / 2.0) * W
    y1, y2 = (cy - h / 2.0) * H, (cy + h / 2.0) * H
    if x2 - x1 < min_side:
        mx = (x1 + x2) / 2.0
        x1, x2 = mx - min_side / 2.0, mx + min_side / 2.0
    if y2 - y1 < min_side:
        my = (y1 + y2) / 2.0
        y1, y2 = my - min_side / 2.0, my + min_side / 2.0
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(W, x2)), int(min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def prepare_from_crack1280(out_dir: str, size: int = TARGET, max_per: int = 0,
                           margin: float = CRACK_MARGIN) -> dict:
    """隧道裂缝数据集(单类)多边形标签 -> {out_dir}/crack_{k}.png (BGRA 羽化)。
    遍历 train+valid labels(空标签跳过), 每行一个多边形裁一张贴图。
    返回 {crack: n}。max_per>0 时上限 n 张。"""
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for split in ("train", "valid"):
        lbls = sorted(glob.glob(os.path.join(CRACK1280, split, "labels", "*.txt")))
        for lp in lbls:
            if max_per and n >= max_per:
                break
            stem = os.path.splitext(os.path.basename(lp))[0]
            imgp = _find_img(os.path.join(CRACK1280, split, "images"), stem)
            if imgp is None:
                continue
            img = cv2.imread(imgp)
            if img is None:
                continue
            for line in open(lp, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                if max_per and n >= max_per:
                    break
                crop = _crop_bbox_padded(img, _poly_to_bbox(line.split()), margin)
                if crop is None or crop.size == 0:
                    continue
                crop = cover_crop(crop, size)
                tex = _feather_alpha(crop, wall_bgr=WALL_BGR)   # 裂缝=墙面类, 边混墙色
                n += 1
                cv2.imwrite(os.path.join(out_dir, f"crack_{n}.png"), tex)
    return {"crack": n}


def pick_images(cls_name: str, n: int, source: str = "data") -> list[str]:
    """确定性挑选 n 张图片。source=data 走 CLASSES(单层 glob); data1 走递归/扣件分类。"""
    if source == "data":
        cfg = CLASSES[cls_name]
        base = os.path.join(PRO_DATA, cfg["dir"])
        cands = []
        for ext in cfg["ext_pref"]:
            cands += sorted(glob.glob(os.path.join(base, "*" + ext)))
        if not cands:
            raise FileNotFoundError(f"类 {cls_name} 在 {base} 下未找到图片")
        return _existing(cands)[:n]

    if source == "data1":
        cfg = DATA1_CLASSES[cls_name]
        if "rail" in cfg:                       # 扣件: 从筛好的分类列表取
            cands = _rail_image_by_class().get(cls_name, [])
            if not cands:
                raise FileNotFoundError(f"扣件类 {cls_name} 未筛到图片(检查 4cls 标签/id 映射)")
            return cands[:n]
        cands = []
        for g in cfg["globs"]:
            cands += sorted(glob.glob(os.path.join(DATA1, g), recursive=True))
        cands = _sorted_dedup(cands)
        if not cands:
            raise FileNotFoundError(f"类 {cls_name} 在 data1 下未找到图片")
        return cands[:n]

    raise ValueError(f"未知 source: {source}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("data", "data1", "bing", "crack1280"), default="data")
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-stems", default=None,
                    help="留测图 stem 列表文件(每行一个); 这些图不裁贴图, 防训练贴图与测试图重叠")
    ap.add_argument("n", nargs="?", type=int, default=5, help="每类张数")
    ap.add_argument("out_dir", nargs="?", default="", help="输出目录(兼容位置参数)")
    args = ap.parse_args()

    src = args.source
    n = args.n
    _default_dirs = {"data1": "textures_data1", "bing": "textures_data1",
                     "crack1280": "textures_crack"}
    out_dir = args.out or args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), _default_dirs.get(src, "textures"))
    # 兼容单位置参数用法: python3 ... 5  (第1个位置参数 -> n)
    if n != 5 and not out_dir and len(sys.argv) > 2 and not sys.argv[1].startswith("-"):
        # 若用户只传一个数字, argparse 已把它当 n; 这里无额外逻辑
        pass
    os.makedirs(out_dir, exist_ok=True)

    if src == "bing":
        skip = set()
        if args.skip_stems and os.path.isfile(args.skip_stems):
            skip = {ln.strip() for ln in open(args.skip_stems, encoding="utf-8") if ln.strip()}
            print(f"  ↻ 跳过 {len(skip)} 个留测图 stem(不裁贴图)")
        print(f"源: bing(病集, 全局 8 类 -> 局部类, 按框裁缺陷), 输出 {out_dir}")
        counts = prepare_from_bing(out_dir, size=TARGET, skip_stems=skip)
        if not counts:
            print("  ⚠ 未从病集裁出任何贴图(检查 病集 路径/标签)")
            sys.exit(1)
        for k in sorted(counts):
            print(f"  {k:24s} {counts[k]:>3} 张")
        missing = set(GLOBAL_TO_LOCAL.values()) - set(counts)
        if missing:
            print(f"  ⚠ 以下类无贴图: {sorted(missing)}")
        print(f"✅ 病集贴图已生成到 {out_dir} ({TARGET}x{TARGET})")
        return

    if src == "crack1280":
        max_per = n if n != 5 else CRACK_MAX_DEFAULT
        print(f"源: crack1280(隧道裂缝数据集, 单类, 多边形->外接框裁贴图), "
              f"输出 {out_dir}, 上限 {max_per} 张")
        counts = prepare_from_crack1280(out_dir, size=TARGET, max_per=max_per)
        got = counts.get("crack", 0)
        if not got:
            print("  ⚠ 未裁出任何裂缝贴图(检查数据集路径/标签)")
            sys.exit(1)
        print(f"  crack  {got:>4} 张")
        print(f"✅ 裂缝贴图已生成到 {out_dir} ({TARGET}x{TARGET})")
        return

    classes = DATA1_CLASSES if src == "data1" else CLASSES
    rail_note = " (扣件=轨道固件图按4cls标签id筛)" if src == "data1" else ""
    print(f"源: {src}, 类别 {len(classes)} 类{rail_note}, 输出 {out_dir}")

    for cls_name in classes:
        try:
            picks = pick_images(cls_name, n, src)
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")
            continue
        for i, srcp in enumerate(picks, 1):
            img = cv2.imread(srcp)
            if img is None:
                print(f"  ⚠ 读不到 {srcp}, 跳过")
                continue
            img = cover_crop(img)
            dst = os.path.join(out_dir, f"{cls_name}_{i}.png")
            cv2.imwrite(dst, img)
            print(f"  {os.path.basename(srcp):60s} -> {os.path.basename(dst)}")
        if len(picks) < n:
            print(f"  ⚠ {cls_name}: 只找到 {len(picks)}/{n} 张")
    print(f"✅ 贴图已生成到 {out_dir} (每类 {n} 张, {TARGET}x{TARGET})")


if __name__ == "__main__":
    main()
