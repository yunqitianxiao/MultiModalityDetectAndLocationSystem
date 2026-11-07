#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备管线支架松脱数据集 —— 真实贴片合成生成器
================================================
原理: 用从真实照片抠出的支架贴片(RGBA) + 真实隧道墙面背景, 合成"支架正常 / 支架松脱"两类
     检测图。因为素材 = 真实像素, 所以比扩散模型/3D渲染 更接近实拍; 且合成可批量产出。

适合的缺陷形态(松脱 = 结构/几何级细微异常):
  旋转歪斜(rot) / 下坠悬空(drop) / 剪切弯折(tilt) / 中段脱开(gap) / 掉螺栓
  → 用几何扰动把"正常贴片"改造成"松脱贴片", 后台自动给出贴合缺陷的 YOLO 框。

不依赖 torch/cuda —— 纯 opencv + numpy + PIL, CPU 即可大规模合成。

用法:
  1) 先抠贴片:   python3 抠支架精灵.py --src 你的支架图文件夹 --box 粗框txt
                 (或直接把透明背景 RGBA 支架 PNG 丢进 --sprites 目录)
  2) 预览真实感: python3 生成_支架松脱数据集.py --sprites <贴片目录> --backgrounds <背景图目录> --preview
                 -> 生成 preview_网格.png, 看阴影/颜色/边缘/框是否到位, 不满调参数再批量。
  3) 批量生成:   python3 生成_支架松脱数据集.py --sprites <贴片目录> --backgrounds <背景图目录> --count 2000 --output <数据集根目录>

类别: 0=支架正常  1=支架松脱  (在 --names 可改)

渲染: dot 的同类中文注释风格。输出为标准 Ultralytics YOLO 检测数据集:
  <output>/images/{train,val,test}/*.jpg
  <output>/labels/{train,val,test}/*.txt  (cx cy w h 归一化)
  <output>/data.yaml
  <output>/_split_manifest.txt  (split\t图片名)
  <output>/_stats.txt           (各split 每类 实例/图 计数)
"""
import os, sys, random, argparse, glob, collections, math
import numpy as np
import cv2
from PIL import Image

# ----------------------------------------------------------------------------
# 类别 / 全局常量
# ----------------------------------------------------------------------------
DEFAULT_NAMES = ["支架正常", "支架松脱"]
K_NORMAL, K_ROT, K_DROP, K_TILT, K_GAP = "normal", "rot", "drop", "tilt", "gap"
LOOSE_KINDS = [K_ROT, K_DROP, K_TILT, K_GAP]

# ----------------------------------------------------------------------------
# 贴片 / 背景加载
# ----------------------------------------------------------------------------
def load_sprites(dirpath):
    """加载透明背景 RGBA 贴片(支架)。返回 list[np.uint8 hxwx4]。"""
    sprites = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for p in sorted(glob.glob(os.path.join(dirpath, ext))):
            im = Image.open(p).convert("RGBA")
            arr = np.array(im)
            # 用 alpha 兜底: 若 alpha 全是255(不透明), 用真实背景抠不了, 提示走抠图工具
            if (arr[:, :, 3] < 250).sum() == 0:
                print(f"  [提示] {os.path.basename(p)} 无透明通道(全不透明), 请用 抠支架精灵.py 先抠出透明贴片")
                continue
            sprites.append(arr)
    return sprites

def load_backgrounds(dirpath, size=None):
    """加载隧道墙面背景图, 统一转 RGB。返回 list[np.uint8 hxwx3]。"""
    bgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        for p in sorted(glob.glob(os.path.join(dirpath, ext))):
            im = Image.open(p).convert("RGB")
            bgs.append(np.array(im))
    return bgs

# ----------------------------------------------------------------------------
# 背景增强(避免每张一样)
# ----------------------------------------------------------------------------
def gamma_lut(img, g):
    inv = 1.0 / max(g, 1e-3)
    table = (np.power(np.arange(256) / 255.0, inv) * 255).astype(np.uint8)
    return cv2.LUT(img, table)

def vary_background(bg, target_w, target_h, rng):
    """随机裁剪/缩放/曝光/伽马/噪声, 让同背景也能出很多变体。"""
    h, w = bg.shape[:2]
    # 随机取一个子窗口再缩放到目标大小 (模拟不同机位/距离)
    scale = rng.uniform(0.70, 1.0)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    y0 = rng.randint(0, max(1, h - nh)); x0 = rng.randint(0, max(1, w - nw))
    sub = bg[y0:y0 + nh, x0:x0 + nw]
    sub = cv2.resize(sub, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 曝光
    alpha = rng.uniform(0.82, 1.18); beta = rng.uniform(-14, 14)
    sub = cv2.convertScaleAbs(sub, alpha=alpha, beta=beta)
    # 伽马
    if rng.random() < 0.55:
        sub = gamma_lut(sub, rng.uniform(0.85, 1.22))
    # 传感器噪声
    if rng.random() < 0.8:
        noise = np.random.normal(0, rng.uniform(1.2, 4.0), sub.shape).astype(np.int16)
        sub = np.clip(sub.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # 暗角
    if rng.random() < 0.5:
        sub = vignette(sub, rng.uniform(0.05, 0.22))
    return sub

def vignette(img, strength):
    h, w = img.shape[:2]
    x = np.linspace(-1, 1, w)[None, :]; y = np.linspace(-1, 1, h)[:, None]
    d = np.sqrt(x * x + y * y)
    m = (1 - strength * np.clip(d - 0.5, 0, 0.5) / 0.5).astype(np.float32)[..., None]
    return np.clip(img.astype(np.float32) * m, 0, 255).astype(np.uint8)

# ----------------------------------------------------------------------------
# 贴片单次 warping (旋转/剪切/平移)
# ----------------------------------------------------------------------------
def warp_piece(s, angle, shear, dx, dy):
    h, w = s.shape[:2]
    if w < 2 or h < 2:
        return s
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 1] += shear
    M[0, 2] += dx; M[1, 2] += dy
    out = cv2.warpAffine(s, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    # alpha 轻羽化去锯齿
    a = out[:, :, 3]
    if a.max() > 0:
        a = cv2.GaussianBlur(a, (3, 3), 0)
        out[:, :, 3] = a
    return out

def make_pieces(s, kind, rng):
    """把一张贴片改造成一组 pieces(每个 piece 是 (dx,dy,rgba))。松脱=几何扰动。"""
    th, tw = s.shape[:2]
    if kind == K_NORMAL:
        ang = rng.gauss(0, 2.0); sh = rng.gauss(0, 0.012)
        dx = int(rng.uniform(-0.02, 0.02) * tw); dy = int(rng.uniform(0, 0.03 * th))
        return [(0, 0, warp_piece(s, ang, sh, dx, dy))]

    if kind == K_ROT:   # 旋转歪斜
        ang = rng.uniform(8, 30) * (1 if rng.random() < 0.5 else -1)
        sh = rng.uniform(-0.14, 0.14)
        dx = int(rng.uniform(-0.05, 0.05) * tw); dy = int(rng.uniform(0.05, 0.18) * th)
        return [(0, 0, warp_piece(s, ang, sh, dx, dy))]

    if kind == K_TILT:  # 剪切弯折(结构变形)
        ang = rng.gauss(0, 5); sh = rng.uniform(0.12, 0.26) * (1 if rng.random() < 0.5 else -1)
        dy = int(rng.uniform(0.02, 0.10) * th)
        return [(0, 0, warp_piece(s, ang, sh, 0, dy))]

    if kind == K_DROP:  # 下坠悬空(整体下沉 + 轻微角度)
        ang = rng.uniform(-10, 10); sh = rng.gauss(0, 0.05)
        dy = int(rng.uniform(0.10, 0.35) * th)
        return [(0, 0, warp_piece(s, ang, sh, 0, dy))]

    if kind == K_GAP:   # 中段脱开(上下分离, 背景从缝中露出)
        split_at = int(th * rng.uniform(0.42, 0.62)); gap = int(th * rng.uniform(0.08, 0.26))
        top = s[:split_at].copy(); bot = s[split_at:].copy()
        ang = rng.gauss(0, 3)
        t = warp_piece(top, ang, 0, 0, 0)
        b = warp_piece(bot, ang, rng.gauss(0, 0.05), 0, gap)
        return [(0, 0, t), (0, gap, b)]

    return [(0, 0, s)]

# ----------------------------------------------------------------------------
# 颜色匹配 / 阴影 / 粘贴
# ----------------------------------------------------------------------------
def color_match(p, bg, x, y):
    """让贴片亮度贴近背景, 避免明显"贴纸感"。"""
    a = p[:, :, 3].astype(np.float32) / 255.0
    if a.sum() < 10:
        return p
    sp = (p[:, :, :3].astype(np.float32) * a[..., None]).sum(axis=(0, 1)) / max(a.sum(), 1)
    H, W = bg.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + p.shape[1]), min(H, y + p.shape[0])
    if x1 <= x0 or y1 <= y0:
        return p
    rb = bg[y0:y1, x0:x1].astype(np.float32).mean(axis=(0, 1))
    lum_p = 0.299 * sp[0] + 0.587 * sp[1] + 0.114 * sp[2]
    lum_b = 0.299 * rb[0] + 0.587 * rb[1] + 0.114 * rb[2]
    if lum_b < 1 or lum_p < 1:
        return p
    f = np.clip(lum_b / lum_p, 0.72, 1.28) * 0.96
    out = p.copy()
    out[:, :, :3] = np.clip(p[:, :, :3].astype(np.float32) * f, 0, 255).astype(np.uint8)
    return out

def _darken_region(bg, y0, x0, h, w, strength_float):
    H, W = bg.shape[:2]
    if h <= 0 or w <= 0:
        return
    y1, x1 = min(H, y0 + h), min(W, x0 + w)
    if y1 <= y0 or x1 <= x0:
        return
    reg = bg[y0:y1, x0:x1].astype(np.float32)
    reg *= (1.0 - strength_float)
    bg[y0:y1, x0:x1] = np.clip(reg, 0, 255).astype(np.uint8)

def add_shadow(bg, p, px, py, strength):
    """在贴片下方投一个软接触阴影, 让支架"落"在墙面上。"""
    ph, pw = p.shape[:2]
    sa = p[:, :, 3].astype(np.float32) / 255.0
    if sa.sum() < 5:
        return
    sh = (sa * 255).astype(np.uint8)
    kx = max(3, int(0.04 * pw)) | 1; ky = max(3, int(0.06 * ph)) | 1
    sh = cv2.GaussianBlur(sh, (kx, ky), 0).astype(np.float32) / 255.0
    off = max(2, int(0.05 * ph))
    _darken_region(bg, py + off, px, ph, pw, float(np.clip(sh.max(), 0, 1)) * strength)

def paste_piece(bg, p, px, py):
    """alpha 合成, 处理边界裁剪。"""
    H, W = bg.shape[:2]
    ph, pw = p.shape[:2]
    sx = sy = 0; x0, y0 = px, py
    if x0 < 0: sx = -x0; x0 = 0
    if y0 < 0: sy = -y0; y0 = 0
    wsrc = min(pw - sx, W - x0); hsrc = min(ph - sy, H - y0)
    if wsrc <= 0 or hsrc <= 0:
        return
    sp = p[sy:sy + hsrc, sx:sx + wsrc]
    sa = sp[:, :, 3].astype(np.float32) / 255.0
    reg_b = bg[y0:y0 + hsrc, x0:x0 + wsrc].astype(np.float32)
    spc = sp[:, :, :3].astype(np.float32)
    out = reg_b * (1 - sa[..., None]) + spc * sa[..., None]
    bg[y0:y0 + hsrc, x0:x0 + wsrc] = np.clip(out, 0, 255).astype(np.uint8)

def pieces_bbox(pieces, x, y):
    x0 = y0 = 10 ** 9; x1 = y1 = -1
    for dx, dy, p in pieces:
        a = p[:, :, 3]
        ys, xs = np.where(a > 16)
        if len(xs) == 0:
            continue
        px0, px1 = x + dx + xs.min(), x + dx + xs.max()
        py0, py1 = y + dy + ys.min(), y + dy + ys.max()
        x0 = min(x0, int(px0)); y0 = min(y0, int(py0))
        x1 = max(x1, int(px1)); y1 = max(y1, int(py1))
    if x1 < 0:
        return None
    return x0, y0, x1, y1

# ----------------------------------------------------------------------------
# 生成单张
# ----------------------------------------------------------------------------
def render_with_brackets(bg, sprites, target_w, target_h, clss_pool, rng, loose_sprites=None):
    """
    返回 (image为RGB np.uint8, boxes列表[(cls, cx,cy,w,h 归一化), ...])。
    背景已变化; 在墙上随机放 1~3 个支架, 每个以 clss_pool 依概率选正常/松脱。
    loose_sprites: 真实的"松脱"贴片目录; 提供则松脱类直接用真贴片(只加轻抖动), 否则把正常贴片扰动成松脱。
    """
    im = vary_background(bg, target_w, target_h, rng)
    n_sprites = len(sprites)
    if n_sprites == 0:
        return im, []
    if loose_sprites is None:
        loose_sprites = []
    n_real_loose = len(loose_sprites)
    boxes = []
    n_obj = rng.choices([1, 2, 3], weights=[0.55, 0.32, 0.13])[0]
    # 从大到小排背景深度: 先放"更远"的(更小、在上), 后放更近的(更大、在下), 近的盖住部分
    placed = []
    for _ in range(n_obj):
        cls = rng.choice(clss_pool)
        # 选择贴片来源: 松脱优先用真松脱贴片, 否则用正常贴片
        if cls == 1 and n_real_loose > 0:
            sp = loose_sprites[rng.randrange(n_real_loose)]
            src_is_real_loose = True
        else:
            sp = sprites[rng.randrange(n_sprites)]
            src_is_real_loose = False
        # 尺寸(占图宽比例) —— 真实隧道支架一般较小
        rel_w = rng.uniform(0.10, 0.26)
        tw = int(rel_w * target_w)
        th = int(tw * sp.shape[0] / max(1, sp.shape[1]))
        if tw < 12 or th < 12:
            continue
        s = cv2.resize(sp, (tw, th), interpolation=cv2.INTER_AREA)
        if src_is_real_loose:
            kind = K_NORMAL   # 真松脱贴片: 只做轻抖动(旋转/平移), 保留真实松脱形态
        else:
            kind = K_NORMAL if cls == 0 else rng.choice(LOOSE_KINDS)
        pieces = make_pieces(s, kind, rng)
        # 放置位置: 上方侧壁(管线支架常在那)
        nx = rng.uniform(0.12, 0.88)
        cx = int(nx * target_w)
        cy = int(rng.uniform(0.16, 0.46) * target_h)
        # 靠近左右边缘时横向透视压缩
        fx = 1.0 - 0.30 * abs(nx - 0.5) / 0.5
        for i, (dx, dy, p) in enumerate(pieces):
            pw2 = int(p.shape[1] * fx)
            if pw2 != p.shape[1]:
                p = cv2.resize(p, (pw2, int(p.shape[0] * fx)), interpolation=cv2.INTER_AREA)
                pieces[i] = (dx, dy, p)
        # 由 cy 决定绘制次序(越靠下越近, 后画盖前面)
        placed.append((cy, cx, pieces, cls))
    # 按 cy 排序, 下部的最后画
    placed.sort(key=lambda t: t[0])

    # 先轻羽化整体(避免后期硬边): 已在 warp 里做; 这里做颜色匹配 + 阴影 + 粘贴
    for cy, cx, pieces, cls in placed:
        # 颜色匹配(按首 piece)
        p0 = pieces[0][2]
        cx_adj = cx - p0.shape[1] // 2
        cy_adj = cy - p0.shape[0] // 2
        pieces = [(dx, dy, color_match(p, im, cx_adj + dx, cy_adj + dy)) for dx, dy, p in pieces]
        # 阴影 + 粘贴
        for dx, dy, p in pieces:
            px = cx_adj + dx; py = cy_adj + dy
            strength = rng.uniform(0.18, 0.40)
            add_shadow(im, p, px, py, strength)
        for dx, dy, p in pieces:
            paste_piece(im, p, cx_adj + dx, cy_adj + dy)
        bb = pieces_bbox(pieces, cx_adj, cy_adj)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        # 框外扩 2%
        pad = 0.02
        dw, dh = x1 - x0, y1 - y0
        x0 -= int(dw * pad); y0 -= int(dh * pad); x1 += int(dw * pad); y1 += int(dh * pad)
        x0 = max(0, x0); y0 = max(0, y0); x1 = min(target_w, x1); y1 = min(target_h, y1)
        w = x1 - x0; h = y1 - y0
        if w < 4 or h < 4:
            continue
        boxes.append((cls, (x0 + w / 2) / target_w, (y0 + h / 2) / target_h, w / target_w, h / target_h))
    return im, boxes

# ----------------------------------------------------------------------------
# 数据集组装 / 划分 / 统计 / data.yaml
# ----------------------------------------------------------------------------
def yolo_line(b):
    cls, cx, cy, w, h = b
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"

def write_dataset(root, entries):
    """
    entries: list[(image_ndarray, boxes, stem)]
    写入 YOLO 目录结构, 返回统计 dict。
    """
    for sp in ("images", "labels"):
        for s in ("train", "val", "test"):
            os.makedirs(os.path.join(root, sp, s), exist_ok=True)
    target_w, target_h = entries[0][0].shape[1], entries[0][0].shape[0]

    manifest = []
    stats = collections.defaultdict(lambda: collections.Counter())  # split -> {cls: n}
    imgcnt = collections.Counter()
    # 层次划分(按是否含松脱分层, 保证每 split 都有松脱)
    has_loose = [e for e in entries if any(b[0] != 0 for b in e[1])]
    only_normal = [e for e in entries if not any(b[0] != 0 for b in e[1])]
    R = random.Random(42)
    for pool in (has_loose, only_normal):
        R.shuffle(pool)
        n = len(pool)
        ntr = int(n * 0.8); nva = int(n * 0.1)
        for i, (im, boxes, stem) in enumerate(pool):
            s = "train" if i < ntr else ("val" if i < ntr + nva else "test")
            cv2.imwrite(os.path.join(root, "images", s, stem + ".jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            with open(os.path.join(root, "labels", s, stem + ".txt"), "w") as f:
                for b in boxes:
                    f.write(yolo_line(b))
            manifest.append((s, stem))
            imgcnt[s] += 1
            for b in boxes:
                stats[s][b[0]] += 1

    with open(os.path.join(root, "_split_manifest.txt"), "w") as f:
        for s, stem in manifest:
            f.write(f"{s}\t{stem}\n")

    # data.yaml
    names = DEFAULT_NAMES
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(root)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {len(names)}\nnames: {{{', '.join(str(i)+': '+repr(n) for i,n in enumerate(names))}}}\n")

    # stats
    with open(os.path.join(root, "_stats.txt"), "w") as f:
        f.write("=== 各split 每类实例计数 ===\n")
        for s in ("train", "val", "test"):
            cnt = stats[s]
            line = f"{s}: 图{imgcnt[s]} | " + "  ".join(f"{names[c]}={cnt[c]}" for c in range(len(names)))
            f.write(line + "\n")
        f.write("=== 总计 ===\n")
        tot = collections.Counter()
        for s in stats:
            for c, n in stats[s].items():
                tot[c] += n
        f.write("  " + "  ".join(f"{names[c]}={tot[c]}" for c in range(len(names))) + f"  (总{sum(tot.values())})\n")
    return stats

# ----------------------------------------------------------------------------
# 预览
# ----------------------------------------------------------------------------
def preview_mosaic(samples, cols=5, pad=8, lbl_h=22):
    ims = []
    for img, boxes in samples:
        v = img.copy()
        for cls, cx, cy, w, h in boxes:
            x0 = int((cx - w / 2) * img.shape[1]); y0 = int((cy - h / 2) * img.shape[0])
            x1 = int((cx + w / 2) * img.shape[1]); y1 = int((cy + h / 2) * img.shape[0])
            # x0/x1 已是全分辨率像素, 直接在 v(全分辨率) 上画, 最后统一 resize 到 640x360
            color = (40, 120, 240) if cls == 0 else (40, 40, 230)
            cv2.rectangle(v, (max(0, x0), max(0, y0)), (min(v.shape[1], x1), min(v.shape[0], y1)), color, 2)
            cv2.putText(v, DEFAULT_NAMES[cls], (max(0, x0), max(0, y0) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        ims.append(cv2.resize(v, (640, 360), interpolation=cv2.INTER_AREA))
    if not ims:
        print("  [预览] 无样本")
        return None
    rows = (len(ims) + cols - 1) // cols
    th = int(360 * 1.0) + lbl_h
    canvas = np.full((rows * th, cols * (640 + pad) + pad, 3), 255, np.uint8)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        y0 = r * th; x0 = c * (640 + pad) + pad
        canvas[y0:y0 + 360, x0:x0 + 640] = im
    return canvas

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="设备管线支架松脱 数据集 真实贴片合成生成器")
    ap.add_argument("--sprites", required=True, help="RGBA支架贴片目录(透明背景)")
    ap.add_argument("--loose_sprites", default=None, help="可选: 真松脱贴片目录(松脱类优先用真贴片)")
    ap.add_argument("--backgrounds", required=True, help="隧道墙面背景图目录")
    ap.add_argument("--output", default="output", help="数据集根目录")
    ap.add_argument("--count", type=int, default=1000, help="生成图片数")
    ap.add_argument("--width", type=int, default=1280, help="输出宽")
    ap.add_argument("--height", type=int, default=720, help="输出高")
    ap.add_argument("--loose_ratio", type=float, default=0.5, help="含松脱实例的比例(0~1)")
    ap.add_argument("--preview", action="store_true", help="只出网格预览不写数据集")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    sprites = load_sprites(args.sprites)
    loose_sprites = load_sprites(args.loose_sprites) if args.loose_sprites else []
    bgs = load_backgrounds(args.backgrounds)
    print(f"支架贴片(正常): {len(sprites)} 张  真松脱贴片: {len(loose_sprites)} 张  背景: {len(bgs)} 张")
    if not sprites:
        print("!! 没有可用支架贴片(需透明背景RGBA)。请先 用 制备素材.py / 抠支架精灵.py 抠出贴片。")
        sys.exit(1)
    if not bgs:
        print("!! 没有背景图。请提供隧道墙面背景图目录。")
        sys.exit(1)

    # 类别池: 正常 / 松脱
    clss_pool = [0, 1] if rng.random() < args.loose_ratio else [0, 0, 1]
    # (上面那个随机一次; 改成按 global ratio 决定每批次 —— 用稳定权重)
    clss_pool = [0] * int(round(10 * (1 - args.loose_ratio))) + [1] * int(round(10 * args.loose_ratio))
    clss_pool = clss_pool or [0]

    # 分pool: 因为每张图可能多目标, 简单按目标级比例即可
    def gen_one():
        bg = bgs[rng.randrange(len(bgs))]
        im, boxes = render_with_brackets(bg, sprites, args.width, args.height, clss_pool, rng, loose_sprites=loose_sprites)
        return im, boxes

    if args.preview:
        samples = []
        for k in range(8):
            im, boxes = gen_one()
            samples.append((im, boxes))
        grid = preview_mosaic(samples)
        if grid is not None:
            cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview_网格.png"), grid)
            print(f"预览已写: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preview_网格.png')}")
        return

    # 批量
    entries = []
    for i in range(args.count):
        im, boxes = gen_one()
        stem = f"bracket_{i:06d}"
        entries.append((im, boxes, stem))
    stats = write_dataset(args.output, entries)
    print(f"完成: {args.count} 张 -> {args.output}")
    print(f"label示例: {yolo_line(entries[0][1][0]) if entries[0][1] else '无框'}")
    with open(os.path.join(args.output, "_stats.txt")) as f:
        print(f.read())

if __name__ == "__main__":
    main()
