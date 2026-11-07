#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
制备素材 —— 从 4 张真实"管线支架松脱"图(04xx)抠出贴片, 供 生成_支架松脱数据集.py 使用。
================================================
- 松脱贴片: 直接用 类4 真框(ground-truth) 扩展 -> GrabCut 细化 -> 真实松脱贴片
- 正常贴片: 手动粗框(GrabCut) 从图里抠出"正常支架"部件
- 背景: 复用项目里干净混凝土墙面 data/设备_正常/*.jpg (无需在此处理)

用法:
  python3 制备素材.py --src <04xx图文件夹> --labels <labels文件夹> \
      --out 素材 --preview
  (--preview 只出 抠图_预览.png; 不加则写 sprites_normal/ sprites_loose/)
"""
import os, sys, argparse, glob
import numpy as np
import cv2

# 正常支架贴片: 只在"上部干净亮区"收紧框(避开左下角水印 y>1600 与底部今日水印)。
# 类4松脱框中心点随图不同; 这里只放画面上明显正常、且远离水印的支架。
NORMAL_BOXES = {
    "0401.png": [
        [445, 175, 1125, 470],    # 左上方镀锌托架(亮, 背景深)
        [850, 182, 2135, 512],    # 顶部横向托架(带螺栓)
        [2250, 200, 3200, 600],   # 右上方托架
    ],
    "0402.png": [
        [185, 505, 1450, 1285],   # 镀锌槽钢+两颗螺栓(正常段)
        [300, 0, 1600, 700],      # 顶部黑色保温管
    ],
}
# 松脱贴片若扩展框底边压到蓝色水印块(左下)或今日水印, 则跳过
WATERMARK_Y = 1640

def loose_rect_collides_watermark(y1, x0, x1, W):
    if y1 > WATERMARK_Y:
        return True
    # 左下角蓝色水印块: 大致 x<1950 且 y>1650(W=4032)
    return False

def load_label_loose(labels_dir, stem):
    """返回该图 类4 的所有框 [(cx,cy,w,h) 归一化]。"""
    p = os.path.join(labels_dir, stem + ".txt")
    boxes = []
    if not os.path.exists(p):
        return boxes
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5 or int(parts[0]) != 4:
            continue
        boxes.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return boxes

def box_to_rect_norm(cx, cy, w, h, iw, ih, expand=1.6):
    """归一化中心+宽高 -> 扩展后的像素(左上角,宽,高)。"""
    cw = w * iw; ch = h * ih
    cx = cx * iw; cy = cy * ih
    ncw = cw * expand; nch = ch * expand
    x = max(0, int(cx - ncw / 2)); y = max(0, int(cy - nch / 2))
    return (x, y, min(int(ncw), iw - x), min(int(nch), ih - y))

def grabcut_sprite(img, box, feather=2.5, max_w=1100):
    """内框当前景种子 + 最大连通域(只留中央主体), 避免把周围背景/水印抠进来。"""
    h, w = img.shape[:2]
    x, y, bw, bh = box
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        x = int(x * scale); y = int(y * scale); bw = int(bw * scale); bh = int(bh * scale)
    h, w = img.shape[:2]
    x0 = max(0, x); y0 = max(0, y); x1 = min(w, x + bw); y1 = min(h, y + bh)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    # 前景种子: 框内缩 22%
    fx0 = x0 + int((x1 - x0) * 0.22); fx1 = x1 - int((x1 - x0) * 0.22)
    fy0 = y0 + int((y1 - y0) * 0.22); fy1 = y1 - int((y1 - y0) * 0.22)
    if fx1 > fx0 and fy1 > fy0:
        mask[fy0:fy1, fx0:fx1] = cv2.GC_PR_FGD
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # 保留最大连通域(通常是中央主体), 去除附着背景
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = np.zeros_like(fg); keep[lab == big] = 255
        fg = keep
    fg = cv2.erode(fg, np.ones((2, 2), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(fg, (3, 3), feather if feather > 0 else 0)
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return None
    x0b, x1b = xs.min(), xs.max() + 1; y0b, y1b = ys.min(), ys.max() + 1
    rgba = np.dstack([img, alpha]).astype(np.uint8)
    return rgba[y0b:y1b, x0b:x1b]

def main():
    ap = argparse.ArgumentParser(description="从类4真图抠支架贴片")
    ap.add_argument("--src", required=True, help="04xx 图文件夹")
    ap.add_argument("--labels", required=True, help="labels 文件夹")
    ap.add_argument("--out", default="素材", help="输出目录")
    ap.add_argument("--feather", type=float, default=2.5)
    ap.add_argument("--preview", action="store_true", help="只出预览, 不写库")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "sprites_normal"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "sprites_loose"), exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(args.src, "04*")))
    print(f"类4图: {len(imgs)} 张")
    sprites = []  # (tag, stem, rgba)
    for imp in imgs:
        stem = os.path.splitext(os.path.basename(imp))[0]
        img = cv2.imread(imp)
        if img is None:
            continue
        iw, ih = img.shape[1], img.shape[0]
        # 松脱: 类4真框扩展(若压到水印区则跳过)
        for cx, cy, w, h in load_label_loose(args.labels, stem):
            lx, ly, lw, lh = box_to_rect_norm(cx, cy, w, h, iw, ih)
            if ly + lh > WATERMARK_Y:   # 底边压到水印
                continue
            sp = grabcut_sprite(img, (lx, ly, lw, lh), args.feather)
            if sp is not None:
                sprites.append(("loose", stem, sp))
        # 正常: 手动粗框
        for bx in NORMAL_BOXES.get(os.path.basename(imp), []):
            sp = grabcut_sprite(img, (bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]), args.feather)
            if sp is not None:
                sprites.append(("normal", stem, sp))

    print(f"抠出贴片: {len(sprites)}  (normal={sum(1 for t,_,_ in sprites if t=='normal')}, "
          f"loose={sum(1 for t,_,_ in sprites if t=='loose')})")

    if args.preview:
        from PIL import Image
        cols = 9; cell = 150
        rows = (len(sprites) + cols - 1) // cols
        board = np.full((rows * cell, cols * cell, 3), 245, np.uint8)
        for i, (tag, stem, sp) in enumerate(sprites):
            sp_rgb = sp[:, :, :3]; a = sp[:, :, 3].astype(np.float32) / 255.0
            th, tw = sp.shape[:2]
            sc = min(cell / tw, cell / th)
            if sc < 1:
                sp_rgb = cv2.resize(sp_rgb, (max(1,int(tw*sc)), max(1,int(th*sc))), interpolation=cv2.INTER_AREA)
                a = cv2.resize(a, (sp_rgb.shape[1], sp_rgb.shape[0]), interpolation=cv2.INTER_AREA)
            r, c = divmod(i, cols)
            y0, x0 = r*cell, c*cell
            th, tw = sp_rgb.shape[:2]
            region = board[y0:y0+th, x0:x0+tw].astype(np.float32)
            board[y0:y0+th, x0:x0+tw] = (region*(1-a[...,None]) + sp_rgb.astype(np.float32)*a[...,None]).astype(np.uint8)
            cv2.putText(board, tag[:1], (x0+4, y0+cell-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "素材_预览.png")
        cv2.imwrite(out, board)
        print(f"预览已写: {out}")
        return

    for tag, stem, sp in sprites:
        d = "sprites_loose" if tag == "loose" else "sprites_normal"
        n = len(glob.glob(os.path.join(args.out, d, "*.png")))
        cv2.imwrite(os.path.join(args.out, d, f"{stem}_{tag}_{n:02d}.png"), sp, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"已写贴片库: {args.out}")

if __name__ == "__main__":
    main()
