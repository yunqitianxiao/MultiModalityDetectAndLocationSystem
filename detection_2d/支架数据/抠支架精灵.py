#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抠支架精灵 —— 从真实隧道图里抠出"管线支架"透明贴片(RGBA), 供合成引擎使用。
================================================
默认用 cv2.GrabCut(免torch, CPU很快): 你只需要给每个支架一个"粗框", 它会自动细化出前景抠图。
如果你的素材较模糊(如你说的"不清晰/不够"), 建议:
  - 粗框尽量贴着支架外围, 让 GrabCut 少猜;
  - 抠出后先 `--preview` 看贴片干不干净, 再进合成引擎;
  - 想要更高精度可选 `--sam`(用你已缓存的 sam_vit_b, 但 torch 目前是 CPU, 会慢)。

输入框的 txt 格式(每张图一个同名 txt, 8月23日惯例, 每行): x y w h   (像素, 左上角)
  例: 支架图_001.jpg  ->  支架图_001.txt  ->   "312 88 156 120"

用法:
  python3 抠支架精灵.py --src 你的支架图文件夹 --box 粗框txt文件夹 --output 贴片库 --upsample 2
  python3 抠支架精灵.py --src ... --box ... --preview   # 只拼一张检查图, 不写库

可选: --mask 已有二值掩膜目录(来自 设备_异物入侵_掩膜标注工具), 直接按掩膜抠, 不走 GrabCut。
输出: 透明背景 PNG, 文件名为 <源图stem>_<序号>.png
"""
import os, sys, random, argparse, glob
import numpy as np
import cv2

def load_boxes(box_dir, stem):
    """返回 list[(x,y,w,h)]。找不到框返回 [](该图可跳过)。"""
    for ext in (".txt",):
        p = os.path.join(box_dir, stem + ext)
        if os.path.exists(p):
            boxes = []
            for line in open(p):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                x, y, w, h = (int(float(v)) for v in parts[:4])
                if w > 0 and h > 0:
                    boxes.append((x, y, w, h))
            return boxes
    return []

def grabcut_sprite(img, box, feather=2.5, expand=0.02):
    h, w = img.shape[:2]
    x, y, bw, bh = box
    # 粗框外扩一点容错
    dx = int(bw * expand); dy = int(bh * expand)
    x = max(0, x - dx); y = max(0, y - dy)
    x1 = min(w, x + bw + 2 * dx); y1 = min(h, y + bh + 2 * dy)
    bw, bh = x1 - x, y1 - y
    if bw < 8 or bh < 8:
        return None
    rect = (x, y, bw, bh)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # 轻微腐蚀再去 halo, 再羽化
    fg = cv2.erode(fg, np.ones((2, 2), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(fg, (3, 3), feather if feather > 0 else 0)
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    rgba = np.dstack([img, alpha]).astype(np.uint8)
    return rgba[y0:y1, x0:x1]

def mask_sprite(img, mask_img, feather=2.5):
    alpha = cv2.GaussianBlur(mask_img, (3, 3), feather if feather > 0 else 0)
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    rgba = np.dstack([img, alpha]).astype(np.uint8)
    return rgba[y0:y1, x0:x1]

def main():
    ap = argparse.ArgumentParser(description="抠透明背景支架贴片(RGBA)")
    ap.add_argument("--src", required=True, help="源支架图文件夹")
    ap.add_argument("--box", default=None, help="粗框txt文件夹(GrabCut用)")
    ap.add_argument("--mask", default=None, help="已有二值掩膜文件夹(跳过GrabCut)")
    ap.add_argument("--output", required=True, help="输出贴片库文件夹")
    ap.add_argument("--upsample", type=float, default=1.0, help="输出放大倍数(模糊素材可放大提分辨率)")
    ap.add_argument("--feather", type=float, default=2.5, help="边缘羽化强度")
    ap.add_argument("--preview", action="store_true", help="只拼一张检查图, 不写库")
    ap.add_argument("--sam", action="store_true", help="可选: 用SAM代替GrabCut(当前torch为CPU, 会慢)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        imgs += glob.glob(os.path.join(args.src, ext))
    imgs = sorted(imgs)
    print(f"源图: {len(imgs)} 张")

    if args.sam:
        # 可选 SAM
        try:
            import torch
            from segment_anything import sam_model_registry, SamPredictor
        except Exception as e:
            print(f"!! SAM不可用({e}), 回退GrabCut"); args.sam = False

    sprites = []
    for imp in imgs:
        stem = os.path.splitext(os.path.basename(imp))[0]
        img = cv2.imread(imp)
        if img is None:
            continue
        if args.mask:
            mp = os.path.join(args.mask, stem + ".png")
            if not os.path.exists(mp):
                mp = os.path.join(args.mask, stem + ".jpg")
            if not os.path.exists(mp):
                continue
            m = cv2.imread(mp, 0)
            if m is None:
                continue
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            pieces = [mask_sprite(img, m, args.feather)]
        else:
            boxes = load_boxes(args.box, stem) if args.box else []
            pieces = []
            for box in boxes:
                if args.sam and globals().get("sam_predictor"):
                    sam_predictor.set_image(img)
                    x, y, bw, bh = box
                    for _ in range(1):
                        import torch
                        from segment_anything import SamPredictor as _  # noqa
                    # 用 prompt box 预测
                    from segment_anything import SamPredictor as __
                    import numpy as np
                    input_box = np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]])
                    masks, scores, _ = sam_predictor.predict(
                        point_coords=None, point_labels=None, box=input_box, multimask_output=False)
                    best = (masks[0].astype(np.uint8) * 255)
                    pieces.append(mask_sprite(img, best, args.feather))
                else:
                    pieces.append(grabcut_sprite(img, box, args.feather))
        for j, sp in enumerate(pieces):
            if sp is None:
                continue
            if args.upsample != 1.0:
                sp = cv2.resize(sp, (int(sp.shape[1]*args.upsample), int(sp.shape[0]*args.upsample)),
                                interpolation=cv2.INTER_CUBIC)
            sprites.append((stem, j, sp))

    print(f"抠出贴片: {len(sprites)} 张")
    if args.preview:
        cols = 8; cell = 140
        rows = (len(sprites) + cols - 1) // cols
        board = np.full((rows*cell, cols*cell, 3), 245, np.uint8)
        for i, (stem, j, sp) in enumerate(sprites):
            sp_rgb = sp[:, :, :3]
            a = sp[:, :, 3].astype(np.float32)/255.0
            r, c = divmod(i, cols)
            y0, x0 = r*cell, c*cell
            th, tw = sp.shape[:2]
            sc = min(cell/tw, cell/th)
            if sc < 1:
                sp_rgb = cv2.resize(sp_rgb, (int(tw*sc), int(th*sc)))
                a2 = cv2.resize(a.astype(np.uint8).astype(np.float32), (int(tw*sc), int(th*sc)))
                th, tw = sp_rgb.shape[:2]
            else:
                a2 = a
            region = board[y0:y0+th, x0:x0+tw].astype(np.float32)
            board[y0:y0+th, x0:x0+tw] = (region*(1-a2[...,None]) + sp_rgb.astype(np.float32)*a2[...,None]).astype(np.uint8)
        cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), "抠图_预览.png"), board)
        print(f"预览已写: {os.path.join(os.path.dirname(os.path.abspath(__file__)), '抠图_预览.png')}")
        return

    for stem, j, sp in sprites:
        cv2.imwrite(os.path.join(args.output, f"{stem}_{j}.png"), sp, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print(f"已写入贴片库: {args.output}")

if __name__ == "__main__":
    main()
