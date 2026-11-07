#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 SAM (Segment Anything, ViT-B) 为未标注的管片破损掉块图自动生成掩膜(预标注)

背景: 已有 154 张管片掩膜, 剩 76 张未标注。用 SAM 自动分割作为预标注,
      人工在掩膜标注工具里复核/修整后即可进入训练集。

输入: data/土建_管片掉块/*.jpg|*.png
      (跳过已有 *_mask.png 的, 也跳过 data/土建_管片掉块_已复核.txt 里
       人工标注工具确认过"无缺陷"的图, 避免被重新覆盖)
输出:
  data/土建_管片掉块_掩膜/<名>_mask.png   二值掩膜(白=缺陷255 / 黑=背景0)
  data/土建_管片掉块_SAM预览/<名>.png     原图 + SAM各掩膜彩色叠加, 供人工review

用法:
  python3 SAM_生成管片掩膜.py               # 处理所有未标注图(GPU)
  python3 SAM_生成管片掩膜.py --preview-only # 只出预览, 不写掩膜
  python3 SAM_生成管片掩膜.py --max N        # 只处理前 N 张(先试跑)

说明:
  - SAM 无提示自动分割会分出整图所有目标, 这里按管片掉块经验筛选:
    面积占比 1%~50%(对标已标注掩膜: 中位4%, 88%>1%), 结合 SAM 的 IoU/稳定性分数。
  - 自动结果仅供预标注, 掉块定位可能不准, 务必人工复核后再训练。
"""
import argparse
import os
import sys
import glob
import numpy as np
import torch
from PIL import Image

# ============ 配置 ============
PROJ = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJ, "data", "土建_管片掉块")
MASK_DIR = os.path.join(PROJ, "data", "土建_管片掉块_掩膜")
PREVIEW_DIR = os.path.join(PROJ, "data", "土建_管片掉块_SAM预览")
REVIEWED_FILE = os.path.join(PROJ, "data", "土建_管片掉块_已复核.txt")
CHECKPOINT = os.path.join(PROJ, "sam_models", "sam_vit_b_01ec64.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAM 自动分割参数(点数少一点, 速度更快; 分数阈值高一点, 只留可靠的掩膜)
POINTS_PER_SIDE = 24
PRED_IOU_THRESH = 0.86
STABILITY_THRESH = 0.90
MIN_REGION_AREA = 300

# 掉块筛选(对标已标注掩膜的面积分布: 中位4%, 45%>5%)
# 注意: SAM 无提示自动分割会把整块墙面/阴影等大区域也分出, 面积过大的是干扰,
#       故上限收紧到 15%(对标标注里绝大部分掉块), 且只取面积最大的 2 个合并。
MIN_AREA_PCT = 0.01     # 前景 >= 1% 图面积
MAX_AREA_PCT = 0.15     # 前景 <= 15% 图面积(排除墙面/阴影等大区域)
MAX_MASKS = 2           # 合并前 N 大掩膜


def get_unlabeled():
    """找出没有掩膜且未被人工复核的源图(跳过标注工具里已确认'无缺陷'的图)"""
    reviewed = set()
    if os.path.exists(REVIEWED_FILE):
        with open(REVIEWED_FILE, "r", encoding="utf-8") as f:
            reviewed = {line.strip() for line in f if line.strip()}
    unlabeled = []
    for f in sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")) +
                    glob.glob(os.path.join(SRC_DIR, "*.png"))):
        name = os.path.splitext(os.path.basename(f))[0]
        mask = os.path.join(MASK_DIR, f"{name}_mask.png")
        if not os.path.exists(mask) and name not in reviewed:
            unlabeled.append(f)
    return unlabeled


def filter_masks(anns, img_area):
    """按面积占比 + 分数筛选, 返回按面积降序的掩膜"""
    kept = []
    for a in anns:
        pct = a["area"] / img_area
        if MIN_AREA_PCT <= pct <= MAX_AREA_PCT:
            if a.get("predicted_iou", 0) >= PRED_IOU_THRESH:
                kept.append(a)
    kept.sort(key=lambda a: a["area"], reverse=True)
    return kept[:MAX_MASKS]


def draw_overlay(image, masks):
    """原图 + 掩膜彩色叠加(每类随机色)"""
    overlay = image.copy()
    rng = np.random.RandomState(0)
    for m in masks:
        color = rng.randint(0, 255, 3).tolist()
        overlay[m["segmentation"]] = (0.5 * overlay[m["segmentation"]] +
                                      0.5 * np.array(color)).astype(np.uint8)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview-only", action="store_true", help="只生成预览图, 不写掩膜")
    ap.add_argument("--max", type=int, default=0, help="只处理前 N 张(0=全部)")
    args = ap.parse_args()

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    if not args.preview_only:
        os.makedirs(MASK_DIR, exist_ok=True)

    files = get_unlabeled()
    if args.max > 0:
        files = files[:args.max]
    if not files:
        print("没有未标注的图。")
        return

    print(f"[SAM] 加载模型: {CHECKPOINT}  ({DEVICE})")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT).to(DEVICE)
    generator = SamAutomaticMaskGenerator(
        sam, points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_THRESH,
        min_mask_region_area=MIN_REGION_AREA,
    )

    print(f"[SAM] 共 {len(files)} 张未标注图, 开始自动分割...")
    for i, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        image = np.array(Image.open(path).convert("RGB"))
        img_area = image.shape[0] * image.shape[1]

        anns = generator.generate(image)
        kept = filter_masks(anns, img_area)

        # 预览
        overlay = draw_overlay(image, kept)
        Image.fromarray(overlay).save(os.path.join(PREVIEW_DIR, f"{name}.png"))

        # 掩膜
        if not args.preview_only:
            mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
            for m in kept:
                mask[m["segmentation"]] = 255
            Image.fromarray(mask).save(os.path.join(MASK_DIR, f"{name}_mask.png"))

        n = len(anns)
        print(f"  [{i}/{len(files)}] {name}: SAM掩膜{n}个 -> 保留{len(kept)}个 "
              f"{'(预览)' if args.preview_only else '(已写掩膜)'}")

    print(f"\n完成! 预览: {PREVIEW_DIR}")
    if not args.preview_only:
        print(f"掩膜: {MASK_DIR}  (请用 管片破损掉块_掩膜标注工具.py 复核修整)")


if __name__ == "__main__":
    main()
