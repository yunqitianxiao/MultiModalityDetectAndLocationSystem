# -*- coding: utf-8 -*-
"""
attrmap 可视化抽样 —— 确认 depth/reflec/height 语义与病害贴图位置对应
================================================================
把输出目录的 color 图像 + attrmap(depth/reflec/height 三通道) + GT 框画在一张图上,
直观验证: 病害(墙/轨, 有 laser_retro)在 reflec 通道高亮、depth 呈现近墙面深度、
height 呈现墙面/地面高度。抽样若干帧落盘 PNG, 用 Read 逐个看。

用法:
    .venv/bin/python fusion/viz_attrmap_sample.py \
        --data data_gen/output_data1 --frames 0,40,80,120 --out runs/attrmap_viz_data1
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')        # 无显示环境
import matplotlib.pyplot as plt
import numpy as np

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)


def _norm(arr, vmin=None, vmax=None):
    """线性归一化到 [0,1] 用于显示。vmin/vmax 可手动收紧, 否则取有限 min/max。"""
    with np.errstate(invalid='ignore'):
        lo = np.nanmin(arr) if vmin is None else vmin
        hi = np.nanmax(arr) if vmax is None else vmax
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def draw_boxes(ax, labels, shape):
    """把 YOLO 归一化框画到归一化 axes 上。labels: (N,5) cls cx cy w h。"""
    W, H = shape[1], shape[0]
    for lb in labels:
        if lb.shape[0] == 0:
            continue
        c, cx, cy, w, h = lb
        x = (cx - w / 2) * W
        y = (cy - h / 2) * H
        rect = plt.Rectangle((x, y), w * W, h * H, fill=False,
                             edgecolor='red', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, max(0, y - 3), f"{int(c)}", color='yellow', fontsize=7)


def _load_labels(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.empty((0, 5), np.float32)
    return np.loadtxt(path, dtype=np.float32).reshape(-1, 5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data_gen/output_data1')
    p.add_argument('--frames', default='0,40,80,120', help='逗号分隔帧号')
    p.add_argument('--out', default='runs/attrmap_viz_data1')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames = [int(x) for x in args.frames.split(',')]
    cmap_d = plt.get_cmap('viridis')
    print(f"采样 {frames} 帧 -> {args.out}")

    for i in frames:
        tag = f'{i:06d}'
        img_p = os.path.join(args.data, 'images', f'{tag}.jpg')
        attr_p = os.path.join(args.data, 'attrmaps', f'{tag}.npz')
        lbl_p = os.path.join(args.data, 'labels', f'{tag}.txt')

        img = cv2.imread(img_p)
        if img is None:
            print(f"  !! 读不到 {img_p}, 跳过"); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        a = np.load(attr_p)
        depth, reflec, height = a['depth'], a['reflec'], a['height']
        labels = _load_labels(lbl_p)

        fig, axs = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f'frame {tag}  ({len(labels)} GT)  [attrmap {depth.shape[0]}x{depth.shape[1]}]',
                     fontsize=13)

        # 颜色图像
        axs[0, 0].imshow(img); axs[0, 0].set_title('RGB'); axs[0, 0].axis('off')
        draw_boxes(axs[0, 0], labels, img.shape)

        # depth: viridis
        axs[0, 1].imshow(_norm(depth), cmap=cmap_d)
        axs[0, 1].set_title(f'depth [{np.nanmin(depth):.1f},{np.nanmax(depth):.1f}] m')
        axs[0, 1].axis('off'); draw_boxes(axs[0, 1], labels, depth.shape)

        # reflec: 灰度 (0-1, 缺陷 retro 高亮)
        axs[1, 0].imshow(_norm(reflec), cmap='Greys')
        axs[1, 0].set_title(f'reflec [{np.nanmin(reflec):.2f},{np.nanmax(reflec):.2f}]')
        axs[1, 0].axis('off'); draw_boxes(axs[1, 0], labels, reflec.shape)

        # height: plasma
        axs[1, 1].imshow(_norm(height, vmin=None, vmax=None), cmap='plasma')
        axs[1, 1].set_title(f'height [{np.nanmin(height):.2f},{np.nanmax(height):.2f}] m')
        axs[1, 1].axis('off'); draw_boxes(axs[1, 1], labels, height.shape)

        out_p = os.path.join(args.out, f'{tag}.png')
        plt.tight_layout()
        plt.savefig(out_p, dpi=120)
        plt.close(fig)
        print(f"  ✅ {out_p}")


if __name__ == '__main__':
    main()
