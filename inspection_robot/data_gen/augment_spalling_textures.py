#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 data_gen/textures/spalling_*.png 扩成变体贴图集。

defect_placement 的每类实例数被"唯一贴图数"限制（n_inst = min(n_tex, cap)），
spalling 只有 5 张 -> 每景最多 5 个实例，与裂缝的 16 个不同量级。这里做
翻转/亮度变体扩到 20 张，使单类实验的每景实例数可与裂缝配置对齐。

用法: python3 data_gen/augment_spalling_textures.py
"""
import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "textures")
DST = os.path.join(HERE, "textures_spalling")


def main():
    os.makedirs(DST, exist_ok=True)
    for f in glob.glob(os.path.join(DST, "*.png")):
        os.remove(f)
    srcs = sorted(glob.glob(os.path.join(SRC, "spalling_*.png")))
    if not srcs:
        raise SystemExit(f"未找到源贴图: {SRC}/spalling_*.png")
    k = 0
    for p in srcs:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"读图失败: {p}")
        variants = {
            "orig": img,
            "flipx": cv2.flip(img, 1),
            "flipy": cv2.flip(img, 0),
            "bright": np.clip(img.astype(np.float32) * 1.25, 0, 255).astype(img.dtype),
        }
        for name, v in variants.items():
            k += 1
            cv2.imwrite(os.path.join(DST, f"spalling_{k}.png"), v)
    print(f"✅ {DST}: {k} 张 (源 {len(srcs)} × 4 变体)")


if __name__ == "__main__":
    main()
