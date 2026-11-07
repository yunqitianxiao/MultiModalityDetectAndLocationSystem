#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成隧道 base 世界材质贴图(混凝土 + 管片拼缝 + 污渍)
================================================================
给 inspection_tunnel_base.world 的墙/顶/地加 albedo_map, 让场景不再是"素灰",
缺陷矩形照片背景才谈得上融入(否则纯灰墙一眼假)。产物(均写入 worlds/textures):

  tunnel_concrete_wall.png   4096x256  墙(沿x 40m, 高3m): 混凝土噪点 + 1.5m 竖向管片拼缝
                                       + 一道纵向环缝 + 污渍/水痕, 直接拉伸覆盖整墙
  tunnel_concrete_floor.png  2048x2048 地面: 混凝土噪点 + 轻微污渍
  tunnel_concrete_ceiling    复用 wall 贴图(顶横断面 40x3, 同为长条)

贴图按墙的实际宽高比拉伸(40:3 ≈ 13.3:1), 拼缝间隔 = 宽/管片数(4082/27≈152px=1.5m),
避免依赖 gz PBR 的 UV 平铺(其会一次性拉伸, 不支持 repeat), 直接用"单图整墙"方案。
"""
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "worlds", "textures")


def _lumpy(rng, w, h, method="blobs"):
    """低频大块灰度噪声(±amp), 模拟混凝土浇筑不均匀。"""
    if method == "blobs":
        s = 16
        small = rng.random((h // s + 2, w // s + 2), dtype=np.float32)
        small = small.astype(np.float32)
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        big = cv2.GaussianBlur(big, (0, 0), 8)
        return (big - big.mean())  # 零均值
    # stripes: 竖向水痕(沿x的纵向深浅带)
    col = (rng.uniform(-1, 1, (1, w))).astype(np.float32)
    col = cv2.resize(col, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(col, (0, 0), 6)


def concrete_wall(w=4096, h=256, panels=27, albedo_hi=160, albedo_lo=120, seed=7):
    """生成一张可拉伸覆盖整墙的混凝土+拼缝贴图。"""
    rng = np.random.default_rng(seed)
    # 基底: 中灰逐步到 0.55 附近, 加低频大块 + 细颗粒
    base = np.full((h, w), 140, dtype=np.float32)
    base += _lumpy(rng, w, h, "blobs") * 14
    base += _lumpy(rng, w, h, "stripes") * 8
    base += rng.normal(0, 5, (h, w)).astype(np.float32)          # 细颗粒
    base = np.clip(base, albedo_lo, albedo_hi)
    # 纵向水痕污渍(几条暗色竖条)
    for _ in range(rng.integers(4, 8)):
        cx = int(rng.integers(60, w - 60))
        wid = int(rng.integers(40, 140))
        dark = base[:, max(0, cx - wid // 2):cx + wid // 2] * 0.92
        g = cv2.GaussianBlur(dark, (0, 0), wid // 8)
        base[:, max(0, cx - wid // 2):cx + wid // 2] = g
    # 管片拼缝(竖向), 及一道纵向环缝(横向)
    seam_col = 0.42 * 255 * 0.55    # 深灰缝
    step = w / panels
    for i in range(1, panels):
        x = int(round(i * step))
        base[:, max(0, x - 2):x + 2] = base[:, max(0, x - 2):x + 2] * 0.45 + seam_col * 0.55
    y = h // 2
    base[max(0, y - 2):y + 2, :] = base[max(0, y - 2):y + 2, :] * 0.55 + seam_col * 0.45
    # 砼表面微孔/斑点
    speck = (np.abs(rng.normal(0, 1, (h, w))) < 0.05).astype(np.float32) * 22
    base = np.clip(base + speck, albedo_lo, albedo_hi)
    tex = np.stack([base] * 3, axis=-1).astype(np.uint8)         # BGR 灰
    # 微偏蓝(与旧 wall diffuse 0.55/0.55/0.58 一致)
    b = tex[..., 0].astype(np.int16) * 0.99 + 2
    tex[..., 0] = np.clip(b, 0, 255).astype(np.uint8)
    return tex


def concrete_floor(size=2048, seed=11):
    """地面混凝土: 无拼缝, 大块噪点 + 中部轨床略暗。"""
    rng = np.random.default_rng(seed)
    base = np.full((size, size), 128, dtype=np.float32)
    base += _lumpy(rng, size, size, "blobs") * 16
    base += rng.normal(0, 4, (size, size)).astype(np.float32)
    base = np.clip(base, 105, 152)
    # 中部(# 轨床)略暗: 隧道中线附近 x=0.45~0.55
    c0, c1 = int(0.40 * size), int(0.60 * size)
    base[:, c0:c1] = base[:, c0:c1] * 0.9
    tex = np.stack([base] * 3, axis=-1).astype(np.uint8)
    return tex


def main():
    os.makedirs(TEX, exist_ok=True)
    wall = concrete_wall()
    floor_ = concrete_floor()
    cv2.imwrite(os.path.join(TEX, "tunnel_concrete_wall.png"), wall)
    cv2.imwrite(os.path.join(TEX, "tunnel_concrete_floor.png"), floor_)
    print(f"✅ 已生成:")
    for f in ("tunnel_concrete_wall.png", "tunnel_concrete_floor.png"):
        p = os.path.join(TEX, f)
        print(f"   {p}  ({os.path.getsize(p)//1024} KB)")


if __name__ == "__main__":
    main()
