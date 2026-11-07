# -*- coding: utf-8 -*-
"""
定位链闭环验证 —— 合成隧道 + 合成雷达帧, 端到端跑一遍"像素框 -> 病害记录"。
================================================================
不需要 ROS、不需要真数据, 直接 `python3 仿真验证_定位链.py` 即可跑通并生成:

  - 控制台输出: 病害记录 JSON(里程/环号/时钟方位/3D坐标/尺寸/置信度)
  - docs/定位链demo输出.png: 相机视角 + 3D隧道 + 点云属性图(深度/高度/反射率) 四联图
  - 末尾 PASS/FAIL 断言: 反算的里程/环号/时钟方位/3D坐标 与真值是否一致

场景(世界系, Z上):
  直隧道: 轴线起点 (0,0,1.5), 方向 +x, 起点里程 K1000, 环宽 1.5m, 半径 2.5m
  小车  : 世界位姿 (10, 0, 0), 朝向 +x (正对里程递增方向)
  病害  : 右侧隧道壁 3点钟 位置 P=(17.5, -2.5, 1.5) 的一片"渗漏水"
           (17.5m 刻意避开环边界, 避免环号被毫米级误差翻转)
  雷达  : 顶部 16线/360° 合成帧(含病害处高反射率点)
  检测框: 由病害点投影到图像得到(模拟检测器输出)

预期:  里程≈1017.5m  环号=12  时钟方位=3点钟  3D≈(17.5,-2.5,1.5)
"""
from __future__ import annotations

import numpy as np

from img_px_to_world import (
    DetectionLocalizer, TunnelGeometry, pose_to_T,
    project_to_image, lidar_to_image_attributes, lidar_to_bev_height,
    record_to_json,
)
from 传感器参数 import K, T_CAM_BASE, IMG_SIZE

OUT_PNG = "docs/定位链demo输出.png"

# ---------------------------------------------------------------
# 0. 隧道 + 小车位姿
# ---------------------------------------------------------------
tunnel = TunnelGeometry(
    axis_point=np.array([0.0, 0.0, 1.5]),
    axis_dir=np.array([1.0, 0.0, 0.0]),
    start_mileage=1000.0,
    ring_width=1.5,
    radius=2.5,
    up=np.array([0.0, 0.0, 1.0]),
)
T_WORLD_BASE = pose_to_T(10.0, 0.0, 0.0, yaw=0.0)     # 车在世界(10,0,0), 朝+x

# 真值: 病害中心在右侧隧道壁 3点钟
P_GT = np.array([17.5, -2.5, 1.5])
R_GT = 2.5                                            # 径向=隧道半径(在壁上)

def world_to_base(X):
    """车 yaw=0, 所以 base = world - 车原点。"""
    return X - np.array([10.0, 0.0, 0.0])

# ---------------------------------------------------------------
# 1. 合成雷达帧: 病害区高反射点(密) + 全周隧道壁背景点(疏)
# ---------------------------------------------------------------
def _wall_point(x_w, z_w):
    """隧道壁上 (x_w, z_w) 对应的 y(右侧壁, 半圆)。"""
    dz = z_w - 1.5
    y = -np.sqrt(max(R_GT ** 2 - dz ** 2, 0.0))
    return np.array([x_w, y, z_w])

def synth_frame():
    pts_base, inten = [], []
    # (a) 病害区: 12x12 密网格, 反射率 200 (模拟渗漏水/金属反光)
    xs = np.linspace(17.1, 17.9, 12)
    zs = np.linspace(1.30, 1.70, 12)
    for x in xs:
        for z in zs:
            pts_base.append(world_to_base(_wall_point(x, z)))
            inten.append(200.0)
    # (b) 背景隧道壁: 16线 x 90方位 合成扫描(覆盖 ±15° 仰角, 全周)
    for k in range(16):
        theta = np.deg2rad(-15.0 + 30.0 * k / 15.0)
        for i in range(90):
            phi = np.deg2rad(-180.0 + 360.0 * i / 89.0)
            d = np.array([np.cos(theta) * np.cos(phi),
                          np.cos(theta) * np.sin(phi),
                          np.sin(theta)])
            O = np.array([0.0, 0.0, 0.30])            # 雷达在 base 的位置
            Y0, Z0 = O[1] - 0.0, O[2] - 1.5           # 相对隧道轴
            dy, dz = d[1], d[2]
            a, b, c = dy ** 2 + dz ** 2, 2 * (Y0 * dy + Z0 * dz), Y0 ** 2 + Z0 ** 2 - R_GT ** 2
            disc = b * b - 4 * a * c
            if disc < 0 or a == 0:
                continue
            t = (-b + np.sqrt(disc)) / (2 * a)        # 近侧交点
            if not (0.2 < t < 50):
                continue
            p = O + t * d
            if p[0] < -2:                             # 只要车前方 2m 之后的墙
                continue
            pts_base.append(p)
            inten.append(30.0)
    return np.array(pts_base), np.array(inten)

pts_base, intensity = synth_frame()

# ---------------------------------------------------------------
# 2. "检测器": 把病害点投影到图像, 取外接框作为检测框
# ---------------------------------------------------------------
uv, in_img, z = project_to_image(pts_base, K, T_CAM_BASE, IMG_SIZE)
defect_mask = (intensity > 100) & in_img
du, dv = uv[defect_mask, 0], uv[defect_mask, 1]
bbox = [float(du.min()), float(dv.min()), float(du.max()), float(dv.max())]
detection = [{"class_name": "渗漏水", "bbox": bbox, "confidence": 0.92}]
print("模拟检测框(像素):", [round(v, 1) for v in bbox])

# ---------------------------------------------------------------
# 3. 跑定位链
# ---------------------------------------------------------------
localizer = DetectionLocalizer(K=K, T_cam_base=T_CAM_BASE, tunnel=tunnel, img_size=IMG_SIZE)
record = localizer.locate(detection, pts_base, T_WORLD_BASE, reflectivity=intensity)

print("\n========== 病害定位记录 JSON ==========")
print(record_to_json(record))
defect = record["defects"][0]

# ---------------------------------------------------------------
# 4. 断言: 反算值与真值比对
# ---------------------------------------------------------------
import math
ok = True
err_mileage = abs(defect["里程_m"] - 1017.5)
err_pos = math.dist(defect["3D坐标_m"], P_GT.tolist())
checks = [
    ("里程≈1017.5m", err_mileage < 0.3, f"误差 {err_mileage:.3f}m"),
    ("环号=12", defect["环号"] == 12, f"实际 {defect['环号']}"),
    ("时钟方位=3点钟", defect["时钟方位"] == "3点钟", f"实际 {defect['时钟方位']}"),
    ("径向≈2.5m(在壁上)", abs(defect["径向距离_m"] - R_GT) < 0.3, f"实际 {defect['径向距离_m']}m"),
    ("3D坐标≈真值", err_pos < 0.4, f"偏差 {err_pos:.3f}m"),
]
print("\n========== 闭环验证 ==========")
for name, passed, detail in checks:
    ok &= passed
    print(("  [PASS] " if passed else "  [FAIL] ") + name + "  " + detail)
print("\n  >>> 全部通过, 定位链数学正确 <<<" if ok else "\n  >>> 存在 FAIL, 请检查 <<<")

# ---------------------------------------------------------------
# 5. 可视化(四联图, 供报告"实证材料"用)
# ---------------------------------------------------------------
try:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # 中文字体(报告图里不能出现豆腐块)
    for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
               "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
        if os.path.exists(_f):
            font_manager.fontManager.addfont(_f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=_f).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    # (a) 相机视角: 投影点 + 检测框
    ax = fig.add_subplot(gs[0, 0])
    bg = np.full((IMG_SIZE[1], IMG_SIZE[0]), 30, dtype=np.uint8)
    ax.imshow(bg, cmap="gray", vmin=0, vmax=255)
    m = in_img
    ax.scatter(uv[m, 0], uv[m, 1], s=2, c=intensity[m], cmap="turbo", vmin=0, vmax=255)
    ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                               fill=False, edgecolor="red", lw=2))
    ax.set_title("① 相机视角: 雷达点投影+检测框(红)", fontsize=11)
    ax.set_xlim(0, IMG_SIZE[0]); ax.set_ylim(IMG_SIZE[1], 0)

    # (b) 3D: 隧道 + 车 + 病害(真值红/反算绿)
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    t = np.linspace(0, 2 * np.pi, 60)
    for xc in np.linspace(8, 22, 8):                  # 隧道壁线框
        ax.plot(np.full_like(t, xc), R_GT * np.cos(t), 1.5 + R_GT * np.sin(t),
                color="0.8", lw=0.6)
    ax.scatter(pts_base[:, 0] + 10, pts_base[:, 1], pts_base[:, 2],
               c=intensity, cmap="turbo", vmin=0, vmax=255, s=3)
    ax.scatter(*P_GT, color="red", s=120, marker="^", label="真值 3点钟")
    ax.scatter(*defect["3D坐标_m"], color="green", s=120, marker="o", label="定位链反算")
    ax.scatter(10, 0, 0.3, color="blue", s=80, marker="D", label="小车(雷达)")
    ax.set_xlabel("x(里程方向)"); ax.set_ylabel("y(左)"); ax.set_zlabel("z(上)")
    ax.set_title("② 3D隧道语义映射: 真值 vs 反算", fontsize=11)
    ax.legend(fontsize=8)

    # (c) 点云属性图 -> 2D
    dmap, rmap, hmap = lidar_to_image_attributes(pts_base, T_CAM_BASE, K, IMG_SIZE, intensity)
    for k, (name, arr) in enumerate([("③ 深度图(雷达z)", dmap), ("④ 反射率图", rmap)]):
        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(arr, cmap="turbo", origin="upper")
        ax.set_title(name, fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(hmap, cmap="turbo", origin="upper")
    ax.set_title("⑤ 高度图(世界Z投影到图像)", fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (d) BEV 高度图(俯视, 异物侵限/避障用)
    ax = fig.add_subplot(gs[2, 1])
    bev = lidar_to_bev_height(pts_base, res=0.1, x_range=(6, 14), y_range=(-3, 3))
    im = ax.imshow(bev.T, cmap="turbo", origin="lower",
                   extent=(6, 14, -3, 3), aspect="auto")
    ax.set_title("⑥ BEV俯视高度图(限界/避障)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("空间语义映射与精准定位 · 闭环验证 (相机↔雷达↔隧道)", fontsize=14)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110)
    print(f"\n已保存可视化: {OUT_PNG}")
except Exception as e:
    print(f"\n[可视化跳过] {e}")

raise SystemExit(0 if ok else 1)
