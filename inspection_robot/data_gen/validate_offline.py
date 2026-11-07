# -*- coding: utf-8 -*-
"""
离线验证: 自动标注几何投影 + 属性图生成 (不需要启动仿真)
================================================================
1) 对沿隧道一串机器人位姿, 用 defects.json 投影真值框, 检查框在画面内、数量合理
2) 构造一帧合成墙面点云, 生成深度/反射率/高度属性图, 检查空洞率/深度合理性

用法: python3 data_gen/validate_offline.py
"""
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_gen.common import (  # noqa: E402
    K, T_CAM_BASE, T_LIDAR_BASE, IMG_SIZE,
    odom_to_T_world_base, build_gt_boxes, project_to_image, lidar_to_image_attributes,
    fill_nearest,
)

W, H = IMG_SIZE
here = os.path.dirname(os.path.abspath(__file__))


def test_gt_boxes():
    with open(os.path.join(here, "defects.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    defects = manifest["defects"]

    total_gt = 0
    # 机器人沿 x 从 0.5 到 15, 步长 0.4, 直线行驶
    for x in np.arange(0.5, 15.1, 0.4):
        T_wb = odom_to_T_world_base([float(x), 0.0, 0.18], [0, 0, 0, 1])
        for d in defects:
            box = build_gt_boxes(np.array(d["corners_world"]), T_wb, K, T_CAM_BASE, IMG_SIZE)
            if box is not None:
                cx, cy, w, h = box
                # 归一化框必须在 [0,1] 且宽高在合理范围
                assert 0 <= cx <= 1 and 0 <= cy <= 1, f"框越界: {box}"
                # build_gt_boxes 已保证 >=12px, 这里只抓大错(负数/0)
                assert w >= 0.004 and h >= 0.004, f"框太小: {box}"
                total_gt += 1

    print(f"[GT] 37 个前向位姿 x {len(defects)} 病害, 共 {total_gt} 个有效真值框")
    # 实际录制往返两程 + 0.5s 采样, 覆盖会翻几倍; 这里只要求前向扫描不是坏的
    assert total_gt > len(defects) * 4, "真值框数量过少, 投影/布置有问题"
    print(f"[GT] ✅ 投影数学正常, 前向扫描每病害平均 {total_gt / len(defects):.1f} 个视角")


def test_front_defect_center():
    """缺陷在正前方墙上时, 投影应落在画面中上部(正对相机)。"""
    T_wb = odom_to_T_world_base([1.5, 0.0, 0.18], [0, 0, 0, 1])  # 车在 x=1.5, 病害在 x=5 前方
    d = {"corners_world": [[5.0, 1.24, 0.7], [5.4, 1.24, 0.7],
                           [5.4, 1.24, 1.05], [5.0, 1.24, 1.05]]}  # 左壁 x=5 的病害
    box = build_gt_boxes(np.array(d["corners_world"]), T_wb, K, T_CAM_BASE, IMG_SIZE)
    assert box is not None, "正前方病害应可见"
    cx_px, cy_px = box[0] * W, box[1] * H
    print(f"[GT] 正前方左墙病害投影中心 = ({cx_px:.0f}, {cy_px:.0f})px (左墙应偏左, 此结果正确)")
    assert 50 < cx_px < 1000, f"横向中心异常: {cx_px}"
    assert 50 < cy_px < 600, f"纵向中心异常: {cy_px}"
    assert cx_px < 640, "左墙病害却偏右, 外参方向可能反了"
    print("[GT] ✅ 正前方投影合理(左墙→画面左侧)")


def test_attribute_maps():
    """合成一帧左墙点云, 检查属性图生成与空洞率。"""
    # 模拟 16 线雷达打在左墙 (y=1.25), 车在 (0,0), 墙面 x∈[2,10], z∈[0,1.5]
    rng = np.random.default_rng(0)
    x = np.linspace(2.0, 10.0, 400)
    z_rings = np.linspace(0.2, 1.5, 16)               # 16 根垂直环
    pts = []
    for z in z_rings:
        for xi in x:
            # 加一点噪声, 更像真实扫描
            pts.append([xi + rng.normal(0, 0.01), 1.25 + rng.normal(0, 0.005), z])
    pts = np.array(pts)
    # 转换到车体系: 车在 (0,0,0.18), 雷达在 base z=0.30
    base_pts = pts.copy()
    base_pts[:, 0] -= 0.0
    base_pts[:, 1] -= 0.0
    base_pts[:, 2] -= 0.18 - 0.30                       # world z -> base z (雷达在 base 上方0.30)
    reflec = np.full(len(pts), 0.5)
    depth, reflec_map, height = lidar_to_image_attributes(
        base_pts, T_CAM_BASE, K, IMG_SIZE, reflec)

    valid = np.isfinite(depth)
    hole = 1.0 - valid.mean()
    print(f"[ATTR] 深度图有效像素 {valid.sum()}/{H * W}, 空洞率 {hole:.1%} (16线天生稀疏, 正常)")
    print(f"[ATTR] 深度范围 [{np.nanmin(depth):.2f}, {np.nanmax(depth):.2f}]m (相机系深度=沿视轴距离)")
    # 相机深度 = 前方 x 距离, 墙在 x∈[2,10], 应为 1.5~12m
    assert np.nanmin(depth) > 1.0 and np.nanmax(depth) < 12.0, "深度值不合理"
    assert hole < 0.999, "空洞率过高, 投影有问题"
    print("[ATTR] ✅ 属性图生成正常")

    # 补密(最近邻)应显著提高覆盖率 —— 编码器输入需要稠密图
    filled = fill_nearest(depth)
    cov = np.isfinite(filled).mean()
    print(f"[ATTR] 最近邻补密后覆盖率 {cov:.1%}")
    assert cov > 0.995, "补密后仍大量空洞, 补密实现有问题"
    print("[ATTR] ✅ 最近邻补密正常")


def test_cloud_to_base():
    """雷达点 (0,0,0.30 上方) -> 车体系 z 应 +0.30。"""
    x = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, -0.5]])
    X4 = np.concatenate([x, np.ones((2, 1))], axis=1).T
    b = (T_LIDAR_BASE @ X4).T[:, :3]
    assert np.allclose(b, x + np.array([0, 0, 0.30])), "雷达->车体变换错误"
    print("[LIDAR] ✅ 点云车体系变换正确 (z+0.30)")


if __name__ == "__main__":
    print(f"校验标定: fx={K[0,0]:.1f} img={IMG_SIZE}")
    test_cloud_to_base()
    test_front_defect_center()
    test_gt_boxes()
    test_attribute_maps()
    print("\n🎉 全部离线验证通过")
