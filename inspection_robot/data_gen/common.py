# -*- coding: utf-8 -*-
"""
配对数据生成 · 标定与投影公共模块
================================================================
单数据源原则: 优先从 pro/2/空间语义映射(传感器参数.py + img_px_to_world.py)导入,
保持仿真生成的属性图/真值框与定位链完全一致; 换机器/两项目不同目录时回退到内嵌默认值。

依赖: 仅 numpy (以及可选 pro/2 库)。
坐标系约定与定位链一致:
    base_link : x前/y左/z上
    camera    : z前/x右/y下 (标准针孔, 深度沿 +z)
    lidar     : 雷达系 -> 车体系 外参 T_lidar_base (默认 (0,0,0.30), rpy=0)
"""
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------
# 0. 尝试从 pro/2 空间语义映射 导入单一数据源
# ---------------------------------------------------------------
_LOCALIZE_PATHS = [
    os.path.expanduser("~/workspace/pro/2/空间语义映射"),
]
_have_localize = False
for _p in _LOCALIZE_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        try:
            from img_px_to_world import (  # noqa: F401
                build_K_from_hfov, gz_camera_extrinsic, pose_to_T,
                project_to_image, lidar_to_image_attributes,
            )
            from 传感器参数 import K as _K, T_CAM_BASE as _T_CAM_BASE, \
                T_LIDAR_BASE as _T_LIDAR_BASE, CAMERA
            K = _K
            T_CAM_BASE = _T_CAM_BASE
            T_LIDAR_BASE = _T_LIDAR_BASE
            IMG_SIZE = (CAMERA["width"], CAMERA["height"])
            _have_localize = True
            break
        except Exception:                                    # 路径在但库不可用
            sys.path.remove(_p)

if not _have_localize:
    # ---------- 回退: 内嵌默认(与 传感器参数.py / xacro 一致) ----------
    def build_K_from_hfov(hfov_rad: float, width: int, height: int) -> np.ndarray:
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        return np.array([[fx, 0.0, width / 2.0],
                         [0.0, fx, height / 2.0],
                         [0.0, 0.0, 1.0]])

    def pose_to_T(px, py, pz, roll=0., pitch=0., yaw=0.) -> np.ndarray:
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [px, py, pz]
        return T

    def gz_camera_extrinsic(xyz=(0.22, 0.0, 0.10), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
        R_opt_to_cam = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        T_install = pose_to_T(*xyz, *rpy)
        T_opt = np.eye(4); T_opt[:3, :3] = R_opt_to_cam
        return T_install @ T_opt

    def project_to_image(X_base_pts, K, T_cam_base, img_size=(1280, 720)):
        X_cam = (np.linalg.inv(T_cam_base) @
                 np.concatenate([X_base_pts, np.ones((len(X_base_pts), 1))], axis=1).T).T
        X_cam = X_cam[:, :3]
        z = X_cam[:, 2]
        valid_f = z > 1e-3
        uv = np.full((len(X_base_pts), 2), np.nan)
        uv[:, 0] = K[0, 0] * X_cam[:, 0] / np.where(z, z, 1) + K[0, 2]
        uv[:, 1] = K[1, 1] * X_cam[:, 1] / np.where(z, z, 1) + K[1, 2]
        W, H = img_size
        in_img = valid_f & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        return uv, in_img, z

    def lidar_to_image_attributes(points_base, T_cam_base, K, img_size=(1280, 720),
                                  reflectivity=None, z_world=None):
        H, W = img_size[1], img_size[0]
        uv, in_img, z = project_to_image(points_base, K, T_cam_base, img_size)
        depth_map = np.full((H, W), np.nan, dtype=np.float32)
        reflec_map = np.full((H, W), np.nan, dtype=np.float32)
        height_map = np.full((H, W), np.nan, dtype=np.float32)
        if in_img.sum() == 0:
            return depth_map, reflec_map, height_map
        u = np.clip(np.round(uv[in_img, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(uv[in_img, 1]).astype(int), 0, H - 1)
        depth_map[v, u] = z[in_img].astype(np.float32)
        if reflectivity is not None:
            reflec_map[v, u] = reflectivity[in_img].astype(np.float32)
        zsrc = z_world if z_world is not None else points_base[:, 2]
        height_map[v, u] = zsrc[in_img].astype(np.float32)
        return depth_map, reflec_map, height_map

    K = build_K_from_hfov(1.047, 1280, 720)
    T_CAM_BASE = gz_camera_extrinsic()
    T_LIDAR_BASE = pose_to_T(0.0, 0.0, 0.30)
    IMG_SIZE = (1280, 720)


def quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """四元数(w,x,y,z) -> 旋转矩阵 (与定位链一致)。"""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def odom_to_T_world_base(pos, quat) -> np.ndarray:
    """odom 位姿(xyz + xyzw) -> T_world_base。"""
    T = np.eye(4)
    T[:3, :3] = quat_to_R(quat[3], quat[0], quat[1], quat[2])
    T[:3, 3] = pos
    return T


def build_gt_boxes(corners_world: np.ndarray, T_world_base: np.ndarray,
                   K, T_cam_base, img_size=(1280, 720),
                   require_all_in: bool = True):
    """把病害矩形在世界系下的 4 角投影到图像, 返回 YOLO 归一化框 [cx,cy,w,h] 或 None。

    仅当中心在图像内且(可选)四角都在图像内时给框, 保证自动标注干净。
    corners_world: (4,3) 世界系角点(矩形).
    """
    W, H = img_size
    # 世界 -> 车体
    T_base_world = np.linalg.inv(T_world_base)
    Xw = np.concatenate([corners_world, np.ones((4, 1))], axis=1).T
    Xb = (T_base_world @ Xw).T[:, :3]
    uv, in_img, z = project_to_image(Xb, K, T_cam_base, img_size)

    # 中心点同样要进相机系再投影(不能拿车体系坐标直接当相机坐标)
    center = corners_world.mean(axis=0)
    Xc_base = (T_base_world @ np.concatenate([center, [1.0]]))[:3]
    Xc_cam = (np.linalg.inv(T_cam_base) @
              np.concatenate([Xc_base, [1.0]]))[:3]
    zc = Xc_cam[2]
    if not (0.3 <= zc <= 40.0):               # 深度合理范围(相机系 z)
        return None
    uc = K[0, 0] * Xc_cam[0] / zc + K[0, 2]
    vc = K[1, 1] * Xc_cam[1] / zc + K[1, 2]
    if not (0 <= uc < W and 0 <= vc < H):     # 中心必须在画面内
        return None
    ok = in_img.all() if require_all_in else in_img.any()
    if not ok:
        return None
    u_min, u_max = uv[:, 0].min(), uv[:, 0].max()
    v_min, v_max = uv[:, 1].min(), uv[:, 1].max()
    ww, hh = u_max - u_min, v_max - v_min
    # 过小/侧对相机的退化框(窄缝)不是有效标注目标, 丢弃
    # 12px: 远处病害(10m,0.3m)约33px, 只有侧视窄缝会低于此, 不会被误删
    if ww < 12 or hh < 12:
        return None
    # 宽高比: 正面病害投影约 0.5~2.5, 侧视窄缝会 <0.15
    if min(ww, hh) / max(ww, hh) < 0.15:
        return None
    cx = (u_min + u_max) / 2 / W
    cy = (v_min + v_max) / 2 / H
    w = (u_max - u_min) / W
    h = (v_max - v_min) / H
    return [round(cx, 5), round(cy, 5), round(w, 5), round(h, 5)]


def fill_nearest(map2d: np.ndarray, valid=None) -> np.ndarray:
    """最近邻空洞填充: 每个 NaN 像素取最近有效像素的值 (编码器输入需要稠密图)。

    16 线属性图天然有 ~99% 空洞, 直接喂 CNN 会被空洞吃掉感受野,
    用 scipy 的 distance_transform_edt 做最近邻填充; scipy 不可用时回退到
    迭代膨胀 (4 邻域, 最坏走斜边, 速度慢但结果近似)。

    用法: build_attrmaps.py 里对 depth/reflec/height 三张图分别调用。
    返回与输入同形状的稠密图 (所有元素有限)。
    """
    map2d = np.asarray(map2d, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(map2d)
    if valid.all():
        return map2d.copy()

    try:
        from scipy import ndimage
        # 返回每个像素到最近有效像素的索引
        idx = ndimage.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True)
        filled = map2d[tuple(idx)]
        return np.where(valid, map2d, filled)
    except ImportError:
        # numpy 回退: 逐轮把 4 邻域里"已有有效值"的扩展一格
        out = map2d.copy()
        mask = valid.copy()
        H, W = mask.shape
        k = 0
        while not mask.all() and k < max(H, W) * 2:   # 斜线最长 ~= 宽高和, 循环一定终止
            grown = mask.copy()
            grown[1:, :] |= mask[:-1, :]
            grown[:-1, :] |= mask[1:, :]
            grown[:, 1:] |= mask[:, :-1]
            grown[:, :-1] |= mask[:, 1:]
            new = grown & ~mask
            if not new.any():                          # 没有可扩展的像素, 强制终止
                break
            # 新像素取四邻域有效值的均值 (最近邻的近似)
            s = np.zeros_like(out); c = np.zeros_like(mask, dtype=np.int16)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                sy = slice(max(0, dy), H + dy) if dy >= 0 else slice(dy, None)
                sx = slice(max(0, dx), W + dx) if dx >= 0 else slice(dx, None)
                ty = slice(max(0, -dy), H - dy) if dy <= 0 else slice(dy, H)
                tx = slice(max(0, -dx), W - dx) if dx <= 0 else slice(dx, W)
                ok = np.isfinite(out[sy, sx])
                s[sy, sx][ok] += out[sy, sx][ok]
                c[sy, sx][ok] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                out[new] = np.where(c[new] > 0, s[new] / np.maximum(c[new], 1), out[new])
            mask |= new
            k += 1
        return out


if __name__ == "__main__":
    print("K=\n", np.round(K, 2))
    print("T_cam_base=\n", np.round(T_CAM_BASE, 4))
    print("T_lidar_base=\n", np.round(T_LIDAR_BASE, 4))
    print("IMG_SIZE=", IMG_SIZE, " 源:", "pro/2 定位链" if _have_localize else "内嵌默认")
