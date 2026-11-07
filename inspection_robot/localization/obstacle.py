# -*- coding: utf-8 -*-
"""
几何避障检测 —— 不依赖分割, 纯点云几何
================================================================
架构图"几何避障检测 🚧": 点云 -> BEV高度图 / 深度跳变 -> {障碍物距离/方位}。

隧道里"正常环境"是隧道壁/顶板/钢轨/扣件, 需要检测的是超出正常几何的危险凸起
(异物入侵等立在轨道内的块状物)。纯几何思路:

    BEV 高度图(车前 x∈[0,8] × 轨道带 |y|<0.6)
      -> 相对轨面凸起: z > RAIL_TOP + 0.07 且 z < 2.0(排除壁/顶板)
      -> scipy 连通域聚类
      -> 每簇 = 一个障碍物 {最近距离, 方位角, 高度, 尺寸, 点数}

约束来源(output_v7 world 实测):
  - 轨道带 |y|<0.6: 异物 equip_intrusion 立在轨道内 y∈[0.30,0.52];
    钢轨在 y=±0.717, 扣件贴轨顶 → 排除在轨道带外, 不误报。
  - 高度下界 0.22m: 轨顶 0.15, 异物顶 0.26~0.46, 阈值取 0.22 全检出且高于轨顶一个台阶。
  - 高度上界 2.0m: 隧道壁 z≈2.5、顶板 z=2.75 不是障碍, 排除。
  - 车体系 z 直接可用(车平放 z≈0, 点云是车体系)。

depth_discontinuity() 作为交叉验证的补充通道: 360° 按方位角 bin 取最近命中点,
与相邻 bin 相比出现近距突变 = 该方位存在凸起。仅供佐证, 不参与主断言。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from localization.img_px_to_world import lidar_to_bev_height  # noqa: E402
from localization.tunnel_params import (  # noqa: E402
    BEV_RES, BEV_X_RANGE, BEV_Y_RANGE, INTRUSION_Z_MAX, RAIL_TOP,
)


@dataclass
class Obstacle:
    """一个候选障碍物(几何避障检测输出)。

    distance    : 车前向最近距离(m, 簇内最小 x)—— 喂给路径决策的"停车/减速"依据
    azimuth_deg : 方位角(°), 相对车体 x 轴, 左正右负(atan2(y,x))
    x, y        : 簇质心(车体系, m)
    z_top       : 簇内最高点(m, 车体系)
    height      : 相对轨面高度 z_top - RAIL_TOP(m)
    size        : (x_extent, y_extent)(m), 簇在 BEV 网格上的包络
    n_cells     : 连通域格数
    n_points    : 簇内原始点云点数
    """
    distance: float
    azimuth_deg: float
    x: float
    y: float
    z_top: float
    height: float
    size: tuple
    n_cells: int
    n_points: int

    def as_dict(self) -> dict:
        return {
            '距离_m': round(self.distance, 3),
            '方位_deg': round(self.azimuth_deg, 2),
            'x_m': round(self.x, 3),
            'y_m': round(self.y, 3),
            '顶高_m': round(self.z_top, 3),
            '相对轨面高_m': round(self.height, 3),
            '尺寸_m': [round(v, 3) for v in self.size],
            '格数': self.n_cells,
            '点数': self.n_points,
        }


def bev_height_map(points_base: np.ndarray, x_range=BEV_X_RANGE,
                   y_range=BEV_Y_RANGE, res: float = BEV_RES) -> np.ndarray:
    """车前 BEV 高度图: (ny,nx) float32, 每格取该格内最高点的车体系 z。"""
    return lidar_to_bev_height(points_base, res=res, x_range=x_range, y_range=y_range)


def _cluster_points(points_base: np.ndarray, grid: np.ndarray,
                    labels: np.ndarray, lab_id: int, res: float,
                    x_range, y_range, z_top_cells) -> tuple[list[int], int]:
    """取连通域 lab_id 对应的原始点下标 + 格内点数。"""
    rows, cols = np.where(labels == lab_id)
    cell_indices = rows * labels.shape[1] + cols
    # 每个 BEV 格索引对应的原始点
    nx, ny = grid.shape[1], grid.shape[0]
    ix = np.clip(((points_base[:, 0] - x_range[0]) / res).astype(int), 0, nx - 1)
    iy = np.clip(((points_base[:, 1] - y_range[0]) / res).astype(int), 0, ny - 1)
    flat_cell = iy * nx + ix
    in_x = (points_base[:, 0] >= x_range[0]) & (points_base[:, 0] < x_range[1])
    in_y = (points_base[:, 1] >= y_range[0]) & (points_base[:, 1] < y_range[1])
    valid = in_x & in_y
    cell_set = set(cell_indices.tolist())
    member = np.fromiter((flat_cell[i] in cell_set for i in range(len(points_base))),
                         dtype=bool, count=len(points_base)) & valid
    idx = np.where(member)[0]
    return idx, len(rows)


def find_obstacles(points_base: np.ndarray,
                   x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE, res: float = BEV_RES,
                   z_min: float = 0.22, z_max: float = INTRUSION_Z_MAX,
                   y_track: float = 0.6) -> list[Obstacle]:
    """主入口: 车体系点云 (N,3) -> 障碍物列表(按最近距离升序)。

    流程:
      1. 只保留车前 + 轨道带 |y|<y_track 内的点(排除轨上扣件 y≈±0.717 与隧道壁)。
      2. BEV 高度图, 取相对轨面凸起窗口 [z_min, z_max](轨顶之上一个台阶, 低于顶板)。
      3. scipy.ndimage.label 连通域 -> 每簇一个 Obstacle: 最近距离/方位/高度/尺寸/点数。
    """
    from scipy import ndimage

    pts = np.asarray(points_base, dtype=float)
    if pts.ndim == 2 and pts.shape[1] >= 3:
        xy = pts[:, :3]
    else:
        xy = pts

    # ---- 1. 轨道带 + 车前粗筛 ----
    m = (xy[:, 0] >= x_range[0]) & (xy[:, 0] < x_range[1]) & (np.abs(xy[:, 1]) < y_track)
    sel = xy[m]
    if len(sel) == 0:
        return []

    # ---- 2. BEV 高度图 + 凸起窗口 ----
    # z_min 是"车体系 z 下界"(=轨顶 0.15 之上一个台阶 0.22), 直接比较 grid;
    # z_max 排除隧道壁/顶板(壁 z≈2.5、顶板 2.75, 不是障碍)。
    grid = bev_height_map(sel, x_range=x_range, y_range=y_range, res=res)
    mask = (grid > z_min) & (grid < z_max)
    if not mask.any():
        return []

    # ---- 3. 连通域聚类 ----
    labels, n_lab = ndimage.label(mask)
    nx, ny = grid.shape[1], grid.shape[0]
    x_c = x_range[0] + (np.arange(nx) + 0.5) * res
    y_c = y_range[0] + (np.arange(ny) + 0.5) * res
    XC, YC = np.meshgrid(x_c, y_c)

    obstacles = []
    for lab in range(1, n_lab + 1):
        rows, cols = np.where(labels == lab)
        xs, ys = XC[rows, cols], YC[rows, cols]
        zs = grid[rows, cols]
        # 最近距离 = 簇内最小 x; 方位 = 质心方位
        dist = float(xs.min())
        cx, cy = float(xs.mean()), float(ys.mean())
        az = float(np.degrees(np.arctan2(cy, cx)))
        # 还原簇内原始点(统计点数 + 更精确的质心/最高点)
        idx, n_cells = _cluster_points(sel, grid, labels, lab, res, x_range, y_range, zs)
        if len(idx) >= 1:
            cpts = sel[idx]
            cx, cy = float(cpts[:, 0].mean()), float(cpts[:, 1].mean())
            az = float(np.degrees(np.arctan2(cy, cx)))
            z_top = float(cpts[:, 2].max())
        else:
            z_top = float(zs.max())
        x_extent = float(xs.max() - xs.min())
        y_extent = float(ys.max() - ys.min())
        obstacles.append(Obstacle(
            distance=dist,
            azimuth_deg=az,
            x=cx, y=cy,
            z_top=z_top,
            height=z_top - RAIL_TOP,
            size=(x_extent, y_extent),
            n_cells=n_cells,
            n_points=len(idx),
        ))

    obstacles.sort(key=lambda o: o.distance)
    return obstacles


def depth_discontinuity(points_base: np.ndarray, n_bins: int = 180,
                        r_max: float = 15.0) -> dict:
    """交叉验证通道: 360° 方位角 bin 内最近命中点距离, 找近距突变。

    对每个方位 bin 取最近点距离 r[bin]; 若 r 明显小于相邻 bin(或小于前方
    壁面期望距离), 该方位存在凸起。返回 {bin_deg: distance} 的跳变集合。
    仅供佐证, 不参与主断言(16 线雷达 + 隧道环形反射噪声大)。
    """
    pts = np.asarray(points_base, dtype=float)
    xy = pts[:, :3] if pts.ndim == 2 else pts
    r = np.linalg.norm(xy[:, :2], axis=1)
    az = np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) % 360.0
    b = np.clip((az / (360.0 / n_bins)).astype(int), 0, n_bins - 1)
    nearest = np.full(n_bins, np.inf)
    for i in range(n_bins):
        sel = r[b == i]
        if len(sel):
            nearest[i] = sel.min()
    nearest = np.where(nearest > r_max, np.nan, nearest)
    # 跳变: 与相邻 bin 中值比, 明显更近
    n_shift = max(2, n_bins // 60)
    out = {}
    for i in range(n_bins):
        if np.isnan(nearest[i]):
            continue
        lo = max(0, i - n_shift)
        hi = min(n_bins, i + n_shift + 1)
        nbrs = nearest[np.arange(lo, i).tolist() + np.arange(i + 1, hi).tolist()]
        nbrs = nbrs[~np.isnan(nbrs)]
        if len(nbrs) == 0:
            continue
        med = np.median(nbrs)
        if nearest[i] < 0.6 * med:
            out[round(i * 360.0 / n_bins, 1)] = round(float(nearest[i]), 2)
    return out


if __name__ == '__main__':
    # 冒烟: 用 output_v7 含异物帧验证能检出障碍
    root = os.path.join(_DIR, 'data_gen', 'output_v7')
    pts = np.load(os.path.join(root, 'clouds', '000000.npy'))
    obs = find_obstacles(pts)
    print(f'帧000000 检出 {len(obs)} 个障碍物:')
    for o in obs:
        print(' ', o.as_dict())
