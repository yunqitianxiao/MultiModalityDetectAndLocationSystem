# -*- coding: utf-8 -*-
"""
几何避障检测器(架构图⑤) —— 模型无关点云几何目标检测
================================================================
从 16 线雷达点云直接"算"障碍, 不学、无标注。给定车体系点云 (x前, y左, z上, 地z=0),
剔除 地面/隧道墙/顶/自身底盘, 剩余做体素化欧式聚类, 每个簇 -> 一个障碍框 + 距离/方位。

与融合检测(学的)互补: 这是硬几何安全层 —— 点云里有凸起/异物就避让, 不依赖检测器交不交框。

流程:
  1) 掩膜剔除: 地面(z < z_floor+ε) / 墙(|y|>y_wall) / 顶(z>z_ceil) / 底盘(近原点小半径)。
  2) 体素降采样(下采样, 抗多线噪声) + 体素网格 BFS 连通域聚类(纯 numpy, 无 scipy/sklearn)。
  3) 每簇 -> 均值中心 / AABB 框 / 点数; 距离=√(x²+y²), 方位=atan2(y,x)(度, 左正右负)。
  4) 过滤小簇(点数<min_pts, 为噪声) 与更远簇(range)。返回障碍列表。

用法:
  from geometric_obstacle_detect import detect_obstacles, Obstacle
  obs = detect_obstacles(points_base)            # points_base: (N,3) float, 米
  python geometric_obstacle_detect.py --selftest # 合成世界点云自测
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class Obstacle:
    """一个几何障碍。cx,cy 为簇均值(米); x1..z2 为 AABB; 其余为车体几何。"""
    idx: int
    cx: float
    cy: float
    cz: float
    dist: float          # 距离 = √(cx²+cy²)
    azimuth_deg: float   # atan2(cy,cx), 度, 左正右负
    npts: int
    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float


# ---------------- 默认几何阈值(米) ----------------
Z_FLOOR = -0.20        # base_link 系: 地面 z (轮底贴地 z=0 世界; base 原点离地 0.20, 见 xacro)
Z_FLOOR_EPS = 0.10     # 地面带厚度: z < Z_FLOOR+EPS 剔
Y_WALL = 1.20          # |y|>此剔墙(世界墙内表面 y=±1.25; 降到 1.20 剔净残留墙点长条误报)
Z_CEIL = 2.05          # z > 此剔顶(世界顶 z=2.9, 余量给线)
Z_MIN = 0.0            # z < 此剔钢轨/低背景(钢结构): 钢轨顶 world z=0.15 -> 车体系 -0.05,
                       # 盒子/异物顶可到 0.30+; 剔轨顶以下即剔钢轨、留住障碍。架构图⑤关键开关。
ROBOT_CLEAR_RAD = 0.35 # 原点附近底盘自剔半径(自身车体/近距噪声)
ROBOT_CLEAR_XMAX = 0.30
VOXEL = 0.08           # 体素边长; 也是单障碍内点连通半径
MIN_PTS = 20           # 簇最小点数(过滤噪声碎点)
MAX_RANGE = 25.0       # 只保留此距离内的障碍
MAX_CLUSTERS = 16


def mask_points(p: np.ndarray, z_floor=Z_FLOOR, z_floor_eps=Z_FLOOR_EPS,
                y_wall=Y_WALL, z_ceil=Z_CEIL, z_min=Z_MIN,
                robot_r=ROBOT_CLEAR_RAD):
    """车体系点云 -> 剔掉 地面/钢轨低结构/墙/顶/底盘 后的剩余点(索引)。"""
    keep = np.ones(len(p), bool)
    keep &= p[:, 2] > z_floor + z_floor_eps          # 地面之上
    keep &= p[:, 2] > z_min                          # 钢轨顶以下(轨/低背景)剔; 障碍存留
    keep &= np.abs(p[:, 1]) < y_wall                 # 墙以内
    keep &= p[:, 2] < z_ceil                         # 顶以下
    keep &= ~((np.hypot(p[:, 0], p[:, 1]) < robot_r) & (p[:, 0] < ROBOT_CLEAR_XMAX))  # 底盘
    return keep


def voxel_cluster(p: np.ndarray, voxel=VOXEL):
    """体素化 + 连胞 BFS 聚类(纯 numpy)。p 为已掩膜、下采样后的点。返回簇索引数组(同长, -1=未用)。"""
    if len(p) == 0:
        return np.zeros(0, np.int64)
    keys = np.round(p / voxel).astype(np.int64)
    # 唯一体素 -> 索引
    ukeys, inverse = np.unique(keys, axis=0, return_inverse=True)
    # 26 邻域连通(体素间差 ∈ {-1,0,1}^3 非全0)
    n = len(ukeys)
    parent = np.arange(n)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    # 对每个体素用哈希查 26 邻域
    key_map = {tuple(k): i for i, k in enumerate(ukeys)}
    dirs = [d for d in np.array(np.meshgrid(*([[-1, 0, 1]] * 3), indexing='ij')).reshape(3, -1).T
            if any(d != 0 for d in d)]
    for i, k in enumerate(ukeys):
        for d in dirs:
            j = key_map.get(tuple(k + d))
            if j is not None and j > i:
                union(i, j)
    # 体素根 -> 体素内点数; 簇 id 按包含的体素
    labels = np.array([find(i) for i in range(n)])
    lab = labels[inverse]
    # 把每个体素按根 id 归一到连续 0..K-1
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def downsample(p: np.ndarray, voxel=VOXEL * 1.5):
    """体素降采样: 每体素取一个代表点(均值)。抗多线点密度不均。"""
    if len(p) == 0:
        return p
    keys = np.round(p / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse)
    inv_sort = inverse[order]
    cuts = np.flatnonzero(np.r_[True, inv_sort[1:] != inv_sort[:-1]])
    centers = np.add.reduceat(p[order], cuts, axis=0) / np.diff(np.r_[cuts, len(p)])[:, None]
    return centers


def detect_obstacles(points_base: np.ndarray, *, z_floor=Z_FLOOR, z_floor_eps=Z_FLOOR_EPS,
                     y_wall=Y_WALL, z_ceil=Z_CEIL, z_min=Z_MIN, robot_r=ROBOT_CLEAR_RAD,
                     voxel=VOXEL, min_pts=MIN_PTS, max_range=MAX_RANGE,
                     max_clusters=MAX_CLUSTERS) -> list[Obstacle]:
    """车体系点云 -> 几何障碍列表(已按距离升序)。points_base 至少 (N,3)。"""
    p = np.asarray(points_base, np.float32).reshape(-1, 3)
    p = p[np.isfinite(p).all(1)]              # 过滤 NaN/Inf
    p = p[np.abs(p).sum(1) < 1e4]             # 过滤远爆点(异常大坐标)
    keep = mask_points(p, z_floor, z_floor_eps, y_wall, z_ceil, z_min, robot_r)
    pc = p[keep]
    pc = downsample(pc, voxel)
    if len(pc) < 4:
        return []
    lab = voxel_cluster(pc, voxel)
    obstacles = []
    for ci in range(lab.max() + 1):
        sel = pc[lab == ci]
        if len(sel) < min_pts:
            continue
        cx, cy, cz = float(sel[:, 0].mean()), float(sel[:, 1].mean()), float(sel[:, 2].mean())
        dist = float(np.hypot(cx, cy))
        if dist > max_range:
            continue
        az = float(np.degrees(np.arctan2(cy, cx)))
        obstacles.append(Obstacle(
            idx=ci, cx=cx, cy=cy, cz=cz, dist=dist, azimuth_deg=az, npts=int(len(sel)),
            x1=float(sel[:, 0].min()), y1=float(sel[:, 1].min()), z1=float(sel[:, 2].min()),
            x2=float(sel[:, 0].max()), y2=float(sel[:, 1].max()), z2=float(sel[:, 2].max())))
    obstacles.sort(key=lambda o: o.dist)
    return obstacles[:max_clusters]


# ---------------- 合成世界自测 ----------------
def _synthetic_world(seed=0):
    """合成车体系点云: 地/墙/顶 + red_box/green_box/blue_cylinder。帧=x前/y左/z上, 地z=0。
    障碍在世界 pose 已知, 机器人原点在 x=0。"""
    rng = np.random.default_rng(seed)
    pts = []
    def add(pts_list):
        pts.append(pts_list)
    # 地面: x∈[-5,15], y∈[-1.4,1.4], z≈0
    add(np.c_[rng.uniform(-5, 15, 20000), rng.uniform(-1.4, 1.4, 20000),
              rng.normal(0, 0.015, 20000)])
    # 左右墙 |y|=1.4, z∈[0,2.9]
    for s in (-1, 1):
        add(np.c_[rng.uniform(-5, 15, 5000), s * (1.4 + rng.normal(0, 0.01, 5000)),
                  rng.uniform(0, 2.9, 5000)])
    # 顶 z=2.9
    add(np.c_[rng.uniform(-5, 15, 5000), rng.uniform(-1.4, 1.4, 5000),
              2.9 + rng.normal(0, 0.01, 5000)])
    # red_box: (3, 0.6, z0.12..0.5) 0.8×0.8  (z 从 0.12 抬高, 避 z_floor_eps=0.10 地面带)
    add(np.c_[rng.uniform(3 - 0.4, 3 + 0.4, 1500), rng.uniform(0.6 - 0.4, 0.6 + 0.4, 1500),
              rng.uniform(0.12, 0.5, 1500)])
    # green_box: (6, -0.6, z0.12..0.3) 0.6×0.6
    add(np.c_[rng.uniform(6 - 0.3, 6 + 0.3, 1200), rng.uniform(-0.6 - 0.3, -0.6 + 0.3, 1200),
              rng.uniform(0.12, 0.3, 1200)])
    # blue_cylinder: (9, 0.5, r0.3 z0.12..1)
    th = rng.uniform(0, 2 * np.pi, 1500); r = rng.random(1500) * 0.3
    add(np.c_[9 + r * np.cos(th), 0.5 + r * np.sin(th), rng.uniform(0.12, 1.0, 1500)])
    return np.concatenate(pts, 0)


def selftest(verbose=True):
    p = _synthetic_world()
    # 自测帧: 地面 z=0(与 base_link 默认 Z_FLOOR=-0.20 不同), 于是显式传 z_floor=0.0
    obs = detect_obstacles(p, z_floor=0.0)
    exp = [
        {'name': 'red_box',    'x': 3.0, 'y': 0.6},
        {'name': 'green_box',  'x': 6.0, 'y': -0.6},
        {'name': 'blue_cyl',   'x': 9.0, 'y': 0.5},
    ]
    got = [(o.cx, o.cy) for o in obs]
    if verbose:
        print(f"[selftest] 掩膜+聚类后障碍数 = {len(obs)} (期望 3)")
        for o in obs:
            print(f"  障碍#{o.idx}  center=({o.cx:+.2f},{o.cy:+.2f},{o.cz:.2f}) "
                  f"dist={o.dist:.2f}m az={o.azimuth_deg:+.1f}° npts={o.npts}")
    ok = True
    for k, e in enumerate(exp):
        if k >= len(obs):
            ok = False; print(f"  [FAIL] 未检出 {e['name']} (期望 (x{e['x']},y{e['y']}))"); continue
        dx, dy = abs(obs[k].cx - e['x']), abs(obs[k].cy - e['y'])
        good = dx < 0.45 and dy < 0.45
        print(f"  {'OK ' if good else 'FAIL'} {e['name']}: 检测({obs[k].cx:+.2f},{obs[k].cy:+.2f}) "
              f"vs 期望({e['x']:+.2f},{e['y']:+.2f}) Δ=({dx:.2f},{dy:.2f})")
        ok &= good
    print('>>> 自测' + ('全通过' if ok else '有失败'))
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--pcd', default=None, help='留待: 读真实点云(PCD)')
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    print('用法: python geometric_obstacle_detect.py --selftest')
