#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L2 几何避让闭环(离线) —— 车运动学 + 点云检测 + 避让决策 闭环仿真。
用合成世界点云(墙/地/轨 + 红绿蓝黄盒), 车从原点朝 +x 前进, 遇红盒(3,0.8)应绕开不撞。

用法: python scripts/verify_avoidance.py [--dt .1] [--tmax 20] [--yaw-max .6] [--steer-dist 2.6]
  [--corridor 1]   corridor=1 用走廊约束(world 系剔墙 + 目标车道保持), 0=仅几何避让(剔墙不约束)。
"""
import argparse
import os
import sys

import numpy as np

_D = os.path.dirname(os.path.abspath(__file__))
if _D not in sys.path:
    sys.path.insert(0, _D)
from obstacle_avoid_node import avoid_decision, _ramp, _choose_side, AVOID_V_CRUISE, \
    AVOID_STOP_DIST, AVOID_SLOW_DIST, AVOID_STEER_DIST, AVOID_BODY_FRONT, AVOID_STEER_V_MAX
import geometric_obstacle_detect as so
from obstacle_avoid_node import (wallfree,
                                 lane_keep as _lane_keep,
                                 corridor_keep as _corridor_keep)

# ---- 全局几何(车原点, 前向+x, z上; base 系: 地面 z=-0.20, 即 world_z-0.20) ----
_HW = 1.25        # 墙内面 world y 半宽(= 真 gz 内表面), 车原点时即世界 y
_WALL_TIP = 1.2   # 剔墙 world |y| 阈值(略小于内面 1.25) —— 世界系剔墙阈值


def base_cloud(seed=0):
    """真几何(墙内面 ±1.25), base 系。_synth 用 1.35 会掩盖"1.20 残留墙条"问题, 这里复现真墙。"""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    P = []
    def a(v): P.append(v)
    a(_np.c_[rng.uniform(-20, 20, 60000), rng.uniform(-1.55, 1.55, 60000), -0.20 + rng.normal(0, .01, 60000)])
    for s in (-1, 1):
        a(_np.c_[rng.uniform(-20, 20, 9000), s * (_HW + rng.normal(0, .01, 9000)), rng.uniform(-0.20, 2.85, 9000)])
    a(_np.c_[rng.uniform(-20, 20, 9000), rng.uniform(-1.55, 1.55, 9000), 2.70 + rng.normal(0, .01, 9000)])
    for s in (-1, 1):
        a(_np.c_[rng.uniform(-20, 20, 8000), s * (0.717 + rng.normal(0, .008, 8000)), rng.uniform(-0.20, -0.05, 8000)])
    a(_np.c_[rng.uniform(2.6, 3.4, 4000), rng.uniform(0.4, 1.2, 4000), rng.uniform(-0.20, 0.30, 4000)])
    a(_np.c_[rng.uniform(5.7, 6.3, 4000), rng.uniform(-1.1, -0.5, 4000), rng.uniform(-0.20, 0.10, 4000)])
    th = rng.uniform(0, 2 * _np.pi, 4000); r = rng.random(4000) * 0.3
    a(_np.c_[9 + r * _np.cos(th), 0.8 + r * _np.sin(th), rng.uniform(-0.20, 0.80, 4000)])
    a(_np.c_[rng.uniform(11.55, 12.45, 4000), rng.uniform(-1.1, -0.5, 4000), rng.uniform(-0.20, 0.50, 4000)])
    return _np.concatenate(P, 0)


def car_frame(pc, x, y, yaw):
    """世界系点云 -> 车体系(仅平面旋转+z不变)。"""
    if yaw == 0 and (x == 0 and y == 0):
        return pc
    dx = pc[:, 0] - x; dy = pc[:, 1] - y
    ca, sa = np.cos(yaw), np.sin(yaw)
    px = dx * ca + dy * sa
    py = -dx * sa + dy * ca
    return np.c_[px, py, pc[:, 2]]


def detect(pc, corridor):
    """车体系点云 -> 障碍。corridor=1: 先把点云转世界系剔墙(按 world |y|>WALL_TIP)。"""
    pts = pc
    if corridor:
        # 转回世界(车转向时墙 world 恒 |y|=1.25)剔墙, 转回车体系
        pass
    return so.detect_obstacles(pts, z_min=so.Z_MIN, y_wall=3.0, z_ceil=2.05, max_range=18.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--tmax', type=float, default=22.0)
    ap.add_argument('--corridor', type=int, default=1)
    a = ap.parse_args()

    P0 = base_cloud()               # base 系(车原点, 地面 z=-0.20), 墙内面 ±1.25
    x, y, yaw = 0.0, 0.0, 0.0
    dt = a.dt
    steps = int(a.tmax / dt)
    hit_red = False; hit_wall = False; passed_red = False
    red = np.array([3.0, 0.8])
    print(f"L2 避让闭环: corridor={a.corridor} dt={dt} tmax={a.tmax}s  车初(0,0,0) 红盒(3,0.8)")
    print(f"{'t':>5} {'x':>7} {'y':>7} {'yaw':>6} {'state':>12} {'d':>5} {'az':>6} {'obs':>4}")

    for k in range(steps):
        P = car_frame(P0, x, y, yaw)
        if a.corridor:
            # 世界系剔墙, 用本 L2 场景的门限 _WALL_TIP(=内面 1.25 留余量, 保住靠墙的偏置盒)
            # —— 不要用 node 的默认 ymax(=WALL_WORLD_Y=0.95, 那是给"居中盒"剔墙用的, 会切掉近墙盒)
            Pk = wallfree(P, x, y, yaw, ymax=_WALL_TIP)
        else:
            Pk = P                           # 旧: base系剔墙(残留墙条), 复现 bug
        obs = so.detect_obstacles(Pk, max_range=18.0, y_wall=3.0)
        vx, wz, st, fr = avoid_decision(obs, 0.0, x, y, yaw)
        if st == 'STEER_AVOID' and fr is not None:
            wz = _lane_keep(fr, x, y, yaw)       # odom 车道跟踪(世界系, 稳侧防翻)
        vx, wz = _corridor_keep(vx, wz, st, y)   # 走廊兜底(防冲看不见的墙)
        if k % max(1, int(2 / dt)) == 0:
            d = fr.dist if fr else -1
            az = fr.azimuth_deg if fr else 0
            print(f"{k*dt:>5.1f} {x:>7.2f} {y:>7.2f} {np.degrees(yaw):>6.1f} {st:>12} {d:>5.2f} {az:>6.1f} {len(obs):>4}")
        # 运动学
        x += vx * np.cos(yaw) * dt; y += vx * np.sin(yaw) * dt; yaw += wz * dt
        # 碰撞/绕行判定
        dr = np.hypot(x - red[0], y - red[1])
        if dr < 0.5: hit_red = True
        if x > red[0] + 1.0: passed_red = True
        if abs(y) > _HW - 0.20: hit_wall = True
        if hit_red or hit_wall:
            break
    print("\n===== 结果 =====")
    print(f"绕开红盒(车前过 x>{red[0]+1:.0f}): {'是' if passed_red else '否'}")
    print(f"撞红盒(<0.5m):            {'是' if hit_red else '否'}")
    print(f"撞墙(|y|>{_HW-0.2:.2f}):      {'是' if hit_wall else '否'}")
    print(f"终点: x={x:.2f} y={y:.2f} yaw={np.degrees(yaw):.1f}°")
    ok = passed_red and not hit_red and not hit_wall
    print('>>> L2' + ('通过: 成功绕开红盒且不撞墙' if ok else '未通过'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
