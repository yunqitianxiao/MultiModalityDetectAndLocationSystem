#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""服务器实时标定: rclpy 订阅 /lidar/points 一帧(spin_once 保险退出) -> 重建点云
-> 转 base(z+0.30) -> detect_obstacles -> avoid_decision。用 /usr/bin/python3(rclpy+numpy)。"""
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

_D = os.path.dirname(os.path.abspath(__file__))
for _p in (_D, os.path.join(os.path.dirname(_D), 'localization')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geometric_obstacle_detect as so          # noqa: E402
from obstacle_avoid_node import avoid_decision  # noqa: E402

LIDAR_Z = 0.30   # lidar_link 相对 base_link z 偏移: base = lidar + z_offset


def decode(m):
    step = m.point_step
    arr = np.frombuffer(m.data, dtype=np.uint8).reshape(-1, step)
    off = {f.name: f.offset for f in m.fields}
    def fld(n):
        b = arr[:, off[n]:off[n] + 4].copy()
        return b.view(np.float32).reshape(-1)   # (N,) 而非 (N,1), 修 wallfree/stack
    return fld('x'), fld('y'), fld('z')


def main():
    rclpy.init()
    n = Node('probe_detect')
    got = {'m': None}
    n.create_subscription(PointCloud2, '/lidar/points',
                          lambda m: got.__setitem__('m', m), 10)
    print('[probe] 订阅 /lidar/points, 等一帧(≤25s)...', flush=True)
    rclpy.spin_once(n, timeout_sec=25.0)
    m = got['m']
    if m is None:
        print('NO-MSG: 25s 没收到帧')
        n.destroy_node(); rclpy.shutdown(); return 1
    print(f"frame_id={m.header.frame_id}  N={m.width * m.height}  point_step={m.point_step}")
    x, y, z = decode(m)
    for nm, v in (('x', x), ('y', y), ('z', z)):
        vv = v[np.isfinite(v)]
        print(f"  {nm}: min={np.nanmin(vv):+.2f} max={np.nanmax(vv):+.2f} mean={np.nanmean(vv):+.2f}"
              f"  >0:{int((v>0).sum())} >3:{int((v>3).sum())} in[0,0.6]:{int(((v>=0)&(v<=0.6)).sum())}")
    base = np.stack([x, y, z + LIDAR_Z], 1)
    obs = so.detect_obstacles(base, z_min=so.Z_MIN, max_range=18.0)
    print(f"\n[几何避障] 检出 {len(obs)} 个障碍(lidar->base 后):")
    for o in obs:
        print(f"  # dist={o.dist:.2f}m az={o.azimuth_deg:+.1f}° pos=({o.cx:+.2f},{o.cy:+.2f}) "
              f"y=[{o.y1:+.2f},{o.y2:+.2f}] x跨度={o.x2-o.x1:.2f}m z={o.z1:.2f}..{o.z2:.2f} npts={o.npts}")
    import obstacle_avoid_node as AN
    for o in obs:
        s, g = AN._choose_side(o)
        print(f"    _choose_side(steer={s:+.0f}, gap={g:.2f})  y1={o.y1:+.2f} y2={o.y2:+.2f} az={o.azimuth_deg:+.1f}")
    vx, wz, st, f2 = avoid_decision(obs)         # 传全部障碍(节点用同接口)
    if f2 is not None:
        print(f"[决策] 全障碍->主目标 d={f2.dist:.2f} az={f2.azimuth_deg:+.1f}° -> {st} v={vx:.2f} wz={wz:+.2f}")
    else:
        print(f"[决策] 无前方障碍(az<80) -> {st} 巡航 v={vx:.2f}")
    n.destroy_node(); rclpy.shutdown(); return 0


if __name__ == '__main__':
    sys.exit(main())
