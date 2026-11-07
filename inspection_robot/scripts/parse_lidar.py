#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析 ros2 topic echo --once 抓下的 PointCloud2(JSON), 标定 + 跑几何避障。
读 /tmp/one.json -> 重建 (N,3) lidar 系点云 -> 转 base(z+0.30) -> detect_obstacles -> avoid_decision。"""
import json
import os
import sys

import numpy as np

_D = os.path.dirname(os.path.abspath(__file__))
for _p in (_D, os.path.join(os.path.dirname(_D), 'localization')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geometric_obstacle_detect as so          # noqa: E402
from obstacle_avoid_node import avoid_decision  # noqa: E402

LIDAR_Z = 0.30   # lidar_link 相对 base_link 的 z 偏移(base 系 = lidar 系 z + 0.30)


def main(path='/tmp/one.json'):
    j = json.load(open(path))
    print(f"frame_id = {j['header']['frame_id']}   height={j['height']} width={j['width']} "
          f"point_step={j['point_step']}")
    fields = {f['name']: f['offset'] for f in j['fields']}
    ps = j['point_step']
    raw = np.asarray(j['data'], dtype=np.uint8).reshape(-1, ps)

    def fld(n):
        b = raw[:, fields[n]:fields[n] + 4].copy()
        return b.view(np.float32)
    x, y, z = fld('x'), fld('y'), fld('z')
    n = len(x)
    print(f"点数 N={n}")
    for nm, v in (('x', x), ('y', y), ('z', z)):
        vv = v[np.isfinite(v)]
        print(f"  {nm}: min={np.nanmin(vv):+.2f} max={np.nanmax(vv):+.2f} mean={np.nanmean(vv):+.2f}"
              f"   | >0:{int((v>0).sum())} >3:{int((v>3).sum())} in[0,0.6]:{int(((v>=0)&(v<=0.6)).sum())}")
    base = np.stack([x, y, z + LIDAR_Z], 1)
    obs = so.detect_obstacles(base, z_min=so.Z_MIN, max_range=18.0)
    print(f"\n[几何避障] 检测到 {len(obs)} 个障碍(lidar->base 后):")
    for o in obs:
        print(f"  # dist={o.dist:.2f}m az={o.azimuth_deg:+.1f}° pos=({o.cx:+.2f},{o.cy:+.2f}) "
              f"x跨度={o.x2-o.x1:.2f}m z1..z2={o.z1:.2f}..{o.z2:.2f}")
    if obs:
        fr = min(obs, key=lambda o: o.dist)
        vx, wz, st, f2 = avoid_decision([fr])
        print(f"[决策] 前目标 d={fr.dist:.2f} az={fr.azimuth_deg:+.1f}° -> {st} v={vx:.2f} wz={wz:+.2f}")
    else:
        print("[决策] 无前方障碍 -> FREE 巡航")
    return obs


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/one.json')
