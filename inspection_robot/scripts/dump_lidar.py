#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""服务器点云标定: 订阅 /lidar/points(PointCloud2) 打印 frame_id 与 原始 x/y/z 统计,
确定: (a) 相对哪个 frame; (b) 前向轴是 +x? (c) z 朝上还是朝下; (d) 桶是否需平移。"""
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


def decode(m):
    step = m.point_step
    arr = np.frombuffer(m.data, dtype=np.uint8).reshape(-1, step)
    off = {f.name: f.offset for f in m.fields}
    def fld(n):
        if n not in off:
            return None
        b = arr[:, off[n]:off[n] + 4].copy()
        return b.view(np.float32)
    return {n: fld(n) for n in ('x', 'y', 'z')}


def main():
    rclpy.init()
    n = Node('dump_lidar')
    got = {'k': 0}

    def cb(m):
        got['k'] += 1
        d = decode(m)
        print(f"########## 帧#{got['k']} ##########       frame_id = {m.header.frame_id}")
        for k in ('x', 'y', 'z'):
            v = d[k]
            if v is None:
                print(f"  {k}: 该字段不存在"); continue
            vv = v[np.isfinite(v)]
            print(f"  {k}: N={len(vv)} min={np.nanmin(vv):+.2f} max={np.nanmax(vv):+.2f} "
                  f"mean={np.nanmean(vv):+.2f}")
        for k in ('x', 'y', 'z'):
            v = d[k]
            if v is None:
                continue
            v = v[np.isfinite(v)]
            if k == 'x':
                print(f"  [前向?] x>0: {int((v>0).sum())}  x>3: {int((v>3).sum())}  x<-3: {int((v<-3).sum())}")
            if k == 'y':
                print(f"  [左右] |y|<0.6: {int((np.abs(v)<0.6).sum())}  |y|>1.0: {int((np.abs(v)>1.0).sum())}")
            if k == 'z':
                print(f"  [z向上?] z<-0.1: {int((v<-0.1).sum())}  z>1.0: {int((v>1.0).sum())}  "
                      f"z∈[0,0.5]: {int(((v>=0)&(v<=0.5)).sum())}")
        if got['k'] >= 3:
            n.destroy_node(); rclpy.shutdown()

    n.create_subscription(PointCloud2, '/lidar/points', cb, 10)
    print('[dump_lidar] 订阅 /lidar/points, 等 3 帧...', flush=True)
    rclpy.spin(n)


if __name__ == '__main__':
    main()
