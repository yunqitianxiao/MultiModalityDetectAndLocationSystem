#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从隧道世界几何离线生成 Nav2 占用栅格地图 (tunnel_map.pgm + tunnel_map.yaml)。

为什么离线生成: 本世界是完全已知、静态、确定性的(墙/钢轨/特征物都写死在
worlds/inspection_tunnel_base.world)。直接按同一套几何栅格化, 得到一张
可复现、可版本管理的地图(像 .world 一样成为仓库资产), 不必先开车建图。

几何常量与 worlds/inspection_tunnel_base.world 一一对应(勿改一处忘另一处):
  隧道:  x∈[-19.80,19.80] 可行驶, 侧墙内表面 y=±1.25, 端墙内表面 x=±19.80
  钢轨:  y=±0.717, 宽 0.09 (半宽 0.045)
  特征物: red_box(3,0.8) 0.8×0.8; green_box(6,-0.8) 0.6×0.6 yaw0.4;
          blue_cylinder(9,0.8) r0.3; yellow_box(12,-0.8) 0.9×0.6 yaw0.2

用法:
  python3 scripts/app_nav/gen_tunnel_map.py                       # 默认写到 config/nav2/
  python3 scripts/app_nav/gen_tunnel_map.py --res 0.05 --out config/nav2
"""
import os
import math
import argparse
import struct
import sys

# ---------------- 隧道几何 (与 worlds/inspection_tunnel_base.world 对齐) ----------------
# 侧墙: 中心 y=±1.4, 厚 0.3 -> 内表面 y=±1.25
SIDE_WALL_INNER_Y = 1.25
# 端墙: 中心 x=±20, 厚 0.4 -> 内表面 x=±19.80
END_WALL_INNER_X = 19.80
# 钢轨: y=±0.717, 宽 0.09
RAIL_Y = 0.717
RAIL_HALF_W = 0.045

# 特征物 footprint (中心, 尺寸, yaw)
FEATURES = [
    # dict: kind, cx, cy, w, h, yaw, r
    dict(kind="box", cx=3.0, cy=0.8, w=0.8, h=0.8, yaw=0.0, r=0.0),    # red_box
    dict(kind="box", cx=6.0, cy=-0.8, w=0.6, h=0.6, yaw=0.4, r=0.0),  # green_box
    dict(kind="cyl", cx=9.0, cy=0.8, w=0.0, h=0.0, yaw=0.0, r=0.3),   # blue_cylinder
    dict(kind="box", cx=12.0, cy=-0.8, w=0.9, h=0.6, yaw=0.2, r=0.0), # yellow_box
]

# 默认覆盖范围 & 分辨率
DEFAULT_RES = 0.05
DEFAULT_XMIN, DEFAULT_XMAX = -20.2, 20.2
DEFAULT_YMIN, DEFAULT_YMAX = -1.6, 1.6


def point_in_rotated_box(px, py, cx, cy, w, h, yaw):
    """点 (px,py) 是否在中心 (cx,cy)、半尺寸 w/2,h/2、绕 z 转 yaw 的矩形内。"""
    dx, dy = px - cx, py - cy
    cos, sin = math.cos(yaw), math.sin(yaw)
    # 转到盒子局部系 (矩形是正放的): 逆 yaw
    lx = cos * dx + sin * dy
    ly = -sin * dx + cos * dy
    return abs(lx) <= w / 2.0 and abs(ly) <= h / 2.0


def point_is_occupied(x, y):
    """世界坐标 (x,y) 是否属于障碍(墙/钢轨/特征物/端墙)。只做 2D 平面判定。"""
    # 1) 可行驶内表面之外 -> 端墙/侧墙 (边界外全视为墙)
    if abs(x) > END_WALL_INNER_X:
        return True
    if abs(y) > SIDE_WALL_INNER_Y:
        return True

    # 2) 钢轨 (只在隧道内有效; 外侧已被规则1判为墙)
    if abs(abs(y) - RAIL_Y) <= RAIL_HALF_W:
        return True

    # 3) 特征物
    for f in FEATURES:
        if f["kind"] == "box":
            if point_in_rotated_box(x, y, f["cx"], f["cy"], f["w"], f["h"], f["yaw"]):
                return True
        else:  # cyl
            if math.hypot(x - f["cx"], y - f["cy"]) <= f["r"]:
                return True
    return False


def build_grid(res, xmin, xmax, ymin, ymax):
    """返回 (width, height, rows) —— rows[i][j]=True 表示占用。i=行(0=顶部), j=列(0=左)。"""
    width = int(round((xmax - xmin) / res))
    height = int(round((ymax - ymin) / res))
    xmax = xmin + width * res
    ymax = ymin + height * res

    rows = []
    for r in range(height):
        # 行0=图片顶部=世界最大 y
        y = ymax - (r + 0.5) * res
        row = []
        for c in range(width):
            x = xmin + (c + 0.5) * res
            row.append(point_is_occupied(x, y))
        rows.append(row)

    num_occ = sum(1 for r in rows for v in r if v)
    num_total = width * height
    return width, height, xmin, ymin, res, rows, num_occ, num_total


def write_pgm(path, width, height, rows, res):
    """写 P5 二进制 PGM: 0=占用, 254=自由。"""
    # 地图像素: 0=占用(黑), 254=自由(白)。与 PGM 灰度一致。
    px = []
    for r in rows:
        for v in r:
            px.append(0 if v else 254)
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (width, height))
        f.write(bytes(px))


def write_yaml(path, image_basename, res, xmin, ymin, width, height):
    neg_x = -xmin
    # map yaml origin = 地图 (0,0) 像素(左下角=最小x,最小y) 的世界位姿
    origin = [xmin, ymin, 0.0]
    yaml_text = f"""image: {image_basename}
resolution: {res}
origin: [{origin[0]:.3f}, {origin[1]:.3f}, {origin[2]}]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
    with open(path, "w") as f:
        f.write(yaml_text)


def sanity_check(xmin, ymin, res, num_occ, num_total, width, height):
    """内建简单校验: (0,0) 起点应为自由, 占用占比合理。"""
    # 起点 (0,0) 世界坐标 -> 栅格
    col = int((0.0 - xmin) / res)
    row = height - int((0.0 - ymin) / res) - 1  # 世界 y=0 所在行
    free_start = True
    print(f"[gen_tunnel_map] 占用 {num_occ}/{num_total} 格 ({100.0*num_occ/max(num_total,1):.1f}%)")
    print(f"[gen_tunnel_map] 起点世界 (0,0) 栅格(row={row}, col={col}) 应自由")
    return num_occ, num_total


def main():
    ap = argparse.ArgumentParser(description="生成隧道 Nav2 占用地图 (PGM+YAML)")
    ap.add_argument("--res", type=float, default=DEFAULT_RES)
    ap.add_argument("--out", default=None, help="输出目录(默认 config/nav2)")
    ap.add_argument("--xmin", type=float, default=DEFAULT_XMIN)
    ap.add_argument("--xmax", type=float, default=DEFAULT_XMAX)
    ap.add_argument("--ymin", type=float, default=DEFAULT_YMIN)
    ap.add_argument("--ymax", type=float, default=DEFAULT_YMAX)
    args = ap.parse_args()

    proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = args.out or os.path.join(proj, "config", "nav2")
    os.makedirs(out, exist_ok=True)

    width, height, xmin, ymin, res, rows, num_occ, num_total = build_grid(
        args.res, args.xmin, args.xmax, args.ymin, args.ymax)

    pgm_path = os.path.join(out, "tunnel_map.pgm")
    yaml_path = os.path.join(out, "tunnel_map.yaml")
    write_pgm(pgm_path, width, height, rows, res)
    write_yaml(yaml_path, "tunnel_map.pgm", res, xmin, ymin, width, height)

    sanity_check(xmin, ymin, res, num_occ, num_total, width, height)
    print(f"[gen_tunnel_map] 已写出:\n  {pgm_path}\n  {yaml_path}")
    print(f"[gen_tunnel_map] 尺寸 {width}x{height} @ {res}m, origin=({xmin:.3f},{ymin:.3f})")

    # 快速断言: 起点 (0,0) 必须自由
    col = int((0.0 - xmin) / res)
    row = height - int((0.0 - ymin) / res) - 1
    if rows[row][col]:
        sys.exit("[gen_tunnel_map] ERROR: 起点 (0,0) 被判为占用, 地图有问题")
    return 0


if __name__ == "__main__":
    sys.exit(main())
