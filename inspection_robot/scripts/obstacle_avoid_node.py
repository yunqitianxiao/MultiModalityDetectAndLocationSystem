#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何避障节点 —— 点云几何检测 -> 真避让 -> /cmd_vel (架构图⑤ 上到 ROS)
================================================================
对 inspection_tunnel.world(仅 地/墙/顶 + red_box/green_box/blue_cylinder, 无钢轨/扣件):

    /lidar/points(PointCloud2) -> 车体系点云 -> geometric_obstacle_detect.detect_obstacles
        -> 避让决策(纯函数 avoid_decision, 复用本地 AVOID_* 常量值) -> geometry_msgs/Twist /cmd_vel

避让行为(A 真避让, 用户确认):
  - 前方(|az|<AZ_ABEAM_DEG=80)无障碍 -> 巡航 AVOID_V_CRUISE, 不转。
  - 前方障碍, 真净距 d0=dist-AVOID_BODY_FRONT:
      d0<=STOP(0.5)   -> 刹停(vx=0), 不转(含正前两难)。
      d0<=SLOW(1.6)   -> 减速进场(斜坡)。
      d0<=STEER(2.6)  -> 转向避让: 障碍偏左(az>0)右转(wz<0), 偏右左转; |az|近 0 两难 -> 刹停。
  - 障碍距 > DETECT_MAX 不算。
事件驱动: 有障碍且 |az| 侧向足够 -> 照车宽+余量向空当转; 转不动(墙边) -> 刹停。

用法(服务器, ROS2 bash):
  source /opt/ros/humble/setup.bash && cd /root/autodl-tmp/inspection_robot
  ros2 run ... (或) python3 scripts/obstacle_avoid_node.py --topic /lidar/points

离线单测(不接 ROS, 用合成车体系点云验证检测+决策):
  python3 obstacle_avoid_node.py --offline --synthetic
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_DIR)
_LOCAL = os.path.join(_PKG, 'localization')
for _p in (_DIR, _LOCAL, _PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geometric_obstacle_detect as so  # noqa: E402  检测器(localization/)

# ---- 复用本地 tunnel_params.py 的 AVOID_* 常量值(自足, 免依赖隧道参数模块) ----
AVOID_BODY_FRONT = 0.225    # base 中心 -> 车头前缘
AVOID_V_CRUISE = 0.40
AVOID_STEER_V_MAX = 0.30    # 转向机动最大前速(稍快 -> 转弯半径大, 防原地打转 jackknife)
AVOID_STOP_DIST = 0.50
AVOID_SLOW_DIST = 1.60
AVOID_STEER_DIST = 4.00     # 提前进 STEER(live 实测2.6 太晚, 留给横向时间不足)
AVOID_DETECT_MAX = 8.0
AVOID_RESUME_HYST = 0.30
AVOID_YAW_MAX = 0.18     # 柔和偏航上限: turn radius=v/wz=0.30/0.18≈1.7m 大半径弧线,
                         # 防窄廊里 0.60(→0.5m)原地打转扎墙
AZ_DEAD_CENTER_DEG = 8.0
AZ_ABEAM_DEG = 80.0

# 转向: 空当越大越早转向(进入 STEER 区即满偏), 越接近障碍越收(已积累侧移, 防过度转)
AVOID_STEER_KP = 0.90        # rad/s per (空当侧) —— 手感标定, 后面用 EVAL 调
AVOID_TRACK_HW = 1.075       # 可通过走廊半宽 = 墙(wall 1.25) - 车半宽(0.175), 用于判空当
AVOID_GAP_PASS_MIN = 0.50    # 空当宽度 >= 此才算"可过"(否则刹停)
AVOID_PASS_HALF = 0.30       # 车前进脚印半宽(车半宽0.175+余量): 障碍 base-y 落在此内才算撞头(按距离刹),
                             # 落在侧方(y1>此 或 y2<-此)说明车能旁绕 -> 不刹、继续沿旁道走(live 复测:box近角0.31在侧方却误刹)

# ---- 世界系剔墙 + 走廊安全(车转向时墙在车体系 y 会变, 必须在世界系剔, 否则残墙条+盒
#      粘成横跨整轨簇 -> 判无空当 -> 误刹停) ----
WALL_WORLD_Y = 0.95          # 世界系 |world_y| > 此剔墙. 墙=结构(非障碍), 只作边界由 corridor_keep 兜底;
                             # 剔墙阈值取 0.95(略大于车道 0.65+余量), 让墙内面永不成为"前方障碍"——否则转向
                             # 时残墙条进 FOV 变新 front, 叠让转向扎墙(live 复测 yaw -91.8° 入墙).
CORRIDOR_Y_SAFE = 0.85       # 车世界 |y| > 此 = 逼近墙(墙已剔, 用 odom 兜底), 走向中心纠偏
CORRIDOR_KP = 0.80           # 走廊纠偏 yaw 增益 rad/s per m
CORRIDOR_CAP = 0.50          # 逼墙时前速上限(降速)


def _choose_side(o):
    """依据障碍 AABB(y1..y2) 在走廊里的两侧剩余空当, 选可过的那一侧。

    走廊 y ∈ [-AVOID_TRACK_HW, +AVOID_TRACK_HW]; 障碍占 y∈[y1,y2]。
      left_gap  = AVOID_TRACK_HW - y2   (障碍"左边"? 即正 y 侧余量)
      right_gap = y1 + AVOID_TRACK_HW   (负 y 侧余量)
    返回 (steer_sign, gap): steer_sign=+1 向左转(wz>0), -1 向右转; 两侧都 <gap_min -> (0, gap)。

    这是"真避让"关键: 不只看障碍方位, 要看哪一侧真的能过。横跨整轨 -> 0 -> 刹停。
    """
    y1, y2 = float(o.y1), float(o.y2)
    left_gap = AVOID_TRACK_HW - y2
    right_gap = y1 + AVOID_TRACK_HW
    best = max(left_gap, right_gap)
    if best < AVOID_GAP_PASS_MIN:
        return 0.0, best
    return (1.0 if left_gap >= right_gap else -1.0), best


# 车道跟踪(odom 世界系): 隧道直(沿 +x), 障碍占一侧则走另一侧"车道"。lane_y 只由空当侧决定,
# 用 (lane_y - car_y) 得目标航向再与当前 yaw 比产生比例转向。锚定世界系, 不随车转向漂移,
# 收敛后保持直行 -> 窄廊里不过冲到墙。需 odom(car_y,yaw)。
LANE_Y_INNER = 0.65        # 目标车道世界 |y|(居中障碍 edge±0.29 -> 需 |car_y|>0.29 才净过; 0.65 留 0.36 余量)
LANE_LOOKAHEAD = 1.20      # 前视距离(=> 期望航向 atan2(err, look)); 小=转向更进取进空当
LANE_KP = 0.40             # 航向误差增益 rad/s per rad(0.55 刚起步就满偏过转, 0.40 更柔和, 配 AVOID_YAW_MAX=0.18)


def obs_world_y(front, car_x, car_y, yaw):
    """障碍中心的世界 y(把 base 帧 cx,cy 经 yaw 转回世界系)。>0=障碍在+ y(左)侧。
    世界系恒定值, 不随车转向漂移 —— 这是防"base 帧 AABB 随转向翻转导致朝障碍转"的关键。"""
    sy, cy = math.sin(yaw), math.cos(yaw)
    return car_y + sy * front.cx + cy * front.cy


# 侧向锁定(世界系, 带迟滞): 障碍世界 y 恒定, 但近距聚类噪声使 base 帧簇心漂移,
# 恢复出的 world_y 会在 ±0 附近抖, 一抖就把侧向翻错(走回障碍侧) -> 车过转朝盒、正前误刹。
# 故: 仅当 |world_y| 明显居中(>deadband)才重判侧向; 模糊时沿用上一次锁定侧, 全程不抖翻。
_SIDE_DEADBAND = 0.20    # |world_y| 小于此视为"正前两难/噪声区", 不改侧向
_SIDE_X_RESET = 0.75     # 障碍世界 x 跳变超此 = 换障碍, 重置锁定
_side_lock_x = None      # 当前锁定障碍近似世界 x (identity)
_side_lock = 0.0         # 锁定侧: -1 右 / +1 左 / 0 未定


def _steer_side(front, car_x=0.0, car_y=0.0, yaw=0.0):
    """决定走哪侧(世界系, 稳定, 带迟滞): 障碍在左(+y) -> 走右(-1), 在右 -> 走左(+1)。
    不用 base 帧 AABB 判侧 —— box 在 base 帧的 y-range 会随车转向漂移, 在 y1+y2≈0 处翻转
    (live 复测曾因此过冲到 yaw+41° 朝障碍转)。世界系 box 的 y 恒定, 侧向稳定。
    加迟滞锁定: 近距聚类抖动只出现在 |world_y| 小(近中心)处, 此时沿用既有侧, 不随帧抖翻。"""
    global _side_lock_x, _side_lock
    wy = obs_world_y(front, car_x, car_y, yaw)
    wx = car_x + front.cx                      # 障碍近似世界 x (identity)
    # 换障碍(世界 x 跳变)或首次 -> 重置锁定
    if _side_lock_x is None or abs(wx - _side_lock_x) > _SIDE_X_RESET:
        _side_lock_x = wx
        _side_lock = 0.0
    if abs(wy) > _SIDE_DEADBAND:                # 侧向清晰 -> 锁定(受控更新)
        _side_lock = -1.0 if wy > 0 else +1.0
    elif _side_lock == 0.0:                     # 从未锁定(正前两难) -> base 帧空当大侧兜底
        y1, y2 = float(front.y1), float(front.y2)
        _side_lock = -1.0 if (y1 + AVOID_TRACK_HW) >= (AVOID_TRACK_HW - y2) else +1.0
    return _side_lock


def lane_keep(front, car_x, car_y, yaw):
    """odom 车道跟踪(世界系): 世界侧 -> lane_y(world) -> 目标航向 -> 比例 yaw(封顶)。
    侧向由 obs_world_y 世界系判定, 不随车转向翻转, 收敛后保持直行 -> 窄廊里不过冲到墙。"""
    side = _steer_side(front, car_x, car_y, yaw)
    lane_y = side * LANE_Y_INNER
    err = lane_y - car_y
    desired = math.atan2(err, LANE_LOOKAHEAD)
    wz = float(np.clip(LANE_KP * (desired - yaw), -AVOID_YAW_MAX, AVOID_YAW_MAX))
    return wz


def avoid_decision(obs, v_now: float = 0.0, car_x=0.0, car_y=0.0, yaw=0.0) -> tuple:
    """几何障碍列表 -> (vx, wz, state, front) 。obs: list[so.Obstacle]。

    state in {'FREE','STEER_AVOID','BRAKE'}; front=选中的主障碍(最近前方)。
    真避让:
      - 无前方目标        -> 巡航, 不转。
      - d0<=STOP 或目标侧空当不足 -> 刹停(vx=0, 不转)。
      - STOP<d0<=STEER 且目标侧可过 -> 朝空当偏转, 前速斜坡+封顶。
      - d0>STEER           -> 巡航直行(进 STEER 前不预转, 免漂移)。
    侧向用世界系判定(障碍世界 y 恒定), 不随车转向翻转 → 稳定不振荡。
    """
    front = None
    cands = [o for o in obs if abs(o.azimuth_deg) < AZ_ABEAM_DEG and o.dist <= AVOID_DETECT_MAX]
    if cands:
        cands.sort(key=lambda o: o.dist)
        front = cands[0]
    if front is None:
        return AVOID_V_CRUISE, 0.0, 'FREE', None

    y1, y2 = float(front.y1), float(front.y2)
    side = _steer_side(front, car_x, car_y, yaw)
    # 目标侧空当(障碍真跨度, base 帧 AABB): 够 <min 才刹停(真横跨整轨才无路)
    right_gap = y1 + AVOID_TRACK_HW
    left_gap = AVOID_TRACK_HW - y2
    gap_on_side = (left_gap if side > 0 else right_gap)
    if gap_on_side < AVOID_GAP_PASS_MIN:
        return 0.0, 0.0, 'BRAKE', front          # 目标侧空当不足(横跨整轨), 刹停
    d0 = front.dist - AVOID_BODY_FRONT          # 真净距(径向)
    in_path = (y1 <= AVOID_PASS_HALF) and (y2 >= -AVOID_PASS_HALF)   # 障碍横在车前进脚印内
    if in_path and d0 <= AVOID_STOP_DIST:
        return 0.0, 0.0, 'BRAKE', front          # 真撞头: 障碍在脚迹内且太近, 刹停
    if d0 > AVOID_STEER_DIST:
        return AVOID_V_CRUISE, 0.0, 'FREE', front  # 远, 直行(免漂移)
    # STEER 窗口: wz=朝空当偏转(无 odom 时由 node 用 lane_keep(odom)覆盖)。
    # 速度: 障碍在车前进脚印内(in_path) -> 接近斜坡到 0(真撞头风险);
    #       在侧方可旁绕(非 in_path) -> 保持机动巡航, 不刹到 0 —— 否则车贴着盒(径向距~0.7)就停死。
    if in_path:
        v = min(_ramp(d0), AVOID_STEER_V_MAX)
    else:
        v = AVOID_STEER_V_MAX
    ease = float(np.clip((d0 - AVOID_STOP_DIST) / (AVOID_STEER_DIST - AVOID_STOP_DIST), 0.0, 1.0))
    wz = float(np.clip(side * AVOID_STEER_KP * ease, -AVOID_YAW_MAX, AVOID_YAW_MAX))
    return v, wz, 'STEER_AVOID', front


def _ramp(d0: float) -> float:
    """真净距 -> 目标前速 (STOP=0 .. SLOW=巡航) 线性斜坡 + 巡航封顶。"""
    if d0 <= AVOID_STOP_DIST:
        return 0.0
    if d0 >= AVOID_SLOW_DIST:
        return AVOID_V_CRUISE
    return AVOID_V_CRUISE * (d0 - AVOID_STOP_DIST) / (AVOID_SLOW_DIST - AVOID_STOP_DIST)


def odom_yaw(qz, qw):
    """四元数 -> 平面 yaw (绕 z)。"""
    return math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * qz * qz)


def wallfree(pts, car_x, car_y, yaw, ymax=WALL_WORLD_Y):
    """车体系点云 -> 在世界系剔墙(|world_y|>ymax)。车转向时墙在车体系 y 变化,
    必须用车方向转回世界系按 世界|y| 剔, 否则 1.20~1.25 残墙条 + 盒粘成横跨簇(误刹停)。
    返回剔墙后的车体系点云(x,y,z 不变, 只做过滤)。"""
    if len(pts) == 0:
        return pts
    sy, cy = math.sin(yaw), math.cos(yaw)
    wy = sy * pts[:, 0] + cy * pts[:, 1] + car_y
    keep = np.abs(wy) < ymax
    return pts[keep]


def corridor_keep(vx, wz, state, car_y):
    """墙已从点云剔掉 -> 用 odom 的 car_world_y 兜底: 车逼近 ±CORRIDOR_Y_SAFE 就降速+向中心纠偏,
    防车漂移冲进(看不见的)墙。返回 (vx, wz)。target 为车朝向走廊中心(世界 y=0)。"""
    if abs(car_y) <= CORRIDOR_Y_SAFE:
        return vx, wz
    # 越界(接近墙): 压到 CORRIDOR_CAP 且转向世界 y=0。car_y>0(左侧) -> 右转(wz<0); <0 -> 左转
    yaw_corr = -np.sign(car_y) * CORRIDOR_KP * (abs(car_y) - CORRIDOR_Y_SAFE)
    yaw_corr = float(np.clip(yaw_corr, -AVOID_YAW_MAX, AVOID_YAW_MAX))
    # 若正在 STEER/BRAKE(有障碍), 让障碍决策优先; 且 FREE 时兜底
    if state in ('BRAKE', 'INSPECT'):
        return vx, wz
    v = min(vx, CORRIDOR_CAP)
    wz = yaw_corr if abs(yaw_corr) > abs(wz) else wz
    return v, wz


# ---------------- PointCloud2 -> 车体系点云 ----------------
def pc2_to_base(cloud, lidar_z: float = 0.30):
    """sensor_msgs/PointCloud2(在 lidar_link 系, z-up) -> (N,3) 车体系点云。
    实测 frame=inspection_robot/base_link/lidar, 原点世界 z=0.50(离地) -> 地面点 z_lidar≈-0.50。
    lidar_link 相对 base_link 高 0.30 -> base 系 = [x, y, z+lidar_z](轴对齐, 前向+x, z 上)。"""
    from sensor_msgs.msg import PointCloud2
    pt_step = cloud.point_step
    arr = np.frombuffer(cloud.data, dtype=np.uint8).reshape(-1, pt_step)
    off = {f.name: f.offset for f in cloud.fields}
    def field(f):
        b = arr[:, off[f]:off[f] + 4].copy()          # (N,4) uint8, 连续
        return b.view(np.float32).reshape(-1)         # -> (N,) float32(每行 4 字节 = 1 float)
    x = field('x').astype(np.float64)
    y = field('y').astype(np.float64)
    z = field('z').astype(np.float64)
    return np.stack([x, y, z + lidar_z], 1)


# ---------------- ROS 节点 ----------------
def run_node(topic, rate_hz=20, lidar_z=0.30):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import PointCloud2

    rclpy.init()
    node = Node('obstacle_avoid_node')
    pub = node.create_publisher(Twist, '/cmd_vel', 10)
    state = {'v': 0.0, 'car': (0.0, 0.0, 0.0)}   # car = (x, y, yaw) 世界, 来自 odom

    def odom_cb(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        state['car'] = (float(p.x), float(p.y), odom_yaw(float(q.z), float(q.w)))

    def cb(msg):
        car_x, car_y, yaw = state['car']
        pts = pc2_to_base(msg, lidar_z)
        pts = wallfree(pts, car_x, car_y, yaw)          # 世界系剔墙(定残留横跨簇之根因)
        # 墙已由世界系剔, base 系 y_wall 放宽避免误剔(车转向时盒在 base 系 py 变大)
        obs = so.detect_obstacles(pts, y_wall=3.0)
        vx, wz, st, front = avoid_decision(obs, state['v'], car_x, car_y, yaw)
        if st == 'STEER_AVOID' and front is not None:
            wz = lane_keep(front, car_x, car_y, yaw)     # odom 车道跟踪(世界系, 稳侧防翻)
        vx, wz = corridor_keep(vx, wz, st, car_y)        # 墙不可见, 用 odom 兜底防冲墙
        state['v'] = vx
        t = Twist(); t.linear.x = float(vx); t.angular.z = float(wz)
        pub.publish(t)
        if front is not None:
            node.get_logger().info(
                f"[{st}] front d={front.dist:.2f}m az={front.azimuth_deg:+.0f}° "
                f"v={vx:.2f} wz={wz:+.2f} obs={len(obs)} car_y={car_y:+.2f}", throttle_duration_sec=0.5)
        # ---- 调试: 全障碍 + 决策细节 -> /tmp/dbg.txt (每 ~0.6s) ----
        _dbg_dump = getattr(node, '_dbg_last', 0.0)
        if front is not None and time.time() - _dbg_dump >= 0.6:
            node._dbg_last = time.time()
            with open('/tmp/dbg.txt', 'a') as f:
                f.write(f"pose=({car_x:+.2f},{car_y:+.2f},{math.degrees(yaw):+5.1f}°) v={vx:.2f}{st} n={len(obs)}\n")
                for o in obs:
                    s2, _ = _choose_side(o)
                    wy2 = obs_world_y(o, car_x, car_y, yaw)
                    f.write(f"  idx{o.idx} d={o.dist:.2f} az={o.azimuth_deg:+6.0f} cy={o.cy:+.2f}"
                            f" y=[{o.y1:+.2f},{o.y2:+.2f}] n={o.npts} w_y={wy2:+.2f} side={s2:+.0f}\n")
                sid = _steer_side(front, car_x, car_y, yaw)
                gs = (AVOID_TRACK_HW - front.y2) if sid > 0 else (front.y1 + AVOID_TRACK_HW)
                ip = (front.y1 <= AVOID_PASS_HALF) and (front.y2 >= -AVOID_PASS_HALF)
                f.write(f"  >> front {st} d={front.dist:.2f} az={front.azimuth_deg:+.0f}"
                        f" side={sid:+.0f} gap_side={gs:.2f} in_path={ip}\n")

    node.create_subscription(PointCloud2, topic, cb, 10)
    from nav_msgs.msg import Odometry
    node.create_subscription(Odometry, '/odom', odom_cb, 10)
    node.get_logger().info(f'几何避障节点: sub {topic} -> pub /cmd_vel ({rate_hz}Hz)')
    from rclpy.executors import MultiThreadedExecutor
    exec_ = MultiThreadedExecutor(); exec_.add_node(node); exec_.spin()
    exec_.shutdown(); node.destroy_node(); rclpy.shutdown()


# ---------------- 离线单测(不接 ROS) ----------------
def _synth():
    """真实 inspection_tunnel.world 几何, 车体系(base 原点离地 0.20 -> 车系 z = world-z - 0.20).
    含 地面/墙/顶/端墙/钢轨/红绿蓝黄盒。障碍件在 y=±0.8。"""
    rng = np.random.default_rng(0)
    pts = []
    def add(a): pts.append(a)
    # 地面 world z=0 -> 车系 -0.20
    add(np.c_[rng.uniform(-20, 20, 60000), rng.uniform(-1.55, 1.55, 60000),
              -0.20 + rng.normal(0, .01, 60000)])
    # 左右墙内表面 world |y|≈1.35 -> 车系 y=±1.35; z -0.20..2.85
    for s in (-1, 1):
        add(np.c_[rng.uniform(-20, 20, 9000), s * (1.35 + rng.normal(0, .01, 9000)),
                  rng.uniform(-0.20, 2.85, 9000)])
    # 顶 world z=2.9 -> 车系 2.70
    add(np.c_[rng.uniform(-20, 20, 9000), rng.uniform(-1.55, 1.55, 9000),
              2.70 + rng.normal(0, .01, 9000)])
    # 钢轨 y=±0.717, world z 0..0.15 -> 车系 -0.20..-0.05 (应被 z_min=0 剔)
    for s in (-1, 1):
        add(np.c_[rng.uniform(-20, 20, 8000), s * (0.717 + rng.normal(0, .008, 8000)),
                  rng.uniform(-0.20, -0.05, 8000)])
    # 红盒 world(3, 0.8, z0..0.5) size0.8 -> 车系 z -0.20..0.30, y0.4..1.2
    add(np.c_[rng.uniform(2.6, 3.4, 4000), rng.uniform(0.4, 1.2, 4000),
              rng.uniform(-0.20, 0.30, 4000)])
    # 绿盒 world(6,-0.8,z0..0.3) size0.6 -> 车系 z -0.20..0.10, y-1.1..-0.5
    add(np.c_[rng.uniform(5.7, 6.3, 4000), rng.uniform(-1.1, -0.5, 4000),
              rng.uniform(-0.20, 0.10, 4000)])
    # 蓝柱 world(9,0.8,z0..1.0) r0.3 -> 车系 z -0.20..0.80
    th = rng.uniform(0, 2 * np.pi, 4000); r = rng.random(4000) * 0.3
    add(np.c_[9 + r * np.cos(th), 0.8 + r * np.sin(th), rng.uniform(-0.20, 0.80, 4000)])
    # 黄盒 world(12,-0.8,z0..0.7) size0.9x0.6 -> 车系 z -0.20..0.50
    add(np.c_[rng.uniform(11.55, 12.45, 4000), rng.uniform(-1.1, -0.5, 4000),
              rng.uniform(-0.20, 0.50, 4000)])
    return np.concatenate(pts, 0)


def _fake(d, y):
    """构造一个位于 (x=d, y) 长 0.6 宽的几何障碍(围绕 y)。"""
    az = float(np.degrees(np.arctan2(y, d)))
    return so.Obstacle(idx=0, cx=d, cy=y, cz=0.3, dist=float(np.hypot(d, y)),
                       azimuth_deg=az, npts=100, x1=d, y1=y - 0.3, z1=0.1,
                       x2=d, y2=y + 0.3, z2=0.5)


def selftest():
    p = _synth()
    obs = so.detect_obstacles(p, max_range=18.0, z_min=so.Z_MIN)
    print(f'[离线] 检测到障碍 {len(obs)}')
    for o in obs:
        print(f'  dist={o.dist:.2f}m az={o.azimuth_deg:+.1f}° pos=({o.cx:+.2f},{o.cy:+.2f}) '
              f'x跨度={o.x2-o.x1:.2f}m')
    # 断言: 4 件障碍被检出(红3/绿6/蓝9/黄12, y≈±0.8), 且无长条(钢轨/端墙已被剔)
    exp = [(3.0, 0.8), (6.0, -0.8), (9.0, 0.8), (12.0, -0.8)]
    ok = True
    got = sorted([(round(o.cx, 1), round(o.cy, 1)) for o in obs])
    for e in exp:
        near = any(abs(o.cx - e[0]) < 0.5 and abs(o.cy - e[1]) < 0.5 for o in obs)
        print(f'  {"OK " if near else "FAIL"} 检出 {e} -> 位置')
        ok &= near
    for o in obs:
        if (o.x2 - o.x1) > 2.0:
            print(f'  [FAIL] 长条残留 x={o.cx:.1f} x跨度={o.x2-o.x1:.2f}m (钢轨/端墙未剔净)')
            ok = False
    print(f'[离线] 钢轨剔除: {len(obs)} 件均非长条 -> {"OK" if ok else "FAIL"}')

    print('[离线] 避让决策轨迹(从 7m 逼近):')
    # 偏左障碍(y=+0.5, az>0) -> 空当在右(负y), 应向右转 wz<0
    sims = [(7.0, 0.50, 'FREE'), (2.2, 0.50, 'STEER_AVOID'),
            (0.9, 0.50, 'STEER_AVOID'), (0.4, 0.50, 'BRAKE')]
    for d, y, exp in sims:
        vx, wz, st, fr = avoid_decision([_fake(d, y)])
        print(f'  d={d:.1f} y={y:+.2f} -> {st:<12} v={vx:.2f} wz={wz:+.2f} (期望 {exp})', end='')
        if st == 'STEER_AVOID':
            print(f'  [转向侧={-1 if wz<0 else +1}]', end='')
        print()
        ok &= (st == exp)
    ok &= (st != 'STEER_AVOID' or wz < 0)     # 障碍在左 -> 应右转(wz<0)
    # 横跨整轨(居中 y∈[-0.9,0.9]) -> 无空当 -> BRAKE
    wide = so.Obstacle(idx=0, cx=1.0, cy=0, cz=0.3, dist=1.0, azimuth_deg=0.0, npts=300,
                       x1=0.5, y1=-0.9, z1=0.1, x2=1.5, y2=0.9, z2=0.5)
    vx, wz, st, fr = avoid_decision([wide])
    print(f'  横跨整轨 d=1.0 y=[-0.9,0.9] -> {st} v={vx:.2f} (期望 BRAKE)')
    ok &= (st == 'BRAKE')
    print('>>> 离线避让单测' + ('全通过' if ok else '有失败'))
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--offline', action='store_true')
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--topic', default='/lidar/points')
    ap.add_argument('--rate', type=int, default=20)
    a = ap.parse_args()
    if a.offline:
        raise SystemExit(0 if selftest() else 1)
    run_node(a.topic, a.rate)
