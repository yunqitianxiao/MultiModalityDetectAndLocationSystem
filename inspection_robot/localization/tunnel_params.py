# -*- coding: utf-8 -*-
"""
隧道几何与传感器参数 —— 换隧道/换车只改这里
================================================================
与当前仿真世界对齐:
  - 隧道为矩形净空(非圆盾构): 墙内表面 y=±1.25(净宽2.5m), 顶板底 z=2.75(净高2.75m),
    轨顶 z≈0.15, 轨道 y=±0.717, 全长40m(x∈[-20,20]) —— 见 worlds/inspection_tunnel_base.world。
  - 传感器标定直接读 output_v7/calibration.json(K/T_cam_base/T_lidar_base/img_size),
    与数据管线一致, 不硬编码。

定位链 TunnelGeometry 用"轴线 + 里程 + 环宽"的通用模型, 里程/环号/时钟方位只依赖
轴点/轴方向/环宽, 与隧道截面形状无关(矩形/圆形均可); radius 仅用于可视化/限界。
**轴点必须设隧道端墙 x=-20(而非世界原点)**: 否则里程=1000+x 在 x<0 时出现负里程/负环号。

避障检测专用阈值:
  RAIL_TOP        : 轨顶车体系高度(m), 高于它一个台阶即视为凸起
  INTRUSION_Z_MAX : 凸起高度上界, 用于排除隧道壁/顶板(它们 z≈2.5/2.75, 不是障碍)
"""
import json
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from localization.img_px_to_world import TunnelGeometry  # noqa: E402


# ---------------------------------------------------------------
# 1. 隧道几何(与 worlds/inspection_tunnel_base.world 对齐)
# ---------------------------------------------------------------
TUNNEL = dict(
    axis_point=np.array([-20.0, 0.0, 1.5]),   # 隧道端墙 x=-20 处为里程起点
    axis_dir=np.array([1.0, 0.0, 0.0]),       # 轴线方向 = 里程递增方向
    start_mileage=1000.0,                     # 起点里程(m), 例如 K1000
    ring_width=1.5,                           # 单环宽度(m)
    radius=2.5,                               # 隧道半径(m), 仅可视化/限界参考
    up=np.array([0.0, 0.0, 1.0]),             # 世界"上"
)

# ---------------------------------------------------------------
# 2. 避障检测阈值
# ---------------------------------------------------------------
RAIL_TOP = 0.15            # 轨顶车体系高度(m)
INTRUSION_Z_MAX = 2.0      # 障碍高度上界(m), 排除隧道壁/顶板(z≈2.5/2.75)
BEV_RES = 0.05             # BEV 高度图分辨率(m)
BEV_X_RANGE = (0.0, 8.0)   # 车前检测范围(m)
BEV_Y_RANGE = (-1.5, 1.5)  # 横向范围(m)(墙内表面 ±1.25, 留裕量)


# ---------------------------------------------------------------
# 3. 避障决策阈值(两轮差速, 无法横移; 阈值受 urdf chassis 尺寸/ DiffDrive 上限约束)
# ---------------------------------------------------------------
AVOID_BODY_FRONT = 0.225     # base_link 中心 -> 车头前缘 (chassis_length/2): 真净距 = distance - 此值
AVOID_BODY_HALF_W = 0.175    # 车体半宽 (chassis_width/2)
AVOID_V_CRUISE = 0.40        # 巡航速度 (m/s), 与录制 vx 一致
AVOID_STEER_V_MAX = 0.25     # 转向机动最大前进速度 (m/s)
AVOID_STOP_DIST = 0.50       # 前向障碍 < 此距离 -> 刹停 (0.225 车头 + 0.275 余量)
AVOID_SLOW_DIST = 1.60       # 开始减速
AVOID_STEER_DIST = 2.60      # 开始转向避让
AVOID_DETECT_MAX = 8.0       # = BEV_X_RANGE[1]
AVOID_RESUME_HYST = 0.30     # 刹停后退出迟滞: 需 d0 > STOP + 此值
AVOID_YAW_MAX = 0.60         # 最大转向角速度 (rad/s), DiffDrive 上限 2.0
AZ_DEAD_CENTER_DEG = 8.0     # |方位| < 此值 = 正前方/两难 -> 不转向, 优先刹停
AZ_ABEAM_DEG = 80.0          # |方位| > 此值 = 已侧掠 (obstacle beside) -> 不计入"前方"
AZ_STEER_LATCH_TICKS = 10    # 转向方向锁存 tick (约 0.5s @20Hz)
GAP_DIST_SIMILAR = 0.8       # 两障碍距离差 < 此值视为"并排"
GAP_PASS_MIN = 0.50          # 缝隙可过最小宽度 = 车宽 0.35 + 0.15 余量

# ---------------------------------------------------------------
# 3b. 路径决策解码器(融合检测 -> 行驶指令; 接近-检查语义, 含异物不避让)
# ---------------------------------------------------------------
# 类 id 为各支局部(土建 0/1/2=裂纹/渗漏/剥落; 设备 0/1=异物入侵/支架松脱)。
# 触发 conf 对齐 conf=0.25 工作点; 不同类可设独立阈值。
DEC_CONF_THR = {0: 0.25, 1: 0.25, 2: 0.25}
DEC_FACE_KP = 0.01           # APPROACH 轻转向增益(rad/s per deg), 让目标居中
DEC_V_CREEP = 0.05           # APPROACH 最低爬行速度(m/s): 斜坡在 d0->STOP 时线性归 0,
                             # 纯运动学会渐进逼近却永远到不了 STOP 触发点(死锁) -> 给下限让它真正顶到
                             # 并锁存 INSPECT。真实车本来就有一个最低起爬速度, 不会渐进到 0。

# ---------------------------------------------------------------
# 4. FOV 遮挡回退 (depth_discontinuity; 深度估计缺失时基于遮挡直避)
# ---------------------------------------------------------------
FOV_FRONT_TRACK_Y = 0.6      # 轨道带 |y|<0.6, 过滤轨/墙 (y=±0.717/±1.25) 的近距跳变
FOV_R_NEAR = 0.4             # 前向遮挡距离下界 (m)
FOV_R_FAR = 3.0              # 前向遮挡上界 (m), 缩到近场
FOV_MIN_BINS = 3             # 前向锥内至少 3 个方位 bin 同时近跳才算遮挡


# ---------------------------------------------------------------
# 5. 标定读取: calibration.json -> (K, T_cam_base, T_lidar_base, img_size)
# ---------------------------------------------------------------
def load_calibration(out_dir: str):
    """读数据管线标定文件(带 4x4 T_cam_base/T_lidar_base 的 json)。

    calibration.json 由 record_paired_data 写入(见 data_gen/output_v7/calibration.json),
    含 K(3x3), T_cam_base/T_lidar_base(4x4), img_size(2,), class_names。
    返回 dict {K, T_cam_base, T_lidar_base, img_size, class_names}。
    """
    path = os.path.join(out_dir, 'calibration.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'缺标定文件: {path}')
    with open(path) as f:
        c = json.load(f)
    K = np.asarray(c['K'], dtype=float)
    T_cam_base = np.asarray(c['T_cam_base'], dtype=float)
    T_lidar_base = np.asarray(c.get('T_lidar_base', np.eye(4)), dtype=float)
    img_size = tuple(int(v) for v in c.get('img_size', [1280, 720]))
    return {
        'K': K, 'T_cam_base': T_cam_base, 'T_lidar_base': T_lidar_base,
        'img_size': img_size, 'class_names': c.get('class_names', []),
    }


def build_tunnel(**overrides) -> TunnelGeometry:
    """构造 TunnelGeometry, 支持临时覆盖(如换起点里程)。"""
    cfg = dict(TUNNEL)
    cfg.update(overrides)
    return TunnelGeometry(**cfg)


if __name__ == '__main__':
    # 冒烟: 打印隧道几何 + 从 output_v7 读标定
    tun = build_tunnel()
    print('TunnelGeometry:', tun.axis_point, tun.axis_dir,
          f'start={tun.start_mileage} ring={tun.ring_width}')
    print('里程/环号 @x=0: ', tun.world_to_tunnel([0.0, 0.0, 0.0]))
    root = os.path.join(_DIR, 'data_gen', 'output_v7')
    cal = load_calibration(root)
    print('标定: K fx=', round(cal['K'][0, 0], 1),
          'T_cam_base=\n', cal['T_cam_base'],
          'T_lidar_base=', cal['T_lidar_base'].tolist(),
          'img_size=', cal['img_size'])
