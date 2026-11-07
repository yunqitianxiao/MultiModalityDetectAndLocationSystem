# -*- coding: utf-8 -*-
"""
传感器与隧道几何参数 —— 唯一需要根据现场改的地方。
================================================================
本文件与仿真小车 URDF (inspection_robot/urdf/inspection_robot.urdf.xacro)
以及隧道世界 (worlds/inspection_tunnel.world) 一一对应, 换真车/换隧道改这里即可,
核心库 img_px_to_world.py 不用动。

外参约定:
    T_cam_base   : 相机光心 -> 车体系 (x前/y左/z上)
    T_lidar_base : 雷达 -> 车体系 (默认与 base_link 重合, rpy=0)
    T_imu_base   : IMU -> 车体系 (定位时用)
"""
import numpy as np
from img_px_to_world import build_K_from_hfov, gz_camera_extrinsic, pose_to_T

# ---------------------------------------------------------------
# 1. 相机 (inspection_robot xacro: horizontal_fov=60°, 1280x720, 装在前方)
# ---------------------------------------------------------------
CAMERA = {
    "width": 1280,
    "height": 720,
    "hfov_rad": 1.047,                 # 60°, 与 xacro horizontal_fov 一致
    "xyz_in_base": [0.22, 0.0, 0.10],  # 光心在车体上的安装位置(与 xacro 相机属性一致)
    "rpy_in_base": [0.0, 0.0, 0.0],    # 安装姿态
}
IMG_SIZE = (CAMERA["width"], CAMERA["height"])

K = build_K_from_hfov(CAMERA["hfov_rad"], CAMERA["width"], CAMERA["height"])
T_CAM_BASE = gz_camera_extrinsic(CAMERA["xyz_in_base"], CAMERA["rpy_in_base"])

# ---------------------------------------------------------------
# 2. 16线激光雷达 (xacro: 顶部 z=0.30, rpy=0)
# ---------------------------------------------------------------
T_LIDAR_BASE = pose_to_T(0.0, 0.0, 0.30)     # 雷达系 -> 车体系

# ---------------------------------------------------------------
# 3. IMU (xacro: z=0.02)
# ---------------------------------------------------------------
T_IMU_BASE = pose_to_T(0.0, 0.0, 0.02)       # IMU系 -> 车体系

# ---------------------------------------------------------------
# 4. 隧道几何 (世界系, Z 上, 与 Gazebo 一致)
#    直隧道模型: 轴线上里程=START_MILEAGE 处为 AXIS_POINT, 沿 AXIS_DIR 递增。
#    请按你们实际模拟/现场隧道改这三个数: 轴线起点、轴线方向、起点里程。
# ---------------------------------------------------------------
TUNNEL = dict(
    axis_point=np.array([0.0, 0.0, 1.5]),   # 隧道轴线起点(世界坐标)
    axis_dir=np.array([1.0, 0.0, 0.0]),     # 轴线方向 = 里程递增方向
    start_mileage=1000.0,                   # 起点里程(m), 例如 K1000
    ring_width=1.5,                         # 单环宽度(m), 盾构环常见 1.2~1.5
    radius=2.5,                             # 隧道半径(m), 可视化/限界用
    up=np.array([0.0, 0.0, 1.0]),           # 世界"上"方向
)

# ---------------------------------------------------------------
# 5. 把上面所有配置聚合, 方便一处 import
# ---------------------------------------------------------------
CONFIG = dict(
    camera=CAMERA,
    img_size=IMG_SIZE,
    K=K,
    T_cam_base=T_CAM_BASE,
    T_lidar_base=T_LIDAR_BASE,
    T_imu_base=T_IMU_BASE,
    tunnel=TUNNEL,
)


if __name__ == "__main__":
    print("K:\n", K)
    print("T_cam_base:\n", np.round(T_CAM_BASE, 4))
    print("T_lidar_base:\n", np.round(T_LIDAR_BASE, 4))
