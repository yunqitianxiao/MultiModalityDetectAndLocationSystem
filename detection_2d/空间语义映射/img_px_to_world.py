# -*- coding: utf-8 -*-
"""
地铁病害巡检 · 空间语义映射与精准定位 核心算法
================================================================
一句话：把 2D 检测框 (u,v,w,h) 变成隧道里的物理坐标
        {里程 / 环号 / 时钟方位 / 3D坐标 / 尺寸(m)}，
        即赛题要求的"像素坐标 -> 物理三维空间 -> 全局地理信息"连续坐标转换链条。

坐标变换链（正投影：图像 -> 世界）：
    像素 (u,v) + 深度 d
      (1) 相机内参 K        -> 相机系 X_cam = K^{-1} [u,v,1] d      # 深度沿相机 z
      (2) 外参 T_cam_base   -> 车体系 X_base = T_cam_base * X_cam
      (3) 位姿 T_world_base -> 世界系 X_world = T_world_base * X_base
      (4) 隧道几何          -> 里程(沿轴线) / 环号(里程/环宽) / 时钟方位(截面内角向)

逆投影（点云 -> 图像，"3D属性图映射回2D"）：
      雷达点(车体系) --T_cam_base + K--> 图像平面
      得到 深度图 / 反射率图 / 高度图(逐像素投影) + BEV俯视高度图(异物侵限/限界判断用)

坐标系约定：
    base_link : x前 / y左 / z上      (REP-103, 与 inspection_robot URDF 一致)
    camera光心: z前 / x右 / y下       (标准针孔模型, 深度沿 z 正方向)
    lidar     : 雷达系(外参 T_lidar_base 可配, 默认与 base_link 重合)
    world     : Z 上                  (与 Gazebo / 惯导里程计一致)
    tunnel    : 里程沿轴线递增; 环号从1起; 时钟方位 12点=正上方, 3点=面向行车方向右侧

设计原则：
    - 只依赖 numpy(+scipy 可选), 不依赖 ROS —— 任何环境可单测、可进报告;
    - 所有标定/几何参数集中在"传感器参数.py", 换车/换隧道只改那里;
    - 核心入口是 DetectionLocalizer.locate(): 喂 检测框 + 雷达点 + 车体位姿, 吐病害记录 JSON。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict

import numpy as np


# ================================================================
# 一、相机内参
# ================================================================
def build_K_from_hfov(hfov_rad: float, width: int, height: int,
                      cx: float | None = None, cy: float | None = None) -> np.ndarray:
    """由水平视场角 + 分辨率构造针孔内参 K (假设正方形像素 fy=fx)。

    地铁真实相机若用张正友标定得到非方像素 K, 直接手动赋值即可, 不必用本函数。
    """
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    cx = width / 2.0 if cx is None else cx
    cy = height / 2.0 if cy is None else cy
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]])


# ================================================================
# 二、坐标系变换 (纯几何, 无学习参数)
# ================================================================
def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """欧拉角(弧度) -> 旋转矩阵, 顺序 Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """四元数(x,y,z,w 顺序不限, 参数语义: w 标量在前) -> 旋转矩阵。"""
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n == 0:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy+qz*qz),     2*(qx*qy-qw*qz),     2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),         1 - 2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),         2*(qy*qz+qw*qx),     1 - 2*(qx*qx+qy*qy)],
    ])


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """R,t -> 4x4 齐次矩阵。"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def pose_to_T(px: float, py: float, pz: float,
              roll: float = 0., pitch: float = 0., yaw: float = 0.,
              qw: float | None = None, qx: float | None = None,
              qy: float | None = None, qz: float | None = None) -> np.ndarray:
    """位姿 -> T_world_base。给 rpy 或四元数二者之一; 给四元数时 rpy 被忽略。"""
    if qw is not None:
        R = quat_to_R(qw, qx, qy, qz)
    else:
        R = rpy_to_R(roll, pitch, yaw)
    return make_T(R, np.array([px, py, pz]))


def gz_camera_extrinsic(xyz_in_base=(0.22, 0.0, 0.10),
                        rpy_in_base=(0.0, 0.0, 0.0)) -> np.ndarray:
    """由 Gazebo/URDF 相机安装位姿 -> T_cam_base。

    Gazebo 相机传感器帧"光轴朝 +x"(与车体前进一致), 而标准针孔模型用
    "光轴朝 +z / x右 / y下"。两者相差一个固定旋转 R_opt->cam:
        opt x(右) -> cam -y ;  opt y(下) -> cam -z ;  opt z(前) -> cam +x
    本函数把"装在前方 rpy=0"的 Gazebo 相机换算成标准针孔光心外参。
    若相机在车上的安装 rpy 非 0, 先按 URDF 的 xacro 改这里。
    """
    R_opt_to_cam = np.array([[0.,  0.,  1.],
                             [-1.,  0.,  0.],
                             [0., -1.,  0.]])
    T_install = pose_to_T(*xyz_in_base, *rpy_in_base)      # cam_link(与base同向) -> base
    T_opt_to_install = make_T(R_opt_to_cam, np.zeros(3))   # 光心 -> cam_link
    return T_install @ T_opt_to_install                    # 光心 -> base


# ================================================================
# 三、正投影: 像素 -> 相机系 -> 车体系 -> 世界系
# ================================================================
def pixel_to_camera(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    """像素 + 深度 -> 相机系坐标 (z_cam = depth)。"""
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.array([x, y, depth])


def camera_to_base(X_cam: np.ndarray, T_cam_base: np.ndarray) -> np.ndarray:
    """相机系 -> 车体系 (外参)。"""
    X4 = np.concatenate([X_cam, [1.0]])
    return (T_cam_base @ X4)[:3]


def base_to_world(X_base: np.ndarray, T_world_base: np.ndarray) -> np.ndarray:
    """车体系 -> 世界系 (位姿)。"""
    X4 = np.concatenate([X_base, [1.0]])
    return (T_world_base @ X4)[:3]


def pixel_to_world(u: float, v: float, depth: float,
                   K: np.ndarray, T_cam_base: np.ndarray,
                   T_world_base: np.ndarray) -> np.ndarray:
    """像素 + 深度 -> 世界系坐标 (整条链的(1)(2)(3))。"""
    return base_to_world(camera_to_base(pixel_to_camera(u, v, depth, K), T_cam_base),
                         T_world_base)


def project_to_image(X_base_pts: np.ndarray, K: np.ndarray,
                     T_cam_base: np.ndarray, img_size=(1280, 720)):
    """车体系下的3D点 -> 图像坐标。

    返回: uv (N,2) 像素坐标; valid (N,) 是否在图像内; depth (N,) 相机系深度。
    用于"点云投影回2D"(逆映射), 也用于深度查询时把雷达点投进图像。
    """
    X_cam = (np.linalg.inv(T_cam_base) @
             np.concatenate([X_base_pts, np.ones((len(X_base_pts), 1))], axis=1).T).T
    X_cam = X_cam[:, :3]
    z = X_cam[:, 2]
    valid_f = z > 1e-3
    uv = np.full((len(X_base_pts), 2), np.nan)
    u = K[0, 0] * X_cam[:, 0] / np.where(z, z, 1) + K[0, 2]
    v = K[1, 1] * X_cam[:, 1] / np.where(z, z, 1) + K[1, 2]
    uv[:, 0], uv[:, 1] = u, v
    W, H = img_size
    in_img = valid_f & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, in_img, z


# ================================================================
# 四、深度来源: 用雷达点云"喂"给检测框 (2D检测 -> 3D 的核心)
# ================================================================
def depth_from_lidar_bbox(points_base: np.ndarray, bbox, K: np.ndarray,
                          T_cam_base: np.ndarray,
                          img_size=(1280, 720), reflectivity: np.ndarray | None = None):
    """取检测框内雷达点的中值深度 + 中值3D点, 作为该病害的深度先验。

    原理: 可见光在暗光/弱纹理下没有绝对尺度, 雷达提供尺度; 一个病害是一个
    面区域, 用框内点云中值比用单点稳得多。若框内点太少(雷达稀疏), 会退化为
    以 bbox 中心为圆心扩展半径找最近点, 保证总有一个深度可用。

    参数:
      points_base : (N,3) 雷达点(车体系)。   reflectivity : (N,) 可选强度。
      bbox        : [x1,y1,x2,y2] 检测框(像素)。
    返回:
      depth, anchor_base, n_pts
      - depth        : 中值深度(相机系 z), 用于像素逆投影
      - anchor_base  : (3,) 该深度的3D锚点(车体系), 作为病害在点云里的代表点
      - n_pts        : 框内有效点数(0 表示退化了)
    """
    x1, y1, x2, y2 = [float(b) for b in bbox]
    uv, in_img, z = project_to_image(points_base, K, T_cam_base, img_size)
    inside = in_img & (uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2)
    idx = np.where(inside)[0]
    if len(idx) >= 3:
        z_mid = float(np.median(z[idx]))
        pt_mid = np.median(points_base[idx], axis=0)      # 车体系锚点
        n = len(idx)
    else:
        # 退化: 框内没有雷达点(太远/太稀疏), 在框中心附近找最近点
        cu, cv = (x1 + x2) / 2, (y1 + y2) / 2
        dist2 = (uv[:, 0] - cu) ** 2 + (uv[:, 1] - cv) ** 2
        valid = in_img & np.isfinite(dist2)
        if valid.sum() == 0:
            raise ValueError("bbox 附近没有任何雷达点, 无法获得深度(检查外参/雷达范围)")
        j = int(np.argmin(np.where(valid, dist2, np.inf)))
        z_mid = float(z[j])
        pt_mid = points_base[j]
        n = 0
    return z_mid, pt_mid, n


# ================================================================
# 五、几何属性反求: 病害实际尺寸(m)
# ================================================================
def bbox_size_meters(bbox, depth: float, K: np.ndarray) -> tuple[float, float]:
    """由检测框 + 中心深度反求物理尺寸 (近似正对相机平面的情形)。

    w = (x2-x1)/fx * d, h = (y2-y1)/fy * d。
    斜视/透视收缩时偏小, 精确做法是把框四角逆投影后拟合平面, 见 README 改进项。
    """
    x1, y1, x2, y2 = [float(b) for b in bbox]
    w = (x2 - x1) / K[0, 0] * depth
    h = (y2 - y1) / K[1, 1] * depth
    return w, h


# ================================================================
# 六、隧道几何: 世界系 -> 里程 / 环号 / 时钟方位
# ================================================================
@dataclass
class TunnelGeometry:
    """地铁隧道(盾构圆隧道)的几何模型。默认直隧道, 弯隧道换 polyline 模型即可。

    axis_point  : 隧道轴线上里程=start_mileage 处的世界坐标 (3,)
    axis_dir    : 轴线单位方向, 指向里程递增方向 (3,)
    start_mileage: 起点里程(m), 通常=某标桩号
    ring_width  : 单环宽度(m), 盾构环常见 1.2~1.5, 用你的数据定
    radius      : 隧道半径(m), 用于可视化/限界, 定位不依赖它
    up          : 世界"上"方向(默认 Z 上, 与 Gazebo 一致)
    """
    axis_point: np.ndarray
    axis_dir: np.ndarray
    start_mileage: float = 0.0
    ring_width: float = 1.5
    radius: float = 2.5
    up: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))

    def __post_init__(self):
        self.axis_point = np.asarray(self.axis_point, dtype=float)
        self.axis_dir = np.asarray(self.axis_dir, dtype=float)
        n = np.linalg.norm(self.axis_dir)
        if n == 0:
            raise ValueError("axis_dir 不能为零向量")
        self.axis_dir = self.axis_dir / n
        self.up = np.asarray(self.up, dtype=float) / np.linalg.norm(self.up)

    @property
    def e_up(self) -> np.ndarray:
        """截面平面内"12点钟"方向(世界"上"在截面上的投影)。"""
        a = self.axis_dir
        u_p = self.up - self.up @ a * a
        return u_p / np.linalg.norm(u_p)

    @property
    def e_right(self) -> np.ndarray:
        """截面平面内"3点钟"方向(面向行车方向 / 里程递增方向时的右手侧)。"""
        return np.cross(self.axis_dir, self.e_up)

    def center_at_mileage(self, mileage: float) -> np.ndarray:
        """里程 m 处隧道圆心(轴线上的点)。"""
        return self.axis_point + self.axis_dir * (mileage - self.start_mileage)

    def world_to_tunnel(self, X_world: np.ndarray) -> dict:
        """世界坐标 -> {里程, 环号, 径向, 时钟方位角, 时钟方位字符串, 隧道圆心}。"""
        X = np.asarray(X_world, dtype=float)
        mileage = float(self.start_mileage + (X - self.axis_point) @ self.axis_dir)
        center = self.center_at_mileage(mileage)
        r = X - center                                  # 径向向量(从圆心指向病害)
        angle = math.atan2(r @ self.e_right, r @ self.e_up)   # 0@12点, +90@3点(顺时针)
        angle_deg = math.degrees(angle) % 360.0
        hour = int(round(angle_deg / 30.0)) % 12
        ring_no = int(math.floor((mileage - self.start_mileage) / self.ring_width)) + 1
        return {
            "里程_m": round(mileage, 3),
            "环号": ring_no,
            "径向距离_m": round(float(np.linalg.norm(r)), 3),
            "时钟方位_deg": round(angle_deg, 1),
            "时钟方位": "12点" if hour == 0 else f"{hour}点钟",
            "隧道圆心": center.tolist(),
        }


# ================================================================
# 七、点云属性图: 深度 / 反射率 / 高度 -> 2D 图像 (逆映射)
# ================================================================
def lidar_to_image_attributes(points_base: np.ndarray, T_cam_base: np.ndarray,
                              K: np.ndarray, img_size=(1280, 720),
                              reflectivity: np.ndarray | None = None,
                              z_world: np.ndarray | None = None):
    """把雷达点云投影成图像分辨率的稀疏属性图(深度/反射率/高度), 供2D检测辅助。

    返回三个 (H,W) float32 数组, 无点处为 NaN:
      depth_map    : 相机系深度 z
      reflec_map   : 反射率强度 (没给 reflectivity 则全 NaN)
      height_map   : 高度(给 z_world 用世界Z, 否则用车体系 z)
    """
    H, W = img_size[1], img_size[0]
    uv, in_img, z = project_to_image(points_base, K, T_cam_base, img_size)
    depth_map = np.full((H, W), np.nan, dtype=np.float32)
    reflec_map = np.full((H, W), np.nan, dtype=np.float32)
    height_map = np.full((H, W), np.nan, dtype=np.float32)
    if in_img.sum() == 0:
        return depth_map, reflec_map, height_map
    u = np.round(uv[in_img, 0]).astype(int)
    v = np.round(uv[in_img, 1]).astype(int)
    u = np.clip(u, 0, W - 1)            # 取整可能越界, 钳到图像内
    v = np.clip(v, 0, H - 1)
    depth_map[v, u] = z[in_img].astype(np.float32)
    if reflectivity is not None:
        reflec_map[v, u] = reflectivity[in_img].astype(np.float32)
    if z_world is not None:
        height_map[v, u] = z_world[in_img].astype(np.float32)
    else:
        height_map[v, u] = points_base[in_img, 2].astype(np.float32)
    return depth_map, reflec_map, height_map


def lidar_to_bev_height(points_base: np.ndarray, res: float = 0.05,
                        x_range=(-3, 3), y_range=(-3, 3), z_world: np.ndarray | None = None):
    """BEV 俯视高度图(网格化点云, 每格取最高点高度), 用于异物侵限/限界/避障。"""
    nx = int((x_range[1] - x_range[0]) / res)
    ny = int((y_range[1] - y_range[0]) / res)
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    if z_world is None:
        z_world = points_base[:, 2]
    ix = ((points_base[:, 0] - x_range[0]) / res).astype(int)
    iy = ((points_base[:, 1] - y_range[0]) / res).astype(int)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if m.sum() == 0:
        return grid
    # 每格取该格内最高点的高度(异物/限界判断关心"最高凸起")
    flat = iy[m] * nx + ix[m]
    order = np.lexsort((z_world[m], flat))
    sorted_flat = flat[order]
    unique, first = np.unique(sorted_flat, return_index=True)
    grid[unique // nx, unique % nx] = z_world[m][order][first].astype(np.float32)
    return grid


# ================================================================
# 八、核心入口: DetectionLocalizer —— 检测框 -> 病害记录
# ================================================================
@dataclass
class DetectionLocalizer:
    """空间语义映射与精准定位解算引擎。

    参数:
      K            : 3x3 相机内参
      T_cam_base   : 4x4 相机光心 -> 车体外参
      tunnel       : TunnelGeometry 隧道几何
      img_size     : 相机分辨率 (W, H)
      default_depth: 找不到雷达深度时的兜底(可选, 单位 m)

    用法:
      loc = DetectionLocalizer(K, T_cam_base, tunnel)
      record = loc.locate(detections, lidar_points_base, T_world_base)
    """
    K: np.ndarray
    T_cam_base: np.ndarray
    tunnel: TunnelGeometry
    img_size: tuple = (1280, 720)
    default_depth: float | None = None

    def locate(self, detections, lidar_points_base: np.ndarray,
               T_world_base: np.ndarray, reflectivity: np.ndarray | None = None) -> dict:
        """对一帧的检测结果做空间语义映射, 返回病害记录 JSON。

        参数:
          detections      : 列表, 每项 {class_id/class_name, bbox:[x1,y1,x2,y2], confidence}
          lidar_points_base: (N,3) 本帧雷达点(车体系), 与相机时间同步
          T_world_base    : 4x4 车体位姿(由惯导+里程计或 FAST-LIO2 给出)
          reflectivity    : (N,) 可选, 雷达强度(写进记录/辅助检测用)
        返回:
          {"frame", "pos_xy_m", "pos_z_m", "defects":[...]}
        """
        records = []
        for d in detections:
            cls_name = str(d.get("class_name", d.get("class_id", "unknown")))
            bbox = d["bbox"]
            conf = float(d.get("confidence", 0.0))
            # (2-1) 检测框 -> 深度(雷达) + 3D锚点(车体系)
            try:
                depth, anchor_base, n_pts = depth_from_lidar_bbox(
                    lidar_points_base, bbox, self.K, self.T_cam_base, self.img_size)
            except ValueError:
                if self.default_depth is None:
                    continue                                # 无深度可用, 跳过该框
                depth = self.default_depth
                anchor_base = None
                n_pts = 0
            # (2-2) 框中心像素 -> 相机系 -> 车体系 -> 世界系
            cu, cv = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            X_cam_center = pixel_to_camera(cu, cv, depth, self.K)
            X_base_center = camera_to_base(X_cam_center, self.T_cam_base)
            X_world = base_to_world(X_base_center, T_world_base)
            # (2-3) 几何属性反求: 实际尺寸(m)
            w_m, h_m = bbox_size_meters(bbox, depth, self.K)
            # (2-4) 隧道系换算
            tun = self.tunnel.world_to_tunnel(X_world)
            records.append({
                "类型": cls_name,
                "置信度": round(conf, 4),
                "像素框": [round(v, 1) for v in bbox],
                "3D坐标_m": [round(v, 3) for v in X_world.tolist()],
                "车体系锚点_m": None if anchor_base is None else [round(v, 3) for v in anchor_base.tolist()],
                "框内雷达点数": int(n_pts),
                "尺寸_m": [round(w_m, 3), round(h_m, 3)],
                **tun,                                   # 里程/环号/时钟方位/径向/圆心
            })
        return {
            "frame_id": None,
            "pos_xy_m": [round(float(T_world_base[0, 3]), 3), round(float(T_world_base[1, 3]), 3)],
            "pos_z_m": round(float(T_world_base[2, 3]), 3),
            "defects": records,
        }


# ================================================================
# 九、输出工具
# ================================================================
def record_to_json(record: dict) -> str:
    """病害记录 -> 缩进 JSON 字符串(可直接进报告/写入文件)。"""
    return json.dumps(record, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 冒烟测试: 保证模块导入即可用(真正的闭环验证在"仿真验证_定位链.py")
    K = build_K_from_hfov(1.047, 1280, 720)
    print("K=\n", K)
    print("fx(fy) ≈", round(K[0, 0], 1), "  cx,cy =", K[0, 2], K[1, 2])
    # 顺时针方位验证: 世界(0,1,0) 在 x 轴向隧道里应为 3 点钟
    tun = TunnelGeometry(axis_point=np.array([0., 0., 1.5]),
                         axis_dir=np.array([1., 0., 0.]), start_mileage=1000.0)
    print(tun.world_to_tunnel(np.array([12.5, -2.0, 1.5])))
