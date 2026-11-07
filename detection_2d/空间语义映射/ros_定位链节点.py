# -*- coding: utf-8 -*-
"""
ROS2 集成节点（模板）—— 把定位链接到 inspection_robot 仿真小车实时跑。
================================================================
在 Gazebo 仿真中订阅 相机/雷达/里程计, 检测 + 定位, 把病害记录按帧打印/发布。
需要 ROS2 Humble 环境（source setup.sh）且装了 ultralytics 才能跑, 本机内存紧张
建议在服务器/大内存机器上跑, 结构上完全可以离线用 rosbag 重放验证。

启动顺序参考（内存够的机器）:
    source setup.sh
    ros2 launch inspection_robot inspection_robot.launch.py
    python3 ros_定位链节点.py --model /path/to/best.pt

对接要点:
    - 话题名与 ros_gz_bridge.yaml 一致: /camera /camera_info /lidar/points /odom
    - 外参/内参/隧道几何全部来自"传感器参数.py", 定位链本身不感知 ROS
    - YOLO 推理换成你三大领域的模型即可, 本文件只演示链路
"""
from __future__ import annotations

import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from img_px_to_world import DetectionLocalizer, TunnelGeometry, pose_to_T, record_to_json
from 传感器参数 import K, T_CAM_BASE, TUNNEL, IMG_SIZE


def pointcloud2_to_xyzi(msg) -> tuple[np.ndarray, np.ndarray]:
    """PointCloud2 -> (N,3) 车体系坐标 + (N,) 强度。字段名自适应 x/y/z/intensity。"""
    from sensor_msgs_py import point_cloud2
    gen = point_cloud2.read_points_numpy(msg, field_names=["x", "y", "z"])
    pts = np.ascontiguousarray(gen)
    inten = None
    try:
        inten = np.ascontiguousarray(
            point_cloud2.read_points_numpy(msg, field_names=["intensity"]))
    except Exception:
        pass
    return pts, inten


def imgmsg_to_np(msg):
    """sensor_msgs/Image -> BGR uint8 图。"""
    from cv_bridge import CvBridge
    return CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")


def odom_to_T(msg) -> np.ndarray:
    """nav_msgs/Odometry -> T_world_base（用四元数, 惯导/里程计的位姿）。"""
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    return pose_to_T(p.x, p.y, p.z, qw=q.w, qx=q.x, qy=q.y, qz=q.z)


def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, PointCloud2
    from nav_msgs.msg import Odometry
    from ultralytics import YOLO
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best.pt", help="YOLO 模型路径")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=512)
    args = ap.parse_args()

    rclpy.init()
    node = Node("defect_localizer")
    model = YOLO(args.model)

    tunnel = TunnelGeometry(**TUNNEL)
    localizer = DetectionLocalizer(K=K, T_cam_base=T_CAM_BASE, tunnel=tunnel, img_size=IMG_SIZE)

    # 用 ROS 参数机制缓存最新一帧, 三路传感器各存各的
    state = {"img": None, "pts": None, "inten": None, "T_wb": None, "stamp": 0.0}

    def on_img(msg):
        state["img"] = imgmsg_to_np(msg)

    def on_pts(msg):
        state["pts"], state["inten"] = pointcloud2_to_xyzi(msg)

    def on_odom(msg):
        state["T_wb"] = odom_to_T(msg)

    node.create_subscription(Image, "/camera", on_img, 10)
    node.create_subscription(PointCloud2, "/lidar/points", on_pts, 10)
    node.create_subscription(Odometry, "/odom", on_odom, 10)

    node.get_logger().info("定位链节点已启动, 等待传感器数据…")

    def tick():
        if state["img"] is None or state["pts"] is None or state["T_wb"] is None:
            return
        # 1) 检测
        res = model.predict(state["img"], conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        dets = [{"class_name": model.names[int(b.cls)], "bbox": b.xyxy[0].tolist(),
                 "confidence": float(b.conf)} for b in res.boxes]
        if not dets:
            return
        # 2) 定位（雷达点在车体系, 位姿来自惯导/里程计）
        record = localizer.locate(dets, state["pts"], state["T_wb"], reflectivity=state["inten"])
        node.get_logger().info("\n" + record_to_json(record))

    timer = node.create_timer(0.2, tick)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
