#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位姿状态向量组装节点 (方案1: 直接订阅 /odom + /imu)
================================================================
把导航状态(位姿+速度)组装成 12 维向量, 供 MLP 编码器/多模态融合网络使用。

状态向量 (12 维):
    [x, y, z,                    位置     (odom, 米; 除以 pos_scale 归一化)
     sin(roll), cos(roll),       姿态     (imu, 平路≈0)
     sin(pitch), cos(pitch),
     sin(yaw), cos(yaw),         yaw 取 odom (与位置同帧, 保证一致)
     vx, vy,                     线速度   (odom, 米/秒)
     wz]                         角速度   (imu 陀螺, 弧度/秒)

为什么这么拼:
    - 角度编码成 (sin, cos) 避免 0/2π 不连续(MLP 学不动的点)
    - 位置/yaw 取 odom, roll/pitch 取 imu: 各取最准的维度
    - 平路隧道里 roll/pitch≈0, 但用 imu 真值比直接假设为 0 更稳

输出:
    /pose_state  (std_msgs/Float64MultiArray, 12 维, 定时发布)
    CSV 可选: --ros-args -p log_csv:=/path/pose_state.csv

用法:
    ros2 run inspection_robot pose_state_node.py
"""
import csv

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray


def quat_to_rpy(w, x, y, z):
    """四元数 -> (roll, pitch, yaw)。numpy 内联实现, 不依赖 tf_transformations。"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class PoseStateNode(Node):
    def __init__(self):
        super().__init__('pose_state_node')

        self.declare_parameter('publish_hz', 20.0)
        self.declare_parameter('pos_scale', 10.0)  # 位置除以该尺度归一化(米)
        self.declare_parameter('log_csv', '')       # 非空则写 CSV
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('imu_topic', '/imu')

        hz = float(self.get_parameter('publish_hz').value)
        self.pos_scale = float(self.get_parameter('pos_scale').value)
        csv_path = self.get_parameter('log_csv').value
        self._odom = None
        self._imu = None

        self._csv = None
        if csv_path:
            self._csv = csv.writer(open(csv_path, 'w', newline=''))
            self._csv.writerow(['t', 'x', 'y', 'z',
                                'sinr', 'cosr', 'sinp', 'cosp', 'siny', 'cosy',
                                'vx', 'vy', 'wz'])

        self.sub_odom = self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self._on_odom, 10)
        self.sub_imu = self.create_subscription(
            Imu, self.get_parameter('imu_topic').value, self._on_imu, 10)

        self.pub = self.create_publisher(Float64MultiArray, '/pose_state', 10)
        self.timer = self.create_timer(1.0 / hz, self._on_timer)

        self.get_logger().info(
            f'pose_state_node 已启动: 订阅 {self.get_parameter("odom_topic").value}'
            f' + {self.get_parameter("imu_topic").value}, 输出 /pose_state @ {hz}Hz')

    def _on_odom(self, msg):
        self._odom = msg

    def _on_imu(self, msg):
        self._imu = msg

    def _on_timer(self):
        if self._odom is None or self._imu is None:
            return  # 两个话题都收到才开始出状态

        odom = self._odom.pose.pose
        twist = self._odom.twist.twist
        imu_o = self._imu.orientation

        p = odom.position
        # yaw 取 odom(与位置同帧), roll/pitch 取 imu(重力参考)
        _, _, yaw = quat_to_rpy(odom.orientation.w, odom.orientation.x,
                                odom.orientation.y, odom.orientation.z)
        roll, pitch, _ = quat_to_rpy(imu_o.w, imu_o.x, imu_o.y, imu_o.z)

        state = np.array([
            p.x / self.pos_scale, p.y / self.pos_scale, p.z / self.pos_scale,
            np.sin(roll), np.cos(roll),
            np.sin(pitch), np.cos(pitch),
            np.sin(yaw), np.cos(yaw),
            twist.linear.x, twist.linear.y,
            self._imu.angular_velocity.z,
        ], dtype=float)

        msg = Float64MultiArray()
        msg.data = state.tolist()
        self.pub.publish(msg)

        if self._csv is not None:
            st = self._odom.header.stamp
            self._csv.writerow(
                [st.sec + st.nanosec * 1e-9] + state.tolist())


def main(args=None):
    rclpy.init(args=args)
    node = PoseStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
