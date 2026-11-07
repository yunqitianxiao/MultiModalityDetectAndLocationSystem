#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
odom -> base_link 动态 TF 广播节点

Gazebo Harmonic 的 DiffDrive 插件只发布 gz 里程计(桥接到 ROS /odom),
本身不发布 odom->base_link 的 TF。本节点订阅 /odom,
把里程计位姿转换为 TF 广播出去, 从而补全:
    odom -> base_link (动态)  -> left_wheel/right_wheel/... (robot_state_publisher 发布)

这样 RViz / FAST-LIO2 等依赖 TF 的工具就能看到小车在地图中运动。

运行:
    ros2 run inspection_robot odom_tf_broadcaster.py
参数:
    child_frame_id: 默认 base_link (也可设为 base_footprint, 若加了该根)
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTfBroadcaster(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')

        self.child_frame = self.declare_parameter('child_frame_id', 'base_link').value
        self.parent_frame = 'odom'

        self.tf_broadcaster = TransformBroadcaster(self)

        # 订阅桥接后的 /odom (nav_msgs/Odometry)
        self.sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.get_logger().info(
            f'odom_tf_broadcaster 已启动: 订阅 /odom, 广播 {self.parent_frame}->{self.child_frame}')

    def odom_callback(self, msg):
        t = TransformStamped()
        # 时间戳与 frame 取自里程计
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame

        # 位置: 直接使用里程计在世界系下的位姿
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfBroadcaster()
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
