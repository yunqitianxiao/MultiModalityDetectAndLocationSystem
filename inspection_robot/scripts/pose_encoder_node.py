#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位姿 MLP 编码器节点 (在线推理)
================================================================
订阅 /pose_state (Float64MultiArray, 12 维)
  -> PoseMLP (data_gen/pose_mlp.py, 3 层 MLP)
  -> 发布 /pose_embedding (Float64MultiArray, 128 维)

权重来源:
  -p ckpt:=/path/model.pt   加载训练好的权重(推荐, 由 train_pose_encoder.py 产出)
  不给 ckpt 时用随机初始化, 仅用于打通话题链路(嵌入无意义, 别直接用于融合)

用法:
  ros2 run inspection_robot pose_encoder_node.py
  ros2 run inspection_robot pose_encoder_node.py --ros-args -p ckpt:=~/pose_mlp.pt
"""
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# 让脚本在源码树/安装目录两种运行方式下都能 import data_gen
_DIR = os.path.dirname(os.path.abspath(__file__))
for _P in (_DIR, os.path.dirname(_DIR)):
    if _P not in sys.path and os.path.isdir(os.path.join(_P, 'data_gen')):
        sys.path.insert(0, _P)
        break

import torch  # noqa: E402
from data_gen.pose_mlp import PoseMLP, load_ckpt  # noqa: E402


class PoseEncoderNode(Node):
    def __init__(self):
        super().__init__('pose_encoder_node')
        self.declare_parameter('ckpt', '')
        self.declare_parameter('state_topic', '/pose_state')

        ckpt = self.get_parameter('ckpt').value
        if ckpt and os.path.exists(ckpt):
            self.model = load_ckpt(ckpt)
            self.get_logger().info(f'位姿编码器已加载权重: {ckpt}')
        else:
            self.model = PoseMLP()
            self.model.eval()
            self.get_logger().warn(
                f'未找到 ckpt({ckpt or "空"}), 使用随机初始化'
                ' —— 仅打通链路, 嵌入无意义')

        self.sub = self.create_subscription(
            Float64MultiArray, self.get_parameter('state_topic').value,
            self._on_state, 10)
        self.pub = self.create_publisher(
            Float64MultiArray, '/pose_embedding', 10)
        self.get_logger().info(
            'pose_encoder_node 已启动: '
            f'{self.get_parameter("state_topic").value} -> '
            f'MLP({self.model.dim_in}->{self.model.dim_emb}) -> /pose_embedding')

    def _on_state(self, msg):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size != self.model.dim_in:
            self.get_logger().warn(
                f'状态维度 {data.size} != 期望 {self.model.dim_in}, 丢弃')
            return
        with torch.no_grad():
            emb = self.model(torch.from_numpy(data).unsqueeze(0)).squeeze(0)
        out = Float64MultiArray()
        out.data = emb.tolist()
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PoseEncoderNode()
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
