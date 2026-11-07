#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 gz 相机抓帧节点: 订阅一个 ROS 图像话题, 存 N 帧到目录。
用法: python3 scripts/gz_capture.py --topic /camera/image_raw --out /tmp/cap --n 3 [--warm 6]
      --warm: 等待首帧的秒数(传感器初始化慢, 默认 6s)。
"""
import argparse
import os
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np


class Capturer(Node):
    def __init__(self, topic: str, out_dir: str, n: int, warm: float):
        super().__init__("gz_capture")
        self.out_dir = out_dir
        self.n = n
        self.warm = warm
        self.saved = 0
        os.makedirs(out_dir, exist_ok=True)
        self.sub = self.create_subscription(Image, topic, self.cb, 10)
        self.t0 = time.time()

    def cb(self, msg: Image):
        if self.saved >= self.n:
            return
        # 传感器未暖机时常为全 0/全 1 帧, 等 warm 秒
        if time.time() - self.t0 < self.warm:
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            bgr = arr[:, :, :3].copy()
        except Exception as e:
            self.get_logger().warn(f"reshape 失败: {e}")
            return
        if bgr.mean() < 1.0:  # 全 0 黑帧
            return
        cv2.imwrite(os.path.join(self.out_dir, f"{self.saved:04d}.png"), bgr)
        self.get_logger().info(f"已存 {self.saved}/{self.n}: mean={bgr.mean():.1f}")
        self.saved += 1
        if self.saved >= self.n:
            self.get_logger().info("抓帧完成, 退出")
            rclpy.shutdown()


def main(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="/camera/image_raw")
    p.add_argument("--out", default="/tmp/cap")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--warm", type=float, default=6.0)
    args = p.parse_args(args)

    rclpy.init()
    cap = Capturer(args.topic, args.out, args.n, args.warm)
    rclpy.spin(cap)
    print(f"✅ 抓帧完成: {args.out}")


if __name__ == "__main__":
    main()
