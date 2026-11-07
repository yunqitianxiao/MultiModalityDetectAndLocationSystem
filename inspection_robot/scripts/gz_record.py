#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gz 原生配对数据采集器 (无 ROS/rclpy 依赖, 直连 gz transport, 跑到 V100 服务器)
================================================================================
等价于 record_paired_data.py, 但跳过 ros_gz_bridge, 直接用 gz 话题:

    订阅 /camera(gz.msgs.Image), /model/inspection_robot/odometry(gz.msgs.Odometry),
          /lidar/points/points(gz.msgs.PointCloudPacked)
    发布 /model/inspection_robot/cmd_vel(gz.msgs.Twist)  驱动小车往返

逐 sample_interval 秒采样, 落盘:
    images/{i}.jpg     相机 BGR 图 (与 record_paired_data 一致, cv2 存盘)
    clouds/{i}.npy     车体系点云 (N,4) [x,y,z,intensity], T_LIDAR_BASE 变换
    poses/{i}.json     里程位姿 x,y,z,qx,qy,qz,qw + stamp
    metadata.json      list[{sample, img_stamp, cloud_stamp, odom_stamp, n_cloud}]
    data.yaml          仅供 build_attrmaps/训练参考 (nc/names 由 defcts 侧补全,
                       这里先写占位, 见 build_labels_offline.py)

labels/{i}.txt 与 calibration.json 由 build_labels_offline.py 离线重算
(几何投影 corner_world -> 图像, 与 record_paired_data 的 build_gt_boxes 完全一致,
 不依赖点云, 从而本采集器无需读取 defects.json)。

用法:
    python3 gz_record.py --out_dir DIR [--vx 0.4] [--t_forward 40] \
        [--sample_interval 0.5] [--sync_tol 0.15] [--seed 0]

依赖: numpy, cv2, gz.transport13, gz.msgs10 (均已在服务器 pip/apt 装到 system python3)。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2

from gz import transport13 as tr
from gz.msgs10 import image_pb2, odometry_pb2, twist_pb2, pointcloud_packed_pb2

# common.py: 服务器无 pro/2 定位链 -> 走内嵌默认 K/T (与 xacro/传感器参数一致)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "data_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from data_gen.common import K, T_CAM_BASE, T_LIDAR_BASE, IMG_SIZE  # noqa: E402


def _stamp(m) -> float:
    h = m.header.stamp
    return h.sec + h.nsec * 1e-9


class GzRecorder:
    def __init__(self, out_dir: str, vx: float, t_forward: float,
                 sample_interval: float, sync_tol: float):
        self.out_dir = out_dir
        self.vx = vx
        self.t_forward = t_forward
        self.sample_interval = sample_interval
        self.sync_tol = sync_tol

        self.node = tr.Node()

        # 话题
        self.node.subscribe(image_pb2.Image, "/camera", self.cb_img)
        self.node.subscribe(odometry_pb2.Odometry,
                            "/model/inspection_robot/odometry", self.cb_odom)
        self.node.subscribe(pointcloud_packed_pb2.PointCloudPacked,
                            "/lidar/points/points", self.cb_pc)
        self.pub_cmd = self.node.advertise(
            "/model/inspection_robot/cmd_vel", twist_pb2.Twist)

        # 状态
        self.latest_img = None            # (stamp, BGR ndarray)
        self.lidar_buf = {}               # stamp -> PointCloudPacked
        self.odom_buf = {}                # stamp -> Odometry
        self.start_time = None
        self._ready_at = None
        self.last_sample = None
        self.n_samples = 0
        self.finished = False
        self._meta = []

        for sub in ("images", "clouds", "poses"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        print(f"[gz_record] vx={vx} t_forward={t_forward}x2 采样={sample_interval}s "
              f"同步容差={sync_tol}s 输出={out_dir}", flush=True)

    # ---------- 回调 ----------
    def cb_img(self, m):
        arr = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, -1)
        if arr.shape[2] == 3:
            arr = arr[:, :, ::-1]          # RGB8 -> BGR 存盘
        self.latest_img = (_stamp(m), arr.copy())

    def cb_pc(self, m):
        s = _stamp(m)
        self.lidar_buf[s] = m
        if len(self.lidar_buf) > 12:
            for k in sorted(self.lidar_buf)[:-8]:
                self.lidar_buf.pop(k, None)

    def cb_odom(self, m):
        s = _stamp(m)
        self.odom_buf[s] = m
        if len(self.odom_buf) > 24:
            for k in sorted(self.odom_buf)[:-16]:
                self.odom_buf.pop(k, None)

    # ---------- 主循环 ----------
    def tick(self):
        if self.finished:
            return
        now = time.time()

        if self.start_time is None:
            if self.latest_img and self.lidar_buf and self.odom_buf:
                if self._ready_at is None:
                    self._ready_at = now
                    print("[gz_record] 三类话题已到, 等待物理稳定 1s", flush=True)
                elif now - self._ready_at >= 1.0:
                    self.start_time = now
                    print("[gz_record] 数据已就绪, 开始行驶", flush=True)
            else:
                self._ready_at = None
            if self.start_time is None:
                self._publish_twist(0.0)
                return

        elapsed = now - self.start_time
        if elapsed < self.t_forward:
            vx = self.vx
        elif elapsed < 2 * self.t_forward:
            vx = -self.vx
        else:
            vx = 0.0
            if not self.finished:
                self.finished = True
                print(f"[gz_record] 录制完成, 共 {self.n_samples} 帧 -> {self.out_dir}",
                      flush=True)
                self._write_meta()
            return
        self._publish_twist(vx)

        if self.last_sample is None or (now - self.last_sample) >= self.sample_interval:
            self.last_sample = now
            self._record_sample(now)

    def _publish_twist(self, vx: float):
        t = twist_pb2.Twist()
        t.linear.x = vx
        t.linear.y = 0.0
        t.linear.z = 0.0
        t.angular.x = t.angular.y = t.angular.z = 0.0
        self.pub_cmd.publish(t)

    # ---------- 同步 + 落盘 ----------
    def _nearest(self, buf, t):
        if not buf:
            return None
        key = min(buf, key=lambda k: abs(k - t))
        if abs(key - t) > self.sync_tol:
            return None
        return buf[key]

    def _record_sample(self, now):
        if self.latest_img is None:
            return
        img_stamp, img = self.latest_img
        odom_msg = self._nearest(self.odom_buf, img_stamp)
        pc_msg = self._nearest(self.lidar_buf, img_stamp)
        if odom_msg is None or pc_msg is None:
            print(f"[gz_record] 同步失败@{now:.2f}s "
                  f"(odom={'无' if odom_msg is None else 'ok'} "
                  f"lidar={'无' if pc_msg is None else 'ok'})", flush=True)
            return

        idx = self.n_samples
        self.n_samples += 1
        prefix = f"{idx:06d}"
        cv2.imwrite(os.path.join(self.out_dir, "images", f"{prefix}.jpg"), img)

        cloud = self._pc_to_base(pc_msg)
        np.save(os.path.join(self.out_dir, "clouds", f"{prefix}.npy"), cloud)

        p = odom_msg.pose.position
        q = odom_msg.pose.orientation
        pose = {"x": p.x, "y": p.y, "z": p.z,
                "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w,
                "stamp": round(img_stamp, 4)}
        with open(os.path.join(self.out_dir, "poses", f"{prefix}.json"), "w") as f:
            json.dump(pose, f, indent=2)

        self._meta.append({
            "sample": prefix, "img_stamp": round(img_stamp, 4),
            "cloud_stamp": round(_stamp(pc_msg), 4),
            "odom_stamp": round(_stamp(odom_msg), 4),
            "pose": pose, "n_cloud": int(cloud.shape[0]),
        })

    def _pc_to_base(self, m) -> np.ndarray:
        n = m.width * m.height
        ps = m.point_step
        buf = np.frombuffer(m.data, dtype=np.uint8)
        rows = as_strided(buf, shape=(n, ps), strides=(ps, 1))
        off = {f.name: f.offset for f in m.field}

        def _f32(name):
            o = off.get(name)
            return None if o is None else \
                np.ascontiguousarray(rows[:, o:o + 4]).view(np.float32).reshape(-1)

        x, y, z = _f32("x"), _f32("y"), _f32("z")
        intensity = _f32("intensity")
        if intensity is None:
            intensity = np.ones(n, dtype=np.float32)
        xyz = np.stack([x, y, z], axis=1)
        ok = np.isfinite(xyz).all(axis=1)
        xyz = xyz[ok]
        intensity = intensity[ok]
        X4 = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1).T
        base = (T_LIDAR_BASE @ X4).T[:, :3]
        return np.concatenate(
            [base, intensity.reshape(-1, 1)], axis=1).astype(np.float32)

    def _write_meta(self):
        with open(os.path.join(self.out_dir, "metadata.json"), "w") as f:
            json.dump(self._meta, f, indent=2)
        yaml = ("path: " + self.out_dir + "\n"
                "train: images\nval: images\ntest: images\n"
                f"nc: 8\n"
                "names:\n" + "\n".join(f"  {i}: n{i}" for i in range(8)))
        with open(os.path.join(self.out_dir, "data.yaml"), "w") as f:
            f.write(yaml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--t_forward", type=float, default=40.0)
    ap.add_argument("--sample_interval", type=float, default=0.5)
    ap.add_argument("--sync_tol", type=float, default=0.15)
    args = ap.parse_args()

    rec = GzRecorder(args.out_dir, args.vx, args.t_forward,
                     args.sample_interval, args.sync_tol)
    # 20Hz 控制循环
    t0 = time.time()
    while not rec.finished:
        rec.tick()
        dt = time.time() - t0 - 0.05
        if dt < 0.05:
            time.sleep(0.05 - dt)
        else:
            t0 = time.time() + dt - 0.05  # 抛掉滞后


if __name__ == "__main__":
    main()
