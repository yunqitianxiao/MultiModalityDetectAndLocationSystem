#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配对数据录制节点 (图像 + 点云 + 里程 + 自动标注真值框)
================================================================
驱动小车沿隧道往返, 以 sim 时间同步保存每帧三元组:
    images/xxx.jpg    RGB 相机图 (1280x720, BGR 存盘)
    clouds/xxx.npy    (N,4) 车体系点云 [x,y,z,intensity]
    poses/xxx.json    里程计位姿 (x,y,z,qx,qy,qz,qw) + 时间戳
    labels/xxx.txt    YOLO 格式自动标注: "class cx cy w h" (由病害世界坐标几何投影, 不靠人工)
    attrmaps/         (由 build_attrmaps.py 后处理生成, 不在此)
另写: calibration.json / defects.json / metadata.json / data.yaml

用法(配合 launch):
    ros2 run inspection_robot record_paired_data.py   (需 launch 已起仿真+桥)
    或直接在 launch 里 record:=true 自动带起。

参数(可用 --ros-args -p ...):
    out_dir       输出目录 (默认 data_gen/output)
    vx            行驶速度 m/s (默认 0.4)
    t_forward     单程行驶秒数 (默认 40, 往返共 2x)
    sample_interval  采样间隔 s (默认 0.5)
    sync_tol      同步容差 s (默认 0.15)
"""
import json
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import as_strided
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

# 保证脚本能 import data_gen.common:
#   - 源码树运行: data_gen 在仓库根 (_ROOT)
#   - install 运行: data_gen 被装到脚本同目录 (_SCRIPT_DIR)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_gen.common import (  # noqa: E402
    K, T_CAM_BASE, T_LIDAR_BASE, IMG_SIZE,
    odom_to_T_world_base, build_gt_boxes,
)


def _stamp(msg) -> float:
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


class PairedDataRecorder(Node):
    def __init__(self):
        super().__init__("paired_data_recorder")

        # 防御式参数: launch 传来的可能是 int('40')/str('40.0'), 一律转 float
        # (带类型的 declare_parameter 遇 int->double 会抛 InvalidParameterTypeException,
        #  用 dynamic_typing=True 声明不绑定类型, 拿到值再自行转换)
        def _pfloat(name: str, default: float) -> float:
            v = self.declare_parameter(
                name, None, ParameterDescriptor(dynamic_typing=True)).value
            return default if v is None else float(v)

        def _pstr(name: str, default: str) -> str:
            v = self.declare_parameter(
                name, None, ParameterDescriptor(dynamic_typing=True)).value
            return default if v is None else str(v)

        self.out_dir = _pstr("out_dir", os.path.join(_ROOT, "data_gen", "output"))
        self.vx = _pfloat("vx", 0.4)
        self.t_forward = _pfloat("t_forward", 40.0)
        self.sample_interval = _pfloat("sample_interval", 0.5)
        self.sync_tol = _pfloat("sync_tol", 0.15)
        os.makedirs(self.out_dir, exist_ok=True)   # 先建输出目录, 后面真值文件可能放这里

        # 病害真值清单:
        #   优先 out_dir/defects.json (run_data_gen.sh 启动前已预拷贝, install 模式下也能找到)
        #   回退 仓库根 data_gen/defects.json (源码树直接 ros2 run 的场景)
        self.defects_json = os.path.join(self.out_dir, "defects.json")
        if not os.path.isfile(self.defects_json):
            self.defects_json = os.path.join(_ROOT, "data_gen", "defects.json")
        with open(self.defects_json, encoding="utf-8") as f:
            manifest = json.load(f)
        self.defects = manifest["defects"]
        self.class_names = manifest["class_names"]

        # 话题
        self.sub_img = self.create_subscription(Image, "/camera/image_raw", self.cb_img, 5)
        self.sub_pc = self.create_subscription(PointCloud2, "/lidar/points", self.cb_pc, 5)
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.cb_odom, 5)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 5)

        # 状态
        self.latest_img = None          # (stamp, np.ndarray BGR)
        self.lidar_buf = {}             # stamp -> (stamp, PointCloud2)
        self.odom_buf = {}              # stamp -> (stamp, Odometry)
        self.start_time = None          # 第一次收到所有话题的时间
        self._ready_at = None           # 三类话题齐全的起始时刻(暖机计时)
        self.last_sample = None
        self.n_samples = 0
        self.finished = False

        self._make_dirs()
        self.get_logger().info(
            f"录制参数: vx={self.vx} t_forward={self.t_forward}x2 "
            f"采样={self.sample_interval}s 同步容差={self.sync_tol}s 输出={self.out_dir}")

        # 20Hz 控制/采样定时器
        self.create_timer(0.05, self.tick)

    # ---------- 回调 ----------
    def cb_img(self, msg: Image):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if arr.shape[2] == 3:                       # RGB8 -> BGR 存盘
            arr = arr[:, :, ::-1]
        self.latest_img = (_stamp(msg), arr.copy())

    def cb_pc(self, msg: PointCloud2):
        self.lidar_buf[_stamp(msg)] = msg
        if len(self.lidar_buf) > 12:                 # 只留最近几帧
            for k in sorted(self.lidar_buf)[:-8]:
                self.lidar_buf.pop(k, None)

    def cb_odom(self, msg: Odometry):
        self.odom_buf[_stamp(msg)] = msg
        if len(self.odom_buf) > 24:
            for k in sorted(self.odom_buf)[:-16]:
                self.odom_buf.pop(k, None)

    # ---------- 主循环 ----------
    def tick(self):
        if self.finished:
            return
        now = self.get_clock().now().nanoseconds * 1e-9

        # 等待数据齐 + 物理稳定
        # 注意: 原判据 now - max(buf) > 1.0 是反的(数据正常流时最新消息≈now,
        #       差值≈0 永不为真, 只有停流>1s 才成立); 改用"三类话题连续到达
        #       ≥1s 再出发", 与 sim/wall 时钟无关。
        if self.start_time is None:
            if self.latest_img and self.lidar_buf and self.odom_buf:
                if self._ready_at is None:
                    self._ready_at = now
                    self.get_logger().info("三类话题已到, 等待物理稳定 1s")
                elif now - self._ready_at >= 1.0:
                    self.start_time = now
                    self.get_logger().info("数据已就绪, 开始行驶")
            else:
                self._ready_at = None                # 话题中断则重置计时
            if self.start_time is None:
                self._publish_twist(0.0)             # 静止等待
                return

        # 行驶相位
        elapsed = now - self.start_time
        if elapsed < self.t_forward:
            vx = self.vx
        elif elapsed < 2 * self.t_forward:
            vx = -self.vx
        else:
            vx = 0.0
            if not self.finished:
                self.finished = True
                self.get_logger().info(f"✅ 录制完成, 共 {self.n_samples} 帧 -> {self.out_dir}")
                self._write_meta()
                # 别在定时器回调里调 rclpy.shutdown(): 实测进程会卡在 futex_wait 永不退出
                # (回调内 shutdown 与 executor 争锁), launch 的 OnProcessExit 因此不触发,
                # run_data_gen.sh 整条卡在第 4 步。此处落盘已全部同步完成, 直接 _exit(0)
                # 让 launch 收到 exit 0 收尾。
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
        self._publish_twist(vx)

        # 采样
        if self.last_sample is None or (now - self.last_sample) >= self.sample_interval:
            self.last_sample = now
            self._record_sample(now)

    def _publish_twist(self, vx: float):
        t = Twist()
        t.linear.x = vx
        t.linear.y = 0.0
        t.linear.z = 0.0
        t.angular.x = t.angular.y = t.angular.z = 0.0
        self.pub_cmd.publish(t)

    # ---------- 同步 + 落盘 ----------
    def _nearest(self, buf: dict, t: float):
        if not buf:
            return None
        key = min(buf, key=lambda k: abs(k - t))
        if abs(key - t) > self.sync_tol:
            return None
        return buf[key]

    def _record_sample(self, now: float):
        if self.latest_img is None:
            return
        img_stamp, img = self.latest_img
        odom_msg = self._nearest(self.odom_buf, img_stamp)
        pc_msg = self._nearest(self.lidar_buf, img_stamp)
        if odom_msg is None or pc_msg is None:
            self.get_logger().warning(f"时间同步失败 @{now:.2f}s (odom={'无' if odom_msg is None else 'ok'}, lidar={'无' if pc_msg is None else 'ok'})")
            return

        idx = self.n_samples
        self.n_samples += 1
        prefix = f"{idx:06d}"
        img_path = os.path.join(self.img_dir, f"{prefix}.jpg")
        import cv2
        cv2.imwrite(img_path, img)

        # 点云 -> 车体系 (N,4)
        cloud = self._pc_to_base(pc_msg)
        np.save(os.path.join(self.cloud_dir, f"{prefix}.npy"), cloud)

        # 位姿
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        pose = {"x": p.x, "y": p.y, "z": p.z,
                "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w,
                "stamp": round(img_stamp, 4)}
        with open(os.path.join(self.pose_dir, f"{prefix}.json"), "w") as f:
            json.dump(pose, f, indent=2)

        # 真值框 (几何投影, YOLO 归一化)
        T_wb = odom_to_T_world_base([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        lines = []
        for d in self.defects:
            box = build_gt_boxes(np.array(d["corners_world"]), T_wb, K, T_CAM_BASE, IMG_SIZE)
            if box:
                lines.append(f"{d['class_id']} {box[0]} {box[1]} {box[2]} {box[3]}")
        with open(os.path.join(self.label_dir, f"{prefix}.txt"), "w") as f:
            f.write("\n".join(lines))

        self._meta.append({
            "sample": prefix, "img_stamp": round(img_stamp, 4),
            "cloud_stamp": round(_stamp(pc_msg), 4), "odom_stamp": round(_stamp(odom_msg), 4),
            "pose": pose, "n_cloud": int(cloud.shape[0]), "n_gt": len(lines),
        })

    def _pc_to_base(self, msg: PointCloud2) -> np.ndarray:
        # read_points_numpy 要求整朵云所有字段同 dtype; gz gpu_lidar 的
        # intensity(FLOAT32)+ring(UINT16) 混型会炸。这里按字段 offset 步长手动解析,
        # 布局自适应、全矢量、无 Python 循环。字段布局(实测):
        #   x/y/z FLOAT32 @0/4/8, [4B pad], intensity FLOAT32 @16, ring UINT16 @24, point_step 32
        n = msg.width * msg.height
        ps = msg.point_step
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        off = {f.name: f.offset for f in msg.fields}
        # 把缓冲区看成 (n, ps) 字节矩阵, 每点一行; 字段列 [o, o+4) 即该字段的 float32
        rows = as_strided(buf, shape=(n, ps), strides=(ps, 1))

        def _f32(name: str):
            o = off.get(name)
            # view(float32) 结果形状 (n,1), 摊平成 (n,) 以便 stack 出 (n,3)
            return None if o is None else np.ascontiguousarray(rows[:, o:o + 4]).view(np.float32).reshape(-1)

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
        return np.concatenate([base, intensity.reshape(-1, 1)], axis=1).astype(np.float32)

    # ---------- 目录/元数据 ----------
    def _make_dirs(self):
        # 属性名必须与 _record_sample 里一致 (img_dir/cloud_dir/pose_dir/label_dir)
        for sub, attr in (("images", "img_dir"), ("clouds", "cloud_dir"),
                          ("poses", "pose_dir"), ("labels", "label_dir")):
            setattr(self, attr, os.path.join(self.out_dir, sub))
            os.makedirs(getattr(self, attr), exist_ok=True)
        self._meta = []

    def _write_meta(self):
        calib = {
            "K": K.tolist(), "T_cam_base": T_CAM_BASE.tolist(),
            "T_lidar_base": T_LIDAR_BASE.tolist(), "img_size": list(IMG_SIZE),
            "class_names": self.class_names,
            "notes": "图片以 BGR 存盘(cv2), 点云为车体系, 标签为 YOLO 归一化坐标",
        }
        with open(os.path.join(self.out_dir, "calibration.json"), "w") as f:
            json.dump(calib, f, indent=2)
        with open(os.path.join(self.out_dir, "metadata.json"), "w") as f:
            json.dump(self._meta, f, indent=2)
        import shutil
        _dst = os.path.join(self.out_dir, "defects.json")
        # run_data_gen.sh 已预拷贝 defects.json 到 out_dir 时 src==dst, 跳过避免 SameFileError
        if os.path.abspath(self.defects_json) != os.path.abspath(_dst):
            shutil.copy(self.defects_json, _dst)
        yaml = ("path: " + self.out_dir + "\n"
                "train: images\nval: images\ntest: images\n"
                f"nc: {len(self.class_names)}\n"
                "names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(self.class_names)))
        with open(os.path.join(self.out_dir, "data.yaml"), "w") as f:
            f.write(yaml)


def main(args=None):
    rclpy.init(args=args)
    node = PairedDataRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info("录制被中断")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
