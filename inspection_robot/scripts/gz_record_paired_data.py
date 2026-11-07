#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 gz 配对数据录制 (无 ROS / 无 ros_gz_bridge, 服务器直采)
================================================================
直接订阅 gz 话题流, 驱动小车往返, 保存与 record_paired_data.py 同格式的数据:
    images/xxx.jpg    RGB 相机图 (1280x720, BGR 存盘)
    clouds/xxx.npy    (N,4) 车体系点云 [x,y,z,intensity]
    poses/xxx.json    里程计位姿 + 时间戳
    labels/xxx.txt    YOLO 格式自动标注 (由病害世界坐标几何投影)
    metadata.json / calibration.json / data.yaml / defects.json

话题(纯 gz):
    /camera                          gz.msgs.Image (base64 RGB8)
    /model/inspection_robot/odometry gz.msgs.Odometry
    /lidar/points/points             gz.msgs.PointCloudPacked
    /model/inspection_robot/cmd_vel  gz.msgs.Twist (驱动)

用法:
    python3 scripts/gz_record_paired_data.py [--out OUT_DIR] [--vx 0.4]
        [--t-forward 40] [--sample-interval 0.5] [--defects data_gen/defects.json]
"""
import argparse
import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

# 复用 data_gen.common 的几何标定 (K / T_CAM_BASE / T_LIDAR_BASE / build_gt_boxes)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "data_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from data_gen.common import (  # noqa: E402
    K, T_CAM_BASE, T_LIDAR_BASE, IMG_SIZE,
    odom_to_T_world_base, build_gt_boxes,
)

CAM_TOPIC = "/camera"
ODOM_TOPIC = "/model/inspection_robot/odometry"
LIDAR_TOPIC = "/lidar/points/points"
TWIST_TOPIC = "/model/inspection_robot/cmd_vel"


def _msg_generator(topic: str, timeout: float = 3.0):
    """yield gz 话题每条 JSON 消息。用 Popen + 后台读线程, 读不到就空转等待。"""
    proc = subprocess.Popen(
        ["gz", "topic", "-e", "-t", topic, "--json-output"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        bufsize=1, text=True,
    )
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    finally:
        proc.kill()


def _stamp(header: dict) -> float:
    st = header.get("stamp", {})
    return st.get("sec", 0) + st.get("nsec", 0) * 1e-9


class TopicReader(threading.Thread):
    """后台线程持续读一个 gz 话题, 丢进队列。"""

    def __init__(self, topic: str, q: queue.Queue, max_q: int = 30):
        super().__init__(daemon=True)
        self.topic, self.q, self.max_q = topic, q, max_q
        self.last_err = None

    def run(self):
        try:
            for m in _msg_generator(self.topic):
                try:
                    self.q.put((time.monotonic(), m))
                    if self.q.qsize() > self.max_q:
                        self.q.get()
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001
            self.last_err = e


def _img_to_bgr(m: dict) -> np.ndarray:
    w, h, step = int(m["width"]), int(m["height"]), int(m["step"])
    a = np.frombuffer(base64.b64decode(m["data"]), dtype=np.uint8).reshape(h, step)[:, : w * 3]
    return a.reshape(h, w, 3)[:, :, ::-1].copy()          # RGB8 -> BGR


def _cloud_to_base(m: dict) -> np.ndarray:
    """PointCloudPacked -> (N,4) 车体系 [x,y,z,intensity]。字段按 offset 解析。"""
    n = int(m["width"]) * int(m["height"])
    ps = int(m["point_step"])
    buf = np.frombuffer(base64.b64decode(m["data"]), dtype=np.uint8)
    off = {f["name"]: int(f["offset"]) for f in m["fields"]}
    xyz_i = np.zeros((n, 4), dtype=np.float32)
    for i, name in enumerate(("x", "y", "z", "intensity")):
        o = off.get(name)
        if o is None:
            if name == "intensity":
                xyz_i[:, 3] = 1.0
            continue
        col = buf.reshape(n, ps)[:, o:o + 4]
        xyz_i[:, i] = np.ascontiguousarray(col).view(np.float32).reshape(-1)
    ok = np.isfinite(xyz_i[:, :3]).all(axis=1)
    xyz_i = xyz_i[ok]
    X4 = np.concatenate([xyz_i[:, :3], np.ones((len(xyz_i), 1))], axis=1).T
    base = (T_LIDAR_BASE @ X4).T[:, :3]
    return np.concatenate([base, xyz_i[:, 3:]], axis=1).astype(np.float32)


def _nearest(buf: dict, t: float, tol: float):
    if not buf:
        return None
    key = min(buf, key=lambda k: abs(k - t))
    return buf[key] if abs(key - t) <= tol else None


def _send_twist(vx: float):
    v = round(float(vx), 4)
    subprocess.run(
        ["gz", "topic", "-t", TWIST_TOPIC, "-m", "gz.msgs.Twist",
         "-p", f"linear: {{x: {v}}}"],
        capture_output=True, timeout=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "data_gen", "output_gz"))
    ap.add_argument("--vx", type=float, default=0.4)
    ap.add_argument("--t-forward", type=float, default=40.0)
    ap.add_argument("--sample-interval", type=float, default=0.5)
    ap.add_argument("--sync-tol", type=float, default=0.15)
    ap.add_argument("--defects", default=os.path.join(_ROOT, "data_gen", "defects.json"))
    args = ap.parse_args()

    out = args.out
    for sub in ("images", "clouds", "poses", "labels"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    with open(args.defects, encoding="utf-8") as f:
        manifest = json.load(f)
    defects = manifest["defects"]
    class_names = manifest["class_names"]

    # 启动三个话题流
    q_img, q_odom, q_lid = queue.Queue(), queue.Queue(), queue.Queue()
    for topic, q in ((CAM_TOPIC, q_img), (ODOM_TOPIC, q_odom), (LIDAR_TOPIC, q_lid)):
        r = TopicReader(topic, q)
        r.start()
        print(f"[collector] 订阅 {topic}")

    # 等待数据齐
    def drain(q):
        m = None
        try:
            while True:
                m = q.get_nowait()[1]
        except queue.Empty:
            return m

    print("[collector] 等待话题数据...")
    for _ in range(60):
        time.sleep(0.5)
        if not q_img.empty() and not q_odom.empty() and not q_lid.empty():
            break
    else:
        print("!!! 超时: 话题无数据, 检查 gz sim 是否在跑 (world 名/话题是否一致)")
        sys.exit(1)

    def purge(q):
        while not q.empty():
            q.get()

    # 计时用 sim 时间: 以首帧图像 stamp 为基准
    _, m0 = q_img.get()
    sim_t0 = _stamp(m0.get("header", {}) or {})
    if not sim_t0:
        sim_t0 = 0.0
    start_wall = time.monotonic()
    last_sample_wall = None
    n_samples = 0
    odom_buf = {}
    meta = []

    def sim_now():
        return sim_t0 + (time.monotonic() - start_wall)

    print(f"[collector] 数据就绪 (sim t0={sim_t0:.2f}s), 开始行驶 vx={args.vx} t_forward={args.t_forward}x2")
    purge(q_img); purge(q_odom); purge(q_lid)
    _send_twist(args.vx)

    finished = False
    while not finished:
        # 驱动相位 (用墙钟, 与 sim 实时率一致)
        elapsed = time.monotonic() - start_wall
        if elapsed < args.t_forward:
            vx = args.vx
        elif elapsed < 2 * args.t_forward:
            vx = -args.vx
        else:
            vx = 0.0
            finished = True
        _send_twist(vx)

        # 收 odom / lidar
        while not q_odom.empty():
            _, m = q_odom.get()
            h = m.get("pose", {}).get("header", {})
            odom_buf[_stamp(h)] = m
        while not q_lid.empty():
            _, m = q_lid.get()

        # 采样: 取最新图像
        now = sim_now()
        img_m = None
        if not q_img.empty():
            _, img_m = q_img.get()
        if img_m is not None and (last_sample_wall is None
                                  or time.monotonic() - last_sample_wall >= args.sample_interval):
            last_sample_wall = time.monotonic()
            img_stamp = _stamp(img_m.get("header", {}) or {})
            odom_m = _nearest(odom_buf, img_stamp, args.sync_tol)
            if odom_m is None:
                print(f"[collector] 同步失败 @{now:.2f} odom 无 (stamp={img_stamp:.2f})")
                continue
            # 清理旧 odom
            for k in [k for k in odom_buf if k < img_stamp - 1.0]:
                odom_buf.pop(k, None)

            prefix = f"{n_samples:06d}"
            n_samples += 1
            img = _img_to_bgr(img_m)
            try:
                cv2_im = __import__("cv2")
                cv2_im.imwrite(os.path.join(out, "images", f"{prefix}.jpg"), img)
            except Exception as e:  # noqa: BLE001
                print(f"  !! 存图失败: {e}")

            # 点云 (gpu_lidar 每帧都发, 直接用最新)
            lid_m = None
            if not q_lid.empty():
                _, lid_m = q_lid.get()
            if lid_m is not None:
                cloud = _cloud_to_base(lid_m)
                np.save(os.path.join(out, "clouds", f"{prefix}.npy"), cloud)
            else:
                cloud = np.zeros((0, 4), dtype=np.float32)

            # 位姿
            p = odom_m["pose"].get("position", {})
            o = odom_m["pose"].get("orientation", {})
            pose = {"x": p.get("x", 0), "y": p.get("y", 0), "z": p.get("z", 0),
                    "qx": o.get("x", 0), "qy": o.get("y", 0), "qz": o.get("z", 0),
                    "qw": o.get("w", 1), "stamp": round(img_stamp, 4)}
            with open(os.path.join(out, "poses", f"{prefix}.json"), "w") as f:
                json.dump(pose, f, indent=2)

            # 真值框
            T_wb = odom_to_T_world_base(
                [pose["x"], pose["y"], pose["z"]], [pose["qx"], pose["qy"], pose["qz"], pose["qw"]])
            lines = []
            for d in defects:
                box = build_gt_boxes(np.array(d["corners_world"]), T_wb, K, T_CAM_BASE, IMG_SIZE)
                if box:
                    lines.append(f"{d['class_id']} {box[0]} {box[1]} {box[2]} {box[3]}")
            with open(os.path.join(out, "labels", f"{prefix}.txt"), "w") as f:
                f.write("\n".join(lines))

            meta.append({
                "sample": prefix, "img_stamp": round(img_stamp, 4),
                "pose": pose, "n_cloud": int(cloud.shape[0]), "n_gt": len(lines),
            })
            if n_samples % 10 == 0:
                print(f"[collector] {n_samples} 帧 @{now:.1f}s robot=({pose['x']:.1f},{pose['y']:.1f})")

    _send_twist(0.0)
    # 元数据
    calib = {
        "K": K.tolist(), "T_cam_base": T_CAM_BASE.tolist(),
        "T_lidar_base": T_LIDAR_BASE.tolist(), "img_size": list(IMG_SIZE),
        "class_names": class_names,
        "notes": "纯 gz 录制; 图片以 BGR 存盘, 点云为车体系, 标签为 YOLO 归一化",
    }
    with open(os.path.join(out, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out, "defects.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {out}\ntrain: images\nval: images\ntest: images\n"
                f"nc: {len(class_names)}\nnames:\n"
                + "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names)))

    print(f"\n🎉 录制完成: {out}  共 {n_samples} 帧")
    print(f"    标签统计: {meta[-1]['n_gt'] if meta else 0} GT/帧")


if __name__ == "__main__":
    main()
