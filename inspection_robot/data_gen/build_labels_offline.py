#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线生成 YOLO 真值框 + calibration.json + data.yaml (无 ROS/rclpy 依赖)
================================================================================
gz_record.py 只落 images/clouds/poses/metadata.json; 真值框由本脚本从
defects.json 的世界系病害角点 + 每帧里程位姿, 用与 record_paired_data.py 完全一致
的 build_gt_boxes(几何投影) 重算。不依赖点云, 从而采集器无需读缺陷清单。

读:
    --out_dir   采集输出目录 (含 poses/*.json, metadata.json)
    --defects   该 seed 的 defects.json (含 defects + class_names)
写:
    labels/{sample}.txt           YOLO "class cx cy w h" (归一化)
    calibration.json              K/T_cam_base/T_lidar_base/img_size/class_names
    data.yaml                     nc + names (取 defects.json 的 class_names)
    defects.json                  拷贝到 out_dir (训练侧读真值/类别用)

用法:
    python3 data_gen/build_labels_offline.py --out_dir DIR --defects data_gen/defects.json
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "data_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# 服务器无 pro/2 -> common 走内嵌默认 K/T (与 xacro 一致), 与采集端一致
from data_gen.common import (  # noqa: E402
    K, T_CAM_BASE, T_LIDAR_BASE, IMG_SIZE,
    odom_to_T_world_base, build_gt_boxes,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--defects", required=True,
                    help="该 seed 的 defects.json (含 defects + class_names)")
    args = ap.parse_args()

    with open(args.defects, encoding="utf-8") as f:
        manifest = json.load(f)
    defects = manifest["defects"]
    class_names = manifest["class_names"]

    meta_path = os.path.join(args.out_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        sys.exit(f"❌ 找不到 {meta_path}, 请先跑 gz_record.py")
    with open(meta_path, encoding="utf-8") as f:
        samples = json.load(f)

    label_dir = os.path.join(args.out_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    n_box = 0
    n_skip = 0
    for meta in samples:
        name = meta["sample"]
        pose = meta["pose"]
        T_wb = odom_to_T_world_base(
            [pose["x"], pose["y"], pose["z"]],
            [pose["qx"], pose["qy"], pose["qz"], pose["qw"]])
        lines = []
        for d in defects:
            box = build_gt_boxes(np.array(d["corners_world"]), T_wb,
                                 K, T_CAM_BASE, IMG_SIZE)
            if box:
                lines.append(f"{d['class_id']} {box[0]} {box[1]} {box[2]} {box[3]}")
        with open(os.path.join(label_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(lines))
        n_box += len(lines)
        n_skip += 1

    # calibration.json
    calib = {
        "K": K.tolist(), "T_cam_base": T_CAM_BASE.tolist(),
        "T_lidar_base": T_LIDAR_BASE.tolist(), "img_size": list(IMG_SIZE),
        "class_names": class_names,
        "notes": "图片以 BGR 存盘(cv2), 点云为车体系, 标签为 YOLO 归一化坐标",
    }
    with open(os.path.join(args.out_dir, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)

    # data.yaml
    yaml = ("path: " + os.path.abspath(args.out_dir) + "\n"
            "train: images\nval: images\ntest: images\n"
            f"nc: {len(class_names)}\n"
            "names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names)))
    with open(os.path.join(args.out_dir, "data.yaml"), "w") as f:
        f.write(yaml)

    # defects.json 拷贝 (训练侧读真值)
    dst = os.path.join(args.out_dir, "defects.json")
    if os.path.abspath(args.defects) != os.path.abspath(dst):
        shutil.copy(args.defects, dst)

    print(f"✅ 离线标签完成: {n_skip} 帧, {n_box} 框 -> {label_dir}")
    print(f"   calibration.json / data.yaml / defects.json 已写入 {args.out_dir}")


if __name__ == "__main__":
    main()
