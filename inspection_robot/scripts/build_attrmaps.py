#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云 -> 稠密属性图 后处理 (编码器输入)
================================================================
把 record_paired_data.py 录制的车体系点云(clouds/*.npy)投影到相机平面,
生成三通道属性图并做最近邻补密, 供点云分支编码器(ResNet)直接读取:

    attrmaps/{sample}.npz   keys = ["depth", "reflec", "height", "mask"]   (H,W) float32 稠密
        depth  相机系深度(沿视轴, m)   —— 几何主干信息
        reflec 反射率 (归一化到 ~[0,1]) —— 材质/湿度信息(渗漏会反光变化)
        height 世界系高度 (m)          —— 帮助区分地面/墙体/病害位置
        mask   补密前有效区 (0/1)      —— 4th 通道, CNN 判"有点/无点", 不参与 fill_nearest

约定:
    - 点云已由录制节点转到车体系 [x,y,z,intensity]
    - 同步容差内机器人位姿漂移 ≤ vx*tol (0.4*0.15=0.06m), 直接按固定外参投影,
      这个有界漂移当作训练噪声, 不做逐帧精配准
    - 16 线天然稀疏(~99% 空洞), 补密前先统计空洞率写进 stats.json

用法:
    python3 scripts/build_attrmaps.py [out_dir]     # 默认 data_gen/output
    python3 scripts/build_attrmaps.py --summary-only
"""
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_gen.common import (  # noqa: E402
    K, T_CAM_BASE, IMG_SIZE, fill_nearest,
    lidar_to_image_attributes, odom_to_T_world_base,
)

H, W = IMG_SIZE[1], IMG_SIZE[0]


def world_z_of_cloud(cloud: np.ndarray, pose: dict) -> np.ndarray:
    """车体系点云 -> 世界系 z (高度图用)。"""
    T_wb = odom_to_T_world_base([pose["x"], pose["y"], pose["z"]],
                                [pose["qx"], pose["qy"], pose["qz"], pose["qw"]])
    Xw = (T_wb @ np.concatenate([cloud[:, :3], np.ones((len(cloud), 1))], axis=1).T).T
    return Xw[:, 2]


def build_for_sample(sample: str, out_dir: str) -> tuple:
    """处理单个样本, 返回 (holes_before: dict, n_valid_px: int) 统计。"""
    cloud = np.load(os.path.join(out_dir, "clouds", f"{sample}.npy"))
    with open(os.path.join(out_dir, "poses", f"{sample}.json"), encoding="utf-8") as f:
        pose = json.load(f)
    if len(cloud) == 0:
        raise ValueError(f"{sample}: 空点云, 跳过")

    # 世界高度 (编码器区分地面/墙/病害位置)
    zworld = world_z_of_cloud(cloud, pose)

    depth, reflec, height = lidar_to_image_attributes(
        cloud[:, :3], T_CAM_BASE, K, IMG_SIZE,
        reflectivity=cloud[:, 3], z_world=zworld,
    )
    n_valid = int(np.isfinite(depth).sum())
    # 4th 通道: 补密前有效区(有雷达点的像素=1, 空洞=0)。必须在 fill_nearest 之前捕获,
    # 因为 fill_nearest 后全图变有限值, 原始有效/空洞信息被毁。
    mask = np.isfinite(depth).astype(np.float32)

    # 反射率归一化: 原始强度可能无标定, 用当前帧 1%/99% 分位缩放稳定数值
    valid_r = reflec[np.isfinite(reflec)]
    if len(valid_r):
        lo, hi = np.percentile(valid_r, 1), np.percentile(valid_r, 99)
        if hi > lo:
            reflec = (reflec - lo) / (hi - lo)
            reflec = np.clip(reflec, 0.0, 1.0)

    # 空洞统计 (补密前); mask 无 NaN, 空洞率恒 0
    holes = {name: float(1.0 - np.isfinite(m).mean())
             for name, m in (("depth", depth), ("reflec", reflec), ("height", height), ("mask", mask))}

    # 最近邻补密: 编码器输入要稠密, 否则 CNN 感受野被空洞吃掉
    maps = {name: fill_nearest(m) for name, m in
            (("depth", depth), ("reflec", reflec), ("height", height))}
    maps["mask"] = mask                 # mask 已 0/1 有限, 不补密
    for m in maps.values():
        assert np.isfinite(m).all(), "补密后仍有非有限像素"

    out = os.path.join(out_dir, "attrmaps")
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f"{sample}.npz"), **maps)
    return holes, n_valid


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
        else os.path.join(_ROOT, "data_gen", "output")
    meta_path = os.path.join(out_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        sys.exit(f"❌ 找不到 {meta_path}, 请先跑 record_paired_data.py")

    with open(meta_path, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"属性图生成: {len(samples)} 帧 -> {out_dir}/attrmaps")

    stats = {"samples": [], "channels": ["depth", "reflec", "height", "mask"]}
    n_skip, n_fail = 0, 0
    for meta in samples:
        s = meta["sample"]
        try:
            holes, n_valid = build_for_sample(s, out_dir)
        except (ValueError, FileNotFoundError, KeyError) as e:
            n_fail += 1
            print(f"  ⚠ {s}: {e}")
            continue
        if n_valid == 0:
            n_skip += 1
            print(f"  - {s}: 投影无有效像素, 跳过")
            continue
        stats["samples"].append({"sample": s, "n_valid": n_valid, **holes})
        if (len(stats["samples"]) % 20) == 0:
            print(f"  ... {len(stats['samples'])}/{len(samples)}")

    # 汇总
    if stats["samples"]:
        for ch in stats["channels"]:
            holes = [x[ch] for x in stats["samples"]]
            stats[f"{ch}_hole_mean"] = float(np.mean(holes))
            stats[f"{ch}_hole_max"] = float(np.max(holes))
        stats["n_samples"] = len(stats["samples"])
        stats["n_skip"] = n_skip
        stats["n_fail"] = n_fail
        stats["img_size"] = list(IMG_SIZE)
        with open(os.path.join(out_dir, "attrmaps", "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        ch0 = stats["samples"][0]
        print(f"\n✅ 完成: {len(stats['samples'])} 帧补密图 (跳过 {n_skip}, 失败 {n_fail})")
        for ch in stats["channels"]:
            print(f"   {ch}: 补密前平均空洞率 {stats[f'{ch}_hole_mean']:.1%} "
                  f"(最大 {stats[f'{ch}_hole_max']:.1%})")
        print(f"   样例帧 {ch0['sample']}: 有效投影点 {ch0['n_valid']}")
        print(f"   输出: {os.path.join(out_dir, 'attrmaps')}/xxx.npz + stats.json")
    else:
        print("⚠ 没有成功生成的帧, 请检查录制数据 / 标定。")


if __name__ == "__main__":
    main()
