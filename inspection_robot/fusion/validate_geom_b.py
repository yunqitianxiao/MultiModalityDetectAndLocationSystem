#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B 路(M2)前提验证: 点云几何特征在 output_big 上跨类是否可分
================================================================
对每帧 attrmap 的每个 GT 框算 box_features(M1 纯几何), 按类聚合分布,
看哪些特征能把类别区分开(决定 M2 过滤/补检用哪个特征+阈值)。

类(0-based, 与 data.yaml 一致):
  0 crack / 1 leakage / 2 rail_fastener_broken / 3 rail_fastener_missing /
  4 rail_fastener_loose / 5 equip_intrusion / 6 spalling / 7 equip_bracket_loose

用法:
  tunnel_venv/bin/python fusion/validate_geom_b.py --data <output_big> [--frames 400]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion.geom_features import load_attrmap, box_features  # noqa: E402

# 关键几何特征(语义见 geom_features.py): 凸起量/高度/反射/离散/密度
KEYS = ["bump_p95", "h_p95", "h_med", "spread", "g_range", "reflec_med", "density"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", type=int, default=400, help="最多处理的帧数")
    args = ap.parse_args()

    # 类名
    calib = os.path.join(args.data, "calibration.json")
    import json
    names = json.load(open(calib))["class_names"] if os.path.isfile(calib) else None

    att_files = sorted(glob.glob(os.path.join(args.data, "attrmaps", "*.npz")))[: args.frames]
    # 类 -> 每类特征列表(list of dict)
    agg = {}
    per_cls = {i: {k: [] for k in KEYS} for i in range(8 if names and len(names) == 8 else 9)}
    n_empty = {i: 0 for i in per_cls}
    n_box = {i: 0 for i in per_cls}

    for apath in att_files:
        stem = os.path.splitext(os.path.basename(apath))[0]
        lp = os.path.join(args.data, "labels", stem + ".txt")
        if not os.path.isfile(lp):
            continue
        am = load_attrmap(apath)
        for line in open(lp, encoding="utf-8"):
            p = line.split()
            if len(p) != 5:
                continue
            c = int(p[0])
            if c not in per_cls:
                continue
            f = box_features(am, [float(x) for x in p[1:5]])
            n_box[c] += 1
            if f.get("empty", True):
                n_empty[c] += 1
                continue
            for k in KEYS:
                v = f.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    per_cls[c][k].append(v)
        n_boxes = sum(1 for _ in open(lp, encoding="utf-8"))  # noqa: F841

    hdr = f"{'类':<20}{'N框':>5}{'空':>4}{'bump95':>9}{'h_p95':>9}{'h_med':>9}{'spread':>8}{'reflec':>8}{'den':>6}"
    print(hdr)
    print("-" * len(hdr))
    for c in sorted(per_cls):
        nm = names[c] if names and c < len(names) else str(c)
        row = [f"{nm:<20}", f"{n_box[c]:>5}", f"{n_empty[c]:>4}"]
        for k in KEYS:
            vals = per_cls[c][k]
            if not vals:
                row.append("   nan")
            else:
                row.append(f"{np.median(vals):>9.3f}")
        print("".join(row))

    # 关键分离检查
    def med(c, k):
        vals = per_cls.get(c, {}).get(k, [])
        return np.median(vals) if vals else float("nan")

    if names and set(range(5)) & set(range(8)) == set():  # 占位, 仅在类名齐全时
        pass
    print("\n=== 分离检查 ===")
    if len(names) == 8:
        fb_k = [2, 4]     # broken, loose (有凸起)
        miss = 3          # missing (无凸起)
        eb = [np.median(per_cls[c]["bump_p95"]) for c in fb_k if per_cls[c]["bump_p95"]]
        eb = np.mean(eb) if eb else float("nan")
        em = med(miss, "bump_p95")
        print(f"[扣件bump95] 缺失={em:.3f}  断裂/松动={eb:.3f}  差={eb-em:+.3f} (越大越可分)")
        print(f"[装备h_p95 ] intrusion={med(5,'h_p95'):.3f}  bracket={med(7,'h_p95'):.3f}  vs crack={med(0,'h_p95'):.3f}")
        print(f"[渗漏reflec] leakage={med(1,'reflec_med'):.3f}  vs crack={med(0,'reflec_med'):.3f}  (越高越异常受检)")


if __name__ == "__main__":
    main()
