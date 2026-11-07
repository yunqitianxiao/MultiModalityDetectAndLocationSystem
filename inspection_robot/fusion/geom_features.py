#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云几何特征(M1)—— 从 attrmap 里"算"出每框的 3D 几何信号, 不学, 天然判别。

背景
====
attrmap 由 common.py 的 lidar_to_image_attributes 把车体系点云用 (K, T_cam_base)
投到跟图像同一平面(720x1280)。所以一张检测框 [u,v] 直接对应 attrmap 同一 [u,v] 的
三通道: depth(到相机距离)/reflec(反射率)/height(世界绝对 z)。

对轨道扣件:
  钢轨顶 z≈0.155m, 扣件顶 z≈0.327m => 扣件是一个 ~17cm 的凸起。
  缺失  => 凸起没了(框内只有轨顶平面, height 分布窄)。
  松动歪斜=> 上表面不再与轨面平齐(框内 height 有梯度 / 顶面法向偏)。
  断裂  => 高度剖面碎裂(height 分布散)。

本模块: 给一个检测框(归一化 xywh)算出几何特征, 用于 M2 点云存在性检测 和
         M3 低容量融合头。全部是"算"出来的, 没有可训参数。

用法(验证):
  .venv/bin/python fusion/geom_features.py --data data_gen/output_v11

快速接口:
  from fusion.geom_features import box_features, load_attrmap
"""
import os, glob, argparse
import numpy as np

# attrmap 通道顺序(H x W)
_KEYS = ("height", "depth", "reflec")


def load_attrmap(path):
    """读 .npz (存原始值) -> dict{'height','depth','reflec'}, 各 (H,W) float32。"""
    d = np.load(path)
    return {k: d[k] for k in _KEYS if k in d}


def _clip_tall(feat, y1, y2, x1, x2):
    y1, y2 = max(0, y1), min(feat.shape[0], y2)
    x1, x2 = max(0, x1), min(feat.shape[1], x2)
    return feat[y1:y2, x1:x2]


def box_features(attrmap, box, rail_top=0.155):
    """box: [cx,cy,w,h] 归一化(0-1) -> 该框的几何特征 dict。

    特征是"框内点云几何"的统计量, 与 2D 表观无关:
      h_min/h_max/h_med/h_std/h_p95    框内世界高度分布
      bump_p95   = h_p95 - rail_top     凸起量(扣件在不在的核心信号)
      spread     = h_p95 - h_med         框内高度离散(断裂/倾斜偏大)
      g_range    = p95-p05 range         剖面碎(断裂)则大
      density    = 有效深度像素占比(16线稀疏, 越满越好)
      depth_med  = 到相机距离(近/远可区分, 辅助)
      reflec_med = 反射率(缺陷薄盒 retro 值不同)
    """
    H, W = attrmap["height"].shape
    cx, cy, w, h = box
    w2, h2 = w * W / 2.0, h * H / 2.0
    x1, x2 = int(cx * W - w2), int(cx * W + w2)
    y1, y2 = int(cy * H - h2), int(cy * H + h2)

    hh = _clip_tall(attrmap["height"], y1, y2, x1, x2)
    dd = _clip_tall(attrmap["depth"], y1, y2, x1, x2)
    rr = _clip_tall(attrmap["reflec"], y1, y2, x1, x2)

    valid = dd > 0.0
    if valid.sum() < 5:           # 框内几乎无点云 -> 返回空几何
        return {"empty": True, "density": float(valid.mean()),
                "h_med": np.nan, "h_p95": np.nan, "bump_p95": np.nan,
                "spread": np.nan, "g_range": np.nan, "depth_med": np.nan,
                "reflec_med": np.nan}
    hv = hh[valid]
    p05, p50, p95 = np.quantile(hv, (0.05, 0.50, 0.95))
    return {
        "empty": False,
        "density": float(valid.mean()),
        "h_min": float(hv.min()), "h_max": float(hv.max()),
        "h_med": float(p50), "h_std": float(hv.std()),
        "h_p95": float(p95),
        "bump_p95": float(p95 - rail_top),      # 核心: 扣件凸起量
        "spread": float(p95 - p50),             # 倾斜/断裂 -> 大
        "g_range": float(p95 - p05),            # 剖面碎裂 -> 大
        "depth_med": float(float(np.median(dd[valid])) if valid.any() else np.nan),
        "reflec_med": float(float(np.median(rr[valid])) if valid.any() else np.nan),
    }


# ---------------- 验证: output_v11 上'缺失/倾斜/断裂 vs 正常'可否分 ----------------
def _demo(data_dir):
    att = glob.glob(os.path.join(data_dir, "attrmaps", "*"))
    lbl = os.path.join(data_dir, "labels")
    # 类: 0crack 1leak 2spall 3broken 4missing 5loose 6equip
    CLASS = {3: "broken", 4: "missing", 5: "loose"}
    stats = {c: [] for c in CLASS.values()}
    stats["normal_rail"] = []
    for att_p in att[:200]:
        stem = os.path.splitext(os.path.basename(att_p))[0]
        lp = os.path.join(lbl, stem + ".txt")
        if not os.path.exists(lp) or os.path.getsize(lp) == 0:
            continue
        am = load_attrmap(att_p)
        for line in open(lp):
            p = line.split()
            if len(p) != 5:
                continue
            c = int(p[0])
            if c in CLASS:
                f = box_features(am, [float(x) for x in p[1:5]])
                if not f["empty"]:
                    stats[CLASS[c]].append((f["bump_p95"], f["spread"], f["h_std"]))
        # 随取一个非病害框当"正常轨面"参照(框中点云密度>0.2 且非病害可用)
        # 简化: 用整帧 rail 高度带拆不出来, 这里不补, 靠 broken/loose vs missing 对照。
    print(f"{'类':<12}{'N':>4}{'bump_p95':>12}{'spread':>10}{'h_std':>9}")
    for k in ("broken", "missing", "loose"):
        a = np.array(stats[k]) if stats[k] else np.zeros((0, 3))
        if len(a) == 0:
            print(f"{k:<12}{0:>4}  (无样本)")
            continue
        print(f"{k:<12}{len(a):>4}"
              f"{a[:,0].mean():>12.3f}{a[:,1].mean():>10.3f}{a[:,2].mean():>9.3f}")
    # 缺失 vs 断裂/倾斜 的分离度
    miss = np.array(stats["missing"])[:, 0] if stats["missing"] else []
    pres = []
    for k in ("broken", "loose"):
        pres += stats[k]
    pres = np.array(pres)[:, 0] if pres else []
    if len(miss) and len(pres):
        print(f"\n[bump_p95] missing={miss.mean():.3f}  present(broken/loose)={pres.mean():.3f}"
              f"  分离度差={pres.mean()-miss.mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    _demo(args.data)
