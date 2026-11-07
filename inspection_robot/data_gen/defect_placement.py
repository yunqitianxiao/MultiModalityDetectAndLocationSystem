# -*- coding: utf-8 -*-
"""
病害布置配置 -> defects.json (配对数据的地面真值清单)
================================================================
确定性(seed)把七类病害"贴"到隧道内, 按类物理位置分三面:
    - 土建3类(裂缝/渗漏水/掉块)  -> 隧道两壁内表面 (y=±1.25, 竖直面贴墙, 法线朝 ±y)
    - 轨道3类(扣件断裂/缺失/松动歪斜) -> 两条钢轨轨顶 (y=±0.717, 小立牌朝行进方向, 法线朝 ±x)
    - 设备1类(异物入侵)          -> 轨道内地面区域 (y∈[-0.55,0.55], 障碍物朝行进方向, 法线朝 ±x)
每病害记录 world 坐标、尺寸、类、贴图路径 + 四角点(自动标注用)。

类序(写进最终 data.yaml): 0=裂缝 crack, 1=渗漏水 leakage, 2=扣件断裂 rail_fastener_broken,
                            3=扣件缺失 rail_fastener_missing, 4=扣件松动歪斜 rail_fastener_loose,
                            5=异物入侵 equip_intrusion, 6=管片破损掉块 spalling,
                            7=管线支架松脱 equip_bracket_loose  (全 8 类)
用法: python3 data_gen/defect_placement.py [seed] [每类病害数] [贴图目录]
      python3 data_gen/defect_placement.py 42 8 data_gen/textures_crack --classes crack --cap 16
"""
import argparse
import glob
import json
import os
import random

import numpy as np

# ---------------- 类元信息 ----------------
# surface: wall=墙面(x-z 平面) / rail=轨顶小立牌 / ground=地面障碍物(后两者 y-z 平面, 面朝 ±x)
# 真实尺度(相对 3m 高墙 / 1435mm 轨距): 裂缝 细长 可达 1~2.5m; 渗漏 大片水渍; 掉块 结构缺口。
# z_range 抬高到墙面中段(真实缺陷多在 0.5~2.4m 高度), 底部锚定由 generate_defects 处理。
CLASSES = {
    "crack": {          # 裂缝: 纵向(沿隧道轴 w 长, 沿墙高 h 矮)。★不能反过来!
        # 相机在隧道中线沿 +x 看, 贴在侧墙 y=±1.25 上的缺陷是斜视的:
        #   画面宽 ∝ fx·y·w/d²  (由 x 向长度经侧向偏置放大)
        #   画面高 ∝ fy·h/d      (由 z 向高度直接透视)
        # 原来配成 w=0.12~0.34 / h=0.75~1.80(细高的横向缝), 投影宽高比≈0.06,
        # 被 common.build_gt_boxes 的 "宽高比<0.15 丢弃"(侧视窄缝)全数滤掉
        # => 全数据集裂缝标注恒为空(8 类那批也一样)。对调长宽后每帧可检出 4~5 个框。
        "id": 0, "surface": "wall", "side_y": 1.25, "z_range": (0.50, 2.20),
        "size": (1.40, 2.60, 0.45, 0.75), "textures": "crack_{i}.png", "anchor": "bottom",
    },
    "leakage": {        # 渗漏水: 大片水渍, 近方形, 但不过大(避免成巨椭圆)
        "id": 1, "surface": "wall", "side_y": 1.25, "z_range": (0.60, 2.20),
        "size": (0.70, 1.40, 0.80, 1.70), "textures": "leakage_{i}.png",
    },
    "rail_fastener_broken": {
        "id": 2, "surface": "rail", "side_y": 0.717, "z_rail_top": 0.155,
        "size": (0.13, 0.19, 0.16, 0.26), "textures": "rail_fastener_broken_{i}.png",
    },
    "rail_fastener_missing": {
        "id": 3, "surface": "rail", "side_y": 0.717, "z_rail_top": 0.155,
        "size": (0.13, 0.19, 0.16, 0.26), "textures": "rail_fastener_missing_{i}.png",
    },
    "rail_fastener_loose": {
        "id": 4, "surface": "rail", "side_y": 0.717, "z_rail_top": 0.155,
        "size": (0.13, 0.19, 0.16, 0.26), "textures": "rail_fastener_loose_{i}.png",
    },
    "equip_intrusion": {
        # 轨道内异物: 尺寸偏小+靠边(|y|>=0.30), 避免相机(y=0)贴脸时张角填满视野整屏过曝白屏。
        "id": 5, "surface": "ground", "y_range": (0.30, 0.52), "z_ground": 0.01,
        "size": (0.25, 0.45, 0.25, 0.45), "textures": "equip_intrusion_{i}.png",
    },
    "spalling": {       # 土建掉块: 结构缺口, 成片剥落
        # 沿隧道轴 w 从 (0.40,0.90) 放宽到 (0.80,1.80): 侧墙斜视下画面宽 ∝ fx·y·w/d²,
        # 原尺寸在 8m 外宽高比跌破 build_gt_boxes 的 0.15 阈值被整片丢弃(实测 60% 空帧)。
        "id": 6, "surface": "wall", "side_y": 1.25, "z_range": (0.50, 2.10),
        "size": (0.80, 1.80, 0.40, 1.10), "textures": "spalling_{i}.png",
    },
    "equip_bracket_loose": {
        # 设备管线支架松脱: 隧道侧挂, 贴墙较合理(非轨内, 避开相机中线), 低处(支架多在下方)。
        "id": 7, "surface": "wall", "side_y": 1.25, "z_range": (0.30, 0.90),
        "size": (0.40, 0.70, 0.40, 0.70), "textures": "equip_bracket_loose_{i}.png",
    },
}
CLASS_NAMES = [c for c, m in sorted(CLASSES.items(), key=lambda kv: kv[1]["id"])]

WALL_Y = 1.25           # 墙内表面 y (左+1.25 / 右-1.25)
RAIL_Y = 0.717          # 钢轨中线 y (标准轨距 1435mm 的一半)
PLANE_T = 0.02          # 病害薄盒厚度(m)
X_RANGE = (2.0, 16.0)   # 沿隧道轴向布置范围
N_TEXTURES = 5          # 每类贴图张数
SEAM_LEN = 1.5          # 管片环/接缝大致间距(m), 语义摆放: 缺陷聚集在接缝附近
WALL_Z_MAX = 2.85       # 墙面可用上限(m), 保证缺陷盒不出墙顶

# 每类最多布置个数: 之前把贴图池全部铺上(每类~25张) -> 墙上"贴纸墙";
# 真实隧道缺陷稀疏, 每类只是少数几处。MAX_PER_CLASS 收敛到真实密度(已设 per_class 时据此兜底)。
MAX_PER_CLASS = {
    "crack": 5, "leakage": 3, "spalling": 4,
    "rail_fastener_broken": 6, "rail_fastener_missing": 6, "rail_fastener_loose": 6,
    "equip_intrusion": 3, "equip_bracket_loose": 3,
}


def _corners(center, w, h, surface):
    """世界系下病害矩形的 4 角。
    wall      : y=const 竖直面 (x-z 平面)
    rail/ground: x=const 朝行进方向竖直面 (y-z 平面)
    """
    cx, cy, cz = center
    if surface == "wall":
        return np.array([
            [cx - w / 2, cy, cz - h / 2],
            [cx + w / 2, cy, cz - h / 2],
            [cx + w / 2, cy, cz + h / 2],
            [cx - w / 2, cy, cz + h / 2],
        ], dtype=float)
    return np.array([
        [cx, cy - w / 2, cz - h / 2],
        [cx, cy + w / 2, cz - h / 2],
        [cx, cy + w / 2, cz + h / 2],
        [cx, cy - w / 2, cz + h / 2],
    ], dtype=float)


def _seam_sites(rng, x_lo, x_hi, n):
    """返回 n 个"接缝"x 位置, 让墙类缺陷语义上聚在结构接缝(而非均匀撒布)。"""
    if n <= 1:
        return [round(rng.uniform(x_lo, x_hi), 3)]
    return [round(x_lo + i * (x_hi - x_lo) / (n - 1), 3) for i in range(n)]


def _wall_z(meta, h, rng):
    """按缺陷高 h 算落在墙内(0.2~WALL_Z_MAX)的中心 z。
    anchor=bottom(裂缝): 底部近地面向上延伸; 其余居中于类高带 [lo,hi]。
    缺陷比类高带还大时, 盒身让中心留在类带内、仅钳制不出墙顶/地面, 而非跳到整墙中点。"""
    lo, hi = meta["z_range"]
    half = h / 2
    if meta.get("anchor") == "bottom":
        c = half + rng.uniform(0.0, 0.30)     # 底部近地面
    else:
        c = rng.uniform(lo, hi)               # 居中于类高带
    lo_z, hi_z = 0.20 + half, WALL_Z_MAX - half
    if lo_z <= hi_z:
        return min(max(c, lo_z), hi_z)
    return (0.20 + WALL_Z_MAX) / 2


def generate_defects(seed: int = 42, per_class: int = 8, textures_dir: str | None = None,
                     only_classes: list[str] | None = None,
                     cap_override: dict | None = None) -> list[dict]:
    """确定性生成病害清单。

    textures_dir 给定时, 从该目录 {类}_*.png 里均匀无重复抽样, 但每类个数被
    MAX_PER_CLASS 收敛(避免"全铺满->贴纸墙"), 每张恰好用一次; 否则按 per_class
    每类随机个数, 纹理在 1..N_TEXTURES 内随机(旧行为)。
    墙类缺陷: x 聚集到有限几个"接缝"附近、z 落在合理高度(裂缝底部锚定), 更接近真实病害分布。

    only_classes 给定时只布置这些类(单类裂缝用), 且 class_id 按选中子集**重新编号**(0..n-1),
    class_names 也只含选中类 —— 下游 nc=len(class_names) 自动变 1。
    cap_override: {类名: 上限}, 覆盖 MAX_PER_CLASS(单类时放宽每景实例数)。"""
    rng = random.Random(seed)
    sel = [c for c in CLASS_NAMES if only_classes is None or c in only_classes]
    local_id = {c: i for i, c in enumerate(sel)}
    defects = []
    for cls_name in sel:
        meta = CLASSES[cls_name]
        wmin, wmax, hmin, hmax = meta["size"]
        n_inst = per_class
        tex_run = None
        if textures_dir:
            n_tex = len(glob.glob(os.path.join(textures_dir, f"{cls_name}_*.png")))
            if n_tex > 0:
                cap = (cap_override or {}).get(cls_name, MAX_PER_CLASS.get(cls_name, per_class))
                n_inst = min(n_tex, cap)
                tex_run = rng.sample(range(1, n_tex + 1), n_inst)   # 无重复, 稀疏取
        # 语义: 墙类先把缺陷 x 聚到有限几个"接缝"附近
        seam_xs = _seam_sites(rng, *X_RANGE, max(1, n_inst)) if meta["surface"] == "wall" else None
        for k in range(n_inst):
            side = rng.choice([1.0, -1.0])          # 左+1 / 右-1
            w = rng.uniform(wmin, wmax)
            h = rng.uniform(hmin, hmax)
            surface = meta["surface"]
            if surface == "wall":
                y = side * meta["side_y"]
                x = seam_xs[k] + rng.uniform(-SEAM_LEN / 2, SEAM_LEN / 2)
                x = min(max(x, X_RANGE[0]), X_RANGE[1])
                z = _wall_z(meta, h, rng)
            elif surface == "rail":
                y = side * meta["side_y"]
                z = meta["z_rail_top"] + h / 2      # 底部贴轨顶
                x = rng.uniform(*X_RANGE)
            else:                                   # ground
                # 轨道内障碍物避开相机中线(y≈0): 薄板若正对相机, 近距离张角填满视野会整屏过曝白屏。
                # 复用已采样的 side 定左右, |y|∈meta["y_range"] 靠边, 相机从侧面掠过; rng 消费次数不变。
                y = side * rng.uniform(*meta["y_range"])
                z = meta["z_ground"] + h / 2        # 底部贴地面
                x = rng.uniform(*X_RANGE)
            tex_i = tex_run[k] if tex_run else rng.randint(1, N_TEXTURES)
            center = (round(x, 3), round(y, 3), round(z, 3))
            defects.append({
                "id": f"{cls_name}_{k + 1:02d}",
                "class_id": local_id[cls_name],
                "class_name": cls_name,
                "surface": surface,
                "center_world": list(center),
                "size_m": [round(w, 3), round(h, 3)],
                "side": "left" if side > 0 else "right",
                "texture": meta["textures"].format(i=tex_i),
                "corners_world": _corners(center, w, h, surface).tolist(),
            })
    # 稳定排序输出
    defects.sort(key=lambda d: d["id"])
    return defects


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", nargs="?", type=int, default=42)
    ap.add_argument("per_class", nargs="?", type=int, default=8)
    ap.add_argument("textures_dir", nargs="?", default=None)
    ap.add_argument("--classes", default=os.environ.get("DEFECT_CLASSES", ""),
                    help="逗号分隔的类名子集(只布置这些类, 如 crack); 默认全部 8 类")
    ap.add_argument("--cap", type=int, default=0,
                    help="覆盖 MAX_PER_CLASS 的每类上限(单类时放宽每景实例数)")
    args = ap.parse_intermixed_args()

    only = [c.strip() for c in args.classes.split(",") if c.strip()] or None
    bad = [c for c in (only or []) if c not in CLASSES]
    if bad:
        ap.error(f"未知类名 {bad}; 可用: {list(CLASSES)}")
    cap_override = {c: args.cap for c in (only or CLASS_NAMES)} if args.cap else None

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "defects.json")
    defects = generate_defects(args.seed, per_class=args.per_class,
                               textures_dir=args.textures_dir,
                               only_classes=only, cap_override=cap_override)
    class_names = [c for c in CLASS_NAMES if only is None or c in only]
    manifest = {
        "seed": args.seed,
        "class_names": class_names,
        "class_ids": {c: i for i, c in enumerate(class_names)},
        "defects": defects,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    n_cls = {}
    for d in defects:
        n_cls[d["class_name"]] = n_cls.get(d["class_name"], 0) + 1
    print(f"✅ 已写入 {out}")
    print(f"   seed={args.seed}, 每类 {args.per_class}, 类别 {class_names}, "
          f"病害总数 {len(defects)}, 分布 {n_cls}")
    print(f"   贴图目录: {args.textures_dir or os.path.join(here, 'textures_data1')} "
          f"(world 脚本需同步传入)")


if __name__ == "__main__":
    main()
