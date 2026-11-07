#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成"设备管线支架松脱"贴图渲染世界 (无真实照片, 仿真合成)
================================================================
隧道侧壁一段钢管(沿隧道轴向 y 走) + 双支脚抱箍(压板+两螺杆+双螺母) + 角钢支架。
5 档松脱变体沿 y 一字排开, 每档一台相机正对拍摄 -> 5 张 1024x1024 近景,
后处理裁 512x512 存为贴图。

"松脱"可见判据(照片里一眼能认出):
  - 角钢支架整体歪斜: 绕 x 轴 roll, 以左侧为锚点, 右侧(图像右)下沉
  - 右侧螺母悬空: 支架右端下沉后, 右螺母下方留出大缝(螺栓脱出感)
  - 左螺母沿螺杆松出(螺纹露出) / 缺失(变体5)
相机: gz-sim8 相机默认沿自身 +z(朝上), 必须 pose pitch=-90° 才朝 +x 看墙。
      画面竖直方向由渲染侧决定, 若上下颠倒在后处理翻 180°。
材质: 纯 diffuse+emissive(与隧道墙面同款, Ogre2 必渲染), 不依赖 PBR 贴图。

用法: python3 scripts/gen_bracket_tex_world.py [输出.sdf] [世界名]
"""
import math
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bracket_tex_world.sdf"
WNAME = sys.argv[2] if len(sys.argv) > 2 else "bracket_tex_world"

# ---------------- 场景常量 ----------------
WALL_X = 0.0            # 墙前表面 x=0 (墙心在 x=+0.1, 厚 0.2 -> 前表面正好 x=0)
WALL_CX = 0.1
PIPE_X = -0.13          # 管轴 x (突出墙前表面 75mm, 管径 0.055)
PIPE_R = 0.055          # 管半径 (DN110)
CAM_DIST = 0.55         # 相机到支架前方距离
CAM_IM = 1024           # 相机分辨率
HFOV = 1.047            # 60°
PZ = 1.15               # 管轴 z
PB = PZ - PIPE_R        # 管底 z = 1.095

# 5 档松脱变体:
#   roll  : 角钢绕 x 歪斜角(负=图像右侧下沉), 锚点=支架左侧
#   nutR  : 右螺母再抬高(mm)  -> 右缝更大
#   nutL  : 左螺母沿螺杆松出(mm) -> 螺纹露出
#   missL : 左螺母缺失
VARIANTS = [
    {"roll": -2,  "nutR": 0,  "nutL": 0},            # 1 轻微: 支架略歪
    {"roll": -5,  "nutR": 3,  "nutL": 2},            # 2 松动: 歪斜+右缝+左螺母松出
    {"roll": -8,  "nutR": 6,  "nutL": 5},            # 3 明显: 一侧下沉, 螺纹露出
    {"roll": -11, "nutR": 10, "nutL": 9},            # 4 严重: 右螺母大缝, 左螺母近脱出
    {"roll": -14, "nutR": 16, "nutL": 0, "missL": True},  # 5 危急: 右螺母脱出, 左螺母缺失
]


def MAT(alb, emis):
    a = " ".join(map(str, alb))
    return (f"          <material>\n"
            f"            <ambient>{a} 1</ambient>\n"
            f"            <diffuse>{a} 1</diffuse>\n"
            f"            <emissive>{emis} {emis} {emis} 1</emissive>\n"
            f"            <specular>0.05 0.05 0.05 1</specular>\n"
            f"          </material>")

# 高对比配色: 亮混凝土墙为背景, 深灰角钢形成清晰剪影, 暖色金属管 + 亮银螺栓/螺母突出"松脱"特征
WALL_MAT  = MAT((0.66, 0.66, 0.68), 0.10)
PIPE_MAT  = MAT((0.46, 0.31, 0.20), 0.06)   # 暗色金属管, 避免过曝成白
BRKT_MAT  = MAT((0.14, 0.16, 0.19), 0.03)   # 深色钢角钢 -> 亮墙上成剪影
BOLT_MAT  = MAT((0.52, 0.54, 0.57), 0.10)   # 中等银灰压板/螺杆/螺母


def vis(name, geo, mat, pose="0 0 0 0 0 0"):
    return (f"      <visual name='{name}'>\n"
            f"        <pose>{pose}</pose>\n"
            f"{geo}\n{mat}      </visual>")


def geom_box(s):
    return f"        <geometry><box><size>{' '.join(map(str, s))}</size></box></geometry>"


def geom_cyl(r, l):
    return f"        <geometry><cylinder><radius>{r}</radius><length>{l}</length></cylinder></geometry>"


def model(name, pose, visuals):
    v = "\n".join(f"    <link name='{n}'>\n{vi}</link>" for n, vi in visuals)
    return f"  <model name='{name}'>\n    <static>true</static>\n    <pose>{pose}</pose>\n{v}\n  </model>\n"


def pipe_model(y0, pipe_len=4.0):
    """沿 y 的钢管段(圆柱默认沿 z, 绕 x 转 π/2 -> 沿 y)。"""
    return model(f"pipe_{y0:+.0f}", f"{PIPE_X} {y0} {PZ} 1.5708 0 0",
                 [("body", vis("body", geom_cyl(PIPE_R, pipe_len), PIPE_MAT))])


def bracket_models(y0, v):
    """角钢支架(歪斜) + 抱箍压板 + 双螺杆 + 双螺母。"""
    roll = math.radians(v["roll"])
    nutR = v["nutR"] / 1000.0
    nutL = v["nutL"] / 1000.0
    models = []

    # --- 角钢支架: 左侧锚点(y=-0.10)绕 x 转 roll, 右侧下沉 ---
    # 模型原点放在锚点 (PIPE_X+0.02, -0.10, PB); 子件局部坐标
    shelf_cx = 0.0                # 局部 x (支架在墙前)
    shelf_cy = 0.10               # 局部 y (从锚点 y=-0.10 到中心 y=0)
    shelf_cz = -0.006             # 局部 z: 顶面 = 管底(承托位)
    strap_cx = PIPE_X + 0.055 - (PIPE_X + 0.02)   # 贴墙竖板局部 x
    strap_cz = -(PB - (PZ + 0.08))                # 竖板中心(向上到压板顶)
    angle_vis = [
        vis("shelf", geom_box((0.07, 0.20, 0.012)),
            BRKT_MAT, f"{shelf_cx} {shelf_cy} {shelf_cz} 0 0 0"),
        vis("strap", geom_box((0.012, 0.12, 0.16)),
            BRKT_MAT, f"{strap_cx} {shelf_cy} {strap_cz} 0 0 0"),
    ]
    models.append(model(f"angle_{y0:+.0f}",
                        f"{PIPE_X + 0.02} {y0 - 0.10} {PB} {roll:.6f} 0 0",
                        [("shelf", angle_vis[0]), ("strap", angle_vis[1])]))

    # --- 抱箍: 压板(管上方) + 两螺杆 + 两螺母(螺母在管底下方/松出) ---
    y_stud = 0.075
    stud_top = PZ + 0.065
    stud_bot = PB - 0.022
    stud_len = stud_top - stud_bot + 0.01
    stud_cz = (stud_top + stud_bot) / 2
    clamp_vis = [
        vis("clamp", geom_box((0.05, 0.16, 0.008)),
            BOLT_MAT, f"{PIPE_X} 0 {stud_top} 0 0 0"),
        vis("stud_L", geom_cyl(0.0045, stud_len),
            BOLT_MAT, f"{PIPE_X} {-y_stud} {stud_cz} 0 0 0"),
        vis("stud_R", geom_cyl(0.0045, stud_len),
            BOLT_MAT, f"{PIPE_X} {y_stud} {stud_cz} 0 0 0"),
    ]
    # 螺母: 管底下方(正常支撑位), 右螺母照常低置, 左螺母可松出/缺失
    nut_cz = PB - 0.012
    vis_nuts = [("clamp", clamp_vis[0]), ("stud_L", clamp_vis[1]), ("stud_R", clamp_vis[2])]
    if not v.get("missL"):
        vis_nuts.append(("nut_L", vis("nut_L", geom_cyl(0.009, 0.014),
                                      BOLT_MAT, f"{PIPE_X} {-y_stud} {nut_cz + nutL} 0 0 0")))
    # 右螺母: 抬得更高 -> 与下沉的支架右端间缝更大
    vis_nuts.append(("nut_R", vis("nut_R", geom_cyl(0.009, 0.014),
                                  BOLT_MAT, f"{PIPE_X} {y_stud} {nut_cz + nutR} 0 0 0")))
    models.append(model(f"clamp_{y0:+.0f}", f"0 0 0 0 0 0", vis_nuts))
    return models


def camera_model(idx, y0):
    # gz-sim8 相机 sensor 视向沿自身 +X 轴(2026-08-19 实测, 非 ±Z)。
    # pitch=0 时 view=(1,0,0)=朝 +x 看前方墙上的支架; pitch=-90° 会朝上照天花板。
    # 相机在 x=-CAM_DIST, 支架在 x≈-0.13 前方, pitch 0 正对。竖直方向随渲染侧, 不必翻转。
    return f"""  <model name='cam_{idx}'>
    <static>true</static>
    <pose>{-CAM_DIST} {y0} {PZ} 0 0 0</pose>
    <link name='link'>
      <visual name='v'>
        <geometry><box><size>0.02 0.02 0.02</size></box></geometry>
      </visual>
      <sensor type='camera' name='cam_{idx}'>
        <always_on>true</always_on>
        <update_rate>10</update_rate>
        <topic>/cam_tex_{idx}</topic>
        <camera>
          <horizontal_fov>{HFOV}</horizontal_fov>
          <image>
            <width>{CAM_IM}</width>
            <height>{CAM_IM}</height>
          </image>
        </camera>
      </sensor>
    </link>
  </model>
"""


def build_world():
    parts = []
    parts.append(model("wall_panel", f"{WALL_CX} 0 1.5 0 0 0",
                       [("wall", vis("wall", geom_box((0.2, 26, 3.0)), WALL_MAT))]))
    parts.append(model("ground_panel", "-2 0 -0.05 0 0 0",
                       [("g", vis("g", geom_box((4, 26, 0.1)), WALL_MAT))]))
    y_positions = [-1.5, -0.75, 0, 0.75, 1.5]
    for idx, (y0, v) in enumerate(zip(y_positions, VARIANTS), 1):
        parts.append(pipe_model(y0))
        parts.extend(bracket_models(y0, v))
        parts.append(camera_model(idx, y0))
    scene = ("    <scene>\n"
             "      <ambient>0.30 0.30 0.32</ambient>\n"
             "      <background>0.08 0.08 0.10</background>\n"
             "      <shadows>false</shadows>\n"
             "    </scene>\n"
             "    <plugin filename=\"gz-sim-physics-system\" name=\"gz::sim::systems::Physics\">\n"
             "    </plugin>\n"
             "    <plugin filename=\"gz-sim-sensors-system\" name=\"gz::sim::systems::Sensors\">\n"
             "      <render_engine>ogre2</render_engine>\n"
             "    </plugin>\n")
    # 每个支架位一盏灯, 保证所有变体均匀照明(远离中心那两档不再偏暗)
    lights = []
    for i, (y0, v) in enumerate(zip(y_positions, VARIANTS), 1):
        lights.append(
            f"    <light type=\"point\" name=\"tex_light_{i}\">\n"
            f"      <cast_shadows>false</cast_shadows>\n"
            f"      <pose>-0.8 {y0} 1.4 0 0 0</pose>\n"
            f"      <diffuse>0.5 0.5 0.5</diffuse>\n"
            f"      <specular>0.1 0.1 0.1</specular>\n"
            f"      <attenuation><range>8</range><constant>1.0</constant><linear>0.05</linear><quadratic>0.005</quadratic></attenuation>\n"
            f"      <intensity>1.0</intensity>\n"
            f"    </light>\n")
    light = "".join(lights)
    world = (f'<?xml version="1.0"?>\n<sdf version="1.9">\n'
             f'  <world name="{WNAME}">\n{scene}{light}{"".join(parts)}  </world>\n</sdf>\n')
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(world)
    print(f"✅ 已生成 {OUT} ({len(VARIANTS)} 档松脱变体)")


if __name__ == "__main__":
    build_world()
