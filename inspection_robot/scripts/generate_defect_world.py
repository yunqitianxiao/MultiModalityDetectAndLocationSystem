#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成带病害的隧道世界文件 (配对数据用)
================================================================
流程:
    1) 读取 inspection_tunnel_base.world (含 <!--__ROBOT__--> 占位符)
    2) 去掉"特征物"(彩色方块, 避免干扰病害检测)
    3) 按 data_gen/defects.json 把病害薄盒(PBR albedo 贴图)注入墙内表面
    4) 复用 generate_world.py 逻辑把机器人模型(URDF->SDF)内嵌
    5) 输出 worlds/defect_tunnel.world

用法: python3 scripts/generate_defect_world.py [defects.json] [输出世界文件]
注: 贴图用绝对路径, 移动项目目录后需重跑本脚本再启动仿真。
"""
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_WORLD = os.path.join(ROOT, "worlds", "inspection_tunnel_base.world")
DEFECTS_JSON = os.path.join(ROOT, "data_gen", "defects.json")
OUT_WORLD = os.path.join(ROOT, "worlds", "defect_tunnel.world")
URDF_XACRO = os.path.join(ROOT, "urdf", "inspection_robot.urdf.xacro")
ROBOT_POSE = "0 0 0.20 0 0 0"
PLANE_T = 0.02          # 轨/地病害薄盒厚度(沿 x, 面朝 ±x), 立牌/障碍物需一定厚度
WALL_THICK = 0.004      # 墙病害厚度(沿 y): 更薄更贴花。单靠薄盒+墙色羽化(贴图边=墙 albedo),
                        # 盒边/阴影几乎不可见, 消掉"粘板感"。

# 掉块剥落: 真实病害是墙面凹陷, 但隧道墙是实心整段、无法挖洞, 故用向外凸出的盒体
# 代替, 让 gpu_lidar 能测到厘米级距离差。定位是"点云支在有几何差异时能否吃到信号"
# 的受控实验, 不代表真实凹陷形态。其余墙病害仍用 WALL_THICK。
DEPTH_BY_CLASS = {"spalling": 0.12}


def run(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print("执行失败:", cmd)
        print(r.stderr[-2000:])
        sys.exit(1)
    return r.stdout


def robot_model_sdf() -> str:
    """xacro -> URDF -> SDF, 提取 <model> 并加初始位姿。"""
    urdf = run(f"xacro {URDF_XACRO}")
    tmp = "/tmp/_robot_generated.urdf"
    with open(tmp, "w") as f:
        f.write(urdf)
    sdf = run(f"gz sdf -p {tmp}")
    m = re.search(r"<model[\s\S]*?</model>", sdf)
    if not m:
        sys.exit("错误: 未找到 <model> 元素")
    model = re.sub(r"<model([^>]*)>", r"<model\1>\n    <pose>%s</pose>" % ROBOT_POSE,
                   m.group(0), count=1)
    return model


# gz-sim gpu_lidar 的 intensity 通道 = 命中材质的 laser_retro (默认 0)。
# 给缺陷按类设 retro, 让属性图 reflec 通道成为"缺陷掩膜" (墙=0, 缺陷=类相关高值)。
# 官方示例: gz-sim8/worlds/gpu_lidar_retro_values_sensor.sdf
RETRO_BY_CLASS = {
    "crack": 500, "leakage": 300, "spalling": 700,
    "rail_fastener_broken": 450, "rail_fastener_missing": 520,
    "rail_fastener_loose": 380, "equip_intrusion": 900, "equip_bracket_loose": 760,
}

# 病害薄盒自发光(暗环境下可见)。注意: 平坦发光项会把贴图的暗色病害特征
# (裂缝/渗漏水渍等, 贴图均值仅 67~194)整体推入近白色带, 细节被冲淡 -> 盒子"发白"。
# PBR albedo_map 生效后 0.05 已足够在点光源间隙里保持可辨, 又不压暗纹理细节。
EMISSIVE_BY_CLASS = {
    "crack": 0.05, "leakage": 0.05, "spalling": 0.05,
    "rail_fastener_broken": 0.05, "rail_fastener_missing": 0.05,
    "rail_fastener_loose": 0.05, "equip_intrusion": 0.05, "equip_bracket_loose": 0.05,
}


def defect_model_xml(d: dict, textures_dir: str) -> str:
    """病害薄盒 SDF 模型。贴图绝对路径 -> PBR albedo_map 贴图材质 + laser_retro。
    wall 病害: 薄盒贴墙(厚度在 y), 面朝 ±y, 保留碰撞
    rail/ground 病害: 薄盒朝行进方向(厚度在 x), 面朝 ±x, 无碰撞(避免挡车)

    注: 曾用 <script><uri>file://xxx.png</uri>, 实测 gz-sim8 Ogre2 明确不支持
    Ogre 材质脚本("does not support Ogre material scripts"), 材质被忽略 ->
    薄盒回落 ambient/diffuse=1 的纯色, 贴图全无 -> 隧道墙上"白斑"。
    改用官方标准 PBR <pbr><metal><albedo_map>>, 所有官方 demo 同款, 必生效。
    laser_retro 是 <visual> 级属性, 不受材质类型影响, gpu_lidar 强度通道不变。"""
    cx, cy, cz = d["center_world"]
    w, h = d["size_m"]
    surface = d.get("surface", "wall")
    tex = os.path.join(textures_dir, d["texture"])
    # DEFECT_RETRO_ZERO=1 关掉按类设的 laser_retro(intensity 通道 = 缺陷掩膜)。
    # 做"点云能不能靠几何看到病害"的实验时必须关, 否则点云支直接拿到 GT。
    retro = (0 if os.environ.get("DEFECT_RETRO_ZERO") == "1"
             else RETRO_BY_CLASS.get(d.get("class_name", ""), 100))
    emissive = EMISSIVE_BY_CLASS.get(d.get("class_name", ""), 0.05)
    if not os.path.isfile(tex):
        print(f"  ⚠ 贴图不存在: {tex}, 跳过 {d['id']}")
        return ""
    _th = WALL_THICK if surface == "wall" else PLANE_T
    if surface == "wall":
        _th = DEPTH_BY_CLASS.get(d.get("class_name", ""), WALL_THICK)
    gx, gy, gz = (w, _th, h) if surface == "wall" else (_th, w, h)
    if surface == "wall" and _th > WALL_THICK:
        # 朝隧道内侧凸出(左墙 y=+1.25 -> 往 y 减小方向), 留 2mm 嵌入避免与墙共面闪烁。
        # 符号写反会把盒子推到墙背后的实体里, gpu_lidar 一个点都测不到。
        cy = cy - (1.0 if cy > 0 else -1.0) * (_th / 2 - 0.002)
    # equip_intrusion = 立体异物(地面凸起): 给真实碰撞(gy 换深度), 让 gz 射线照墙/车能命中,
    # 几何避障检测才会在云里看到它。墙体病害仍薄盒贴墙; 轨/地病害(非入侵)无碰撞(不挡车)。
    if d.get("class_name") == "equip_intrusion":
        gx = 0.30                                  # 沿行进方向深度, 立体积块
    has_coll = (surface == "wall") or (d.get("class_name") == "equip_intrusion")
    coll = f"""        <collision name='collision'>
          <geometry><box><size>{gx} {gy} {gz}</size></box></geometry>
        </collision>
""" if has_coll else ""
    return f"""    <model name='defect_{d['id']}'>
      <static>true</static>
      <pose>{cx} {cy} {cz} 0 0 0</pose>
      <link name='link'>
        <visual name='visual'>
          <laser_retro>{retro}</laser_retro>
          <geometry><box><size>{gx} {gy} {gz}</size></box></geometry>
          <material>
            <ambient>1 1 1 1</ambient>
            <diffuse>1 1 1 1</diffuse>
            <emissive>{emissive} {emissive} {emissive} 1</emissive>
            <specular>0.05 0.05 0.05 1</specular>
            <pbr>
              <metal>
                <albedo_map>file://{tex}</albedo_map>
                <metalness>0.0</metalness>
                <roughness>0.9</roughness>
              </metal>
            </pbr>
          </material>
        </visual>
{coll}      </link>
    </model>
"""


def strip_feature_models(base: str) -> str:
    """去掉 base 世界里"特征物"段(从注释到 <!--__ROBOT__--> 之前)。"""
    start = base.find("<!-- ============ 特征物")
    end = base.find("<!--__ROBOT__-->")
    if start == -1 or end == -1:
        print("⚠ base 世界未找到特征物段/占位符, 跳过剥离")
        return base
    return base[:start] + base[end:]


def main() -> None:
    defects_json = sys.argv[1] if len(sys.argv) > 1 else DEFECTS_JSON
    out_world = sys.argv[2] if len(sys.argv) > 2 else OUT_WORLD
    # 贴图目录可参数化: 默认 data/textures; data1 源用 data_gen/textures_data1
    textures_dir = (sys.argv[3] if len(sys.argv) > 3
                    else os.path.join(ROOT, "data_gen", "textures"))

    with open(defects_json, encoding="utf-8") as f:
        manifest = json.load(f)
    defects = manifest["defects"]

    base = open(BASE_WORLD, encoding="utf-8").read()
    base = strip_feature_models(base)

    # 病害模型
    blocks = []
    for d in defects:
        xml = defect_model_xml(d, textures_dir)
        if xml:
            blocks.append(xml)
    if not blocks:
        sys.exit("错误: 没有任何病害模型生成, 检查贴图目录")
    defects_sdf = "    <!-- ============ 病害模型 (配对数据自动标注真值) ============ -->\n" + "\n".join(blocks)

    # 机器人内嵌
    robot_sdf = robot_model_sdf()

    if "<!--__ROBOT__-->" not in base:
        sys.exit("错误: base 世界缺少占位符 <!--__ROBOT__-->")
    world = base.replace("<!--__ROBOT__-->", defects_sdf + "\n  " + robot_sdf)

    # 写 + 校验 XML
    with open(out_world, "w", encoding="utf-8") as f:
        f.write(world)
    try:
        ET.parse(out_world)
    except ET.ParseError as e:
        sys.exit(f"❌ 生成的 world XML 非法: {e}")

    print(f"✅ 已生成: {out_world}")
    print(f"   病害模型 {len(blocks)} 个, 贴图目录 {textures_dir}")
    print(f"   机器人: inspection_robot @ {ROBOT_POSE}")
    print(f"   XML 校验通过")


if __name__ == "__main__":
    main()
