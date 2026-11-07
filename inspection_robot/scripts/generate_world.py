#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把机器人模型(由 xacro URDF 生成)内嵌进世界文件

背景: 在 gz-sim 中通过 create 服务动态 spawn 模型时, 本环境的 create 服务调用不稳定(超时),
      且 spawn 的模型传感器不发布数据。把模型直接写进世界文件(加载时创建, 与 gz 自带
      camera_sensor.sdf 示例一致)是最可靠的方式。

用法:
    python3 scripts/generate_world.py
    # 修改 urdf/inspection_robot.urdf.xacro 后重跑本脚本再启动

输入:
    urdf/inspection_robot.urdf.xacro        (机器人模型, 唯一数据源)
    worlds/inspection_tunnel_base.world     (环境模板, 含 <!--__ROBOT__--> 占位符)
输出:
    worlds/inspection_tunnel.world          (最终世界, launch 使用)
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF_XACRO = os.path.join(ROOT, "urdf", "inspection_robot.urdf.xacro")
BASE_WORLD = os.path.join(ROOT, "worlds", "inspection_tunnel_base.world")
OUT_WORLD = os.path.join(ROOT, "worlds", "inspection_tunnel.world")
ROBOT_POSE = "0 0 0.20 0 0 0"  # 初始位姿: base_link 略高于地面, 落地后轮子着地


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print("执行失败:", cmd)
        print(r.stderr[-2000:])
        sys.exit(1)
    return r.stdout


def main():
    # 1. xacro -> URDF
    urdf = run(f"xacro {URDF_XACRO}")
    tmp_urdf = "/tmp/_robot_generated.urdf"
    with open(tmp_urdf, "w") as f:
        f.write(urdf)

    # 2. URDF -> SDF (sdformat 处理 <gazebo> 插件/传感器)
    sdf = run(f"gz sdf -p {tmp_urdf}")

    # 3. 提取 <model> 元素
    m = re.search(r"<model[\s\S]*?</model>", sdf)
    if not m:
        print("错误: 转换后的 SDF 中未找到 <model> 元素")
        sys.exit(1)
    model = m.group(0)

    # 4. 给模型加初始位姿(内嵌世界的模型 pose)
    model = re.sub(r"<model([^>]*)>", r"<model\1>\n    <pose>%s</pose>" % ROBOT_POSE,
                   model, count=1)

    # 5. 注入 base 世界
    base = open(BASE_WORLD, encoding="utf-8").read()
    if "<!--__ROBOT__-->" not in base:
        print("错误: base 世界文件缺少占位符 <!--__ROBOT__-->")
        sys.exit(1)
    world = base.replace("<!--__ROBOT__-->", model)
    with open(OUT_WORLD, "w", encoding="utf-8") as f:
        f.write(world)

    print(f"✅ 已生成世界文件: {OUT_WORLD}")
    print(f"   模型: inspection_robot, 初始位姿: {ROBOT_POSE}")


if __name__ == "__main__":
    main()
