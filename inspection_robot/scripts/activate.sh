#!/usr/bin/env bash
# =====================================================================
# 一键激活 巡检项目 Python 环境: 项目 venv + ROS2 Humble
# =====================================================================
# 用法:
#   source scripts/activate.sh       (记得 source, 子 shell 里无效)
#   之后:
#     python    -> .venv/bin/python (numpy/scipy/cv2/torch-CUDA)
#     rclpy     -> 通过 PYTHONPATH 注入可见 (跑 ROS 节点时)
#     ros2/gz   -> 正常可用
# =====================================================================
set -e

# 1. ROS2 Humble (若尚未在环境里)
if [ -z "${ROS_DISTRO:-}" ] || [ "${ROS_DISTRO}" != "humble" ]; then
  source /opt/ros/humble/setup.bash
fi

# 2. 项目 venv (PYTHONPATH 保留, ROS 包仍可见)
_VENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv"
source "${_VENV}/bin/activate"

# 3. 项目自己的 workspace (colcon build 出的包, record 节点等)
_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/install"
if [ -f "${_WS}/setup.bash" ]; then
  # 已在 venv 里, ROS 包路径可能覆盖; 只补 colcon 的 libexec/共享路径
  source "${_WS}/setup.bash" 2>/dev/null || true
fi

# 4. 用户编译的 Harmonic 版 ros_gz_bridge (覆盖 apt 的 Fortress 版)
#    apt 版链接 ign-msgs8, 解析不了 Harmonic 的 gz-msgs10 消息 -> ROS 侧零数据。
#    必须 source 在 apt 之后、且此 workspace 排在 AMENT_PREFIX_PATH 前面才能覆盖。
_ROS_GZ_WS="${ROS_GZ_WS:-$HOME/workspace/ros_gz_ws}"
if [ -f "${_ROS_GZ_WS}/install/setup.bash" ]; then
  source "${_ROS_GZ_WS}/install/setup.bash"
  echo "✅ 已加载 Harmonic 版 ros_gz_bridge: $(ros2 pkg prefix ros_gz_bridge 2>/dev/null)"
fi

echo "✅ 巡检项目环境: python=$(which python) | ROS=${ROS_DISTRO} | VENV=${VIRTUAL_ENV}"
