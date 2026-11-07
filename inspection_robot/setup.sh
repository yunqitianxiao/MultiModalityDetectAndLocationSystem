#!/usr/bin/env bash
# ============================================================
# 巡检小车仿真环境一键加载
# 依次 source 三个环境:
#   1) ROS2 Humble          (/opt/ros/humble)
#   2) ros_gz Harmonic 版    ($ROS_GZ_WS/install)  ← 源码编译, 配 Harmonic
#   3) 本项目包              (本仓库 inspection_robot/install)
# 用法: source setup.sh
# ============================================================
set -e

ROS_GZ_WS="${ROS_GZ_WS:-$HOME/ros_gz_ws}"
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. ROS2 Humble
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
else
  echo "[setup.sh] 警告: 未找到 /opt/ros/humble" >&2
fi

# 2. ros_gz (Harmonic 版, 源码编译)
if [ -f "$ROS_GZ_WS/install/setup.bash" ]; then
  source "$ROS_GZ_WS/install/setup.bash"
else
  echo "[setup.sh] 警告: 未找到 $ROS_GZ_WS/install/setup.bash, 请先编译 ros_gz" >&2
fi

# 3. 本项目
if [ -f "$PROJ_DIR/install/setup.bash" ]; then
  source "$PROJ_DIR/install/setup.bash"
else
  echo "[setup.sh] 警告: 未找到 $PROJ_DIR/install/setup.bash, 请先在项目目录 colcon build" >&2
fi

echo "✅ 环境已加载: Humble + ros_gz(Harmonic) + inspection_robot"
echo "   启动仿真: ros2 launch inspection_robot inspection_robot.launch.py"
