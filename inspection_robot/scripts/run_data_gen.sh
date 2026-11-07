#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# =====================================================================
# 配对数据一键生成: 贴图 -> 病害布置 -> 病害世界 -> 仿真录制 -> 属性图 -> 离线验证
# =====================================================================
#   python3 链 (无需 ROS):  贴图/布置/世界/属性图/验证
#   ros2 launch 段 (需 ROS):  录制节点驱车往返落盘 (record:=true 录完自动退出)
#
# 用法:
#   bash scripts/run_data_gen.sh                       # 输出到 data_gen/output (data 源 8 类)
#   SOURCE=data1 bash scripts/run_data_gen.sh          # data1 源 6 类 -> data_gen/output_data1
#   SOURCE=crack bash scripts/run_data_gen.sh          # 单类裂缝(用户裂缝数据集贴图) -> data_gen/output_crack
#   CRACK_CAP=24 SOURCE=crack bash scripts/run_data_gen.sh    # 每景裂缝实例数(默认 16)
#   OUT_DIR=/data/gen1 bash scripts/run_data_gen.sh    # 自定义输出目录
#   VX=0.3 T_FORWARD=50 SEED=7 bash scripts/run_data_gen.sh   # 行驶参数/随机种子
#   HEADLESS=false bash scripts/run_data_gen.sh        # 带 gz GUI 录制(调试用)
#
# 注意:
#   - 录制段需要先 colcon build 本包(自动执行, 只编 inspection_robot)
#   - headless 下 ogre2 需要可用 GPU/EGL; 无 GPU 服务器会失败, 改用 HEADLESS=false
#     (但带 GUI 需要显示器/X). 若两者都不可用, 请在有显示环境跑录制。
# =====================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 自愈: install(PROGRAMS) 依赖脚本可执行位, 新写的 .py 可能没 +x
chmod +x scripts/record_paired_data.py scripts/odom_tf_broadcaster.py scripts/pose_state_node.py 2>/dev/null || true

# 数据源选择: data(默认, 8 类含掉块/支架) / data1(6 类) / crack(单类"裂缝", 用户裂缝数据集)
SOURCE="${SOURCE:-data}"
TEX_SKIP=""     # 默认不跳过; SOURCE=data1 时按留测 stem 列表裁掉防泄漏
PLACE_EXTRA=""  # 布置阶段额外参数(单类源用)
if [ "$SOURCE" = "crack" ]; then
  TEXTURES_DIR="$ROOT/data_gen/textures_crack"
  WORLD_NAME="defect_tunnel_crack.world"
  TEX_ARG="--source crack1280"  # 用户隧道裂缝数据集(多边形标签)按框裁裂缝贴图
  OUT_DEFAULT="$ROOT/data_gen/output_crack"
  PLACE_TEX="$TEXTURES_DIR"
  FORCE_TEX=1                   # 贴图源稳定但需按上限重裁, 每次强制重生成
  PLACE_EXTRA="--classes crack --cap ${CRACK_CAP:-16}"
elif [ "$SOURCE" = "spalling" ]; then
  # 单类掉块剥落(受控实验): 复用 8 类现成 spalling_*.png, 但病害盒加厚到 12cm
  # 让 gpu_lidar 能测到几何; DEFECT_RETRO_ZERO=1 关掉 laser_retro 掩膜, 否则点云支
  # 直接读到 GT(见 generate_defect_world.py 里 RETRO_BY_CLASS 的说明)。
  TEXTURES_DIR="$ROOT/data_gen/textures_spalling"
  WORLD_NAME="defect_tunnel_spalling.world"
  TEX_ARG=""
  OUT_DEFAULT="$ROOT/data_gen/output_spalling"
  PLACE_TEX=""
  FORCE_TEX=""
  PLACE_EXTRA="--classes spalling --cap ${SPALLING_CAP:-8}"
elif [ "$SOURCE" = "rail" ]; then
  # 单类扣件断裂(受控实验): 轨顶薄盒法线朝 ±x = 沿视轴, 几何差异直接落在
  # depth 通道上(侧墙类病害在 depth 上是恒等的, 见掉块实验的对照)。
  TEXTURES_DIR="$ROOT/data_gen/textures"
  WORLD_NAME="defect_tunnel_rail.world"
  TEX_ARG=""
  OUT_DEFAULT="$ROOT/data_gen/output_rail_broken"
  PLACE_TEX="$TEXTURES_DIR"
  FORCE_TEX=""
  PLACE_EXTRA="--classes rail_fastener_broken --cap ${RAIL_CAP:-8}"
elif [ "$SOURCE" = "data1" ]; then
  TEXTURES_DIR="$ROOT/data_gen/textures_data1"
  WORLD_NAME="defect_tunnel_data1.world"
  TEX_ARG="--source bing"       # data1 贴图来自病集(病害训练集)真实图, 按框裁缺陷
  OUT_DEFAULT="$ROOT/data_gen/output_data1"
  PLACE_TEX="$TEXTURES_DIR"     # 布置时按纹理铺满(每张病集缺陷恰好用一次)
  FORCE_TEX=1                   # 病集贴图每次强制重生成, 避免拿到旧的 6 类贴图
  # 防泄漏: 切分出的留测图(病集土建干净测试集)不裁贴图, 使训练贴图与测试图零重叠
  SKIP_STEMS_FILE="${SKIP_STEMS_FILE:-$ROOT/data_gen/tujian_test_stems.txt}"
  if [ -f "$SKIP_STEMS_FILE" ]; then
    TEX_SKIP="--skip-stems $SKIP_STEMS_FILE"
    echo "  ↻ 检测到留测 stem 列表 $SKIP_STEMS_FILE, 裁贴图时跳过 ($(wc -l < "$SKIP_STEMS_FILE") 个)"
  else
    echo "  ⚠ 未找到留测 stem 列表 $SKIP_STEMS_FILE (先跑 split_bing_tujian.py), 本次不跳过"
    TEX_SKIP=""
  fi
else
  TEXTURES_DIR="$ROOT/data_gen/textures_spalling"
  WORLD_NAME="defect_tunnel.world"
  TEX_ARG=""
  OUT_DEFAULT="$ROOT/data_gen/output"
  PLACE_TEX=""
  FORCE_TEX=""
fi
OUT_DIR="${OUT_DIR:-$OUT_DEFAULT}"
SEED="${SEED:-42}"
VX="${VX:-0.4}"
T_FORWARD="${T_FORWARD:-40.0}"    # 带小数点, 避免 launch 把它当 int 传给 DOUBLE 参数
HEADLESS="${HEADLESS:-false}"

step() { echo; echo "========== $1 =========="; }

step "1/5 病害贴图准备 (source=$SOURCE)"
if [ "$FORCE_TEX" = "1" ]; then
  echo "病集贴图: 强制重生成 (prepare_from_bing 会先清空 $TEXTURES_DIR 再裁新图)"
  python3 "$ROOT/data_gen/prepare_textures.py" $TEX_ARG $TEX_SKIP
elif [ -d "$TEXTURES_DIR" ] && [ -n "$(ls -A "$TEXTURES_DIR" 2>/dev/null)" ]; then
  echo "贴图已存在 ($(ls "$TEXTURES_DIR" | wc -l) 张), 跳过"
else
  python3 "$ROOT/data_gen/prepare_textures.py" $TEX_ARG $TEX_SKIP
fi

step "2/5 病害布置 (seed=$SEED) -> defects.json"
# 位置参数(textures_dir)须排在可选参数前, argparse 才能正确解析
python3 "$ROOT/data_gen/defect_placement.py" "$SEED" 8 ${PLACE_TEX:+"$PLACE_TEX"} ${PLACE_EXTRA}

step "3/5 生成病害世界 ($WORLD_NAME)"
python3 "$ROOT/scripts/generate_defect_world.py" \
  "$ROOT/data_gen/defects.json" "$ROOT/worlds/$WORLD_NAME" "$TEXTURES_DIR"

step "4/5 编译本包 + 启动仿真并录制"
# 清理残留 sim/ros 进程: 多次运行后旧 gz sim / ros2 launch 会残留,
# 多个实例同时发布 /odom /camera -> record 收到混流(odom 在 -172m 和隧道内跳变)。
pkill -f 'gz sim' 2>/dev/null || true
pkill -f 'ros2 launch inspection_robot' 2>/dev/null || true
pkill -f 'record_paired_data' 2>/dev/null || true
pkill -f 'parameter_bridge' 2>/dev/null || true
sleep 2
colcon build --symlink-install --packages-select inspection_robot

# 把病害真值清单预拷到输出目录:
# 录制节点在 install 下运行, 仓库根 data_gen/defects.json 不在它的相对路径上,
# 统一以 out_dir/defects.json 为唯一来源。
mkdir -p "$OUT_DIR"
cp "$ROOT/data_gen/defects.json" "$OUT_DIR/defects.json"
echo ">>> 已预拷贝 defects.json -> $OUT_DIR/defects.json"
# 在子 shell 里 source, 避免污染当前 shell。
# colcon 的 setup.bash 引用未绑定变量(COLCON_TRACE), 会撞 set -u, 先临时关掉。
ROS_GZ_WS="${ROS_GZ_WS:-$HOME/workspace/ros_gz_ws}"
(
  set +u
  source install/setup.bash
  # 关键: 用用户已编译的 Harmonic 版 ros_gz_bridge, 覆盖 apt 的 Fortress 版
  # (apt 版链接 ign-msgs8, 解析不了 Harmonic 的 gz-msgs10 消息 -> ROS 侧零数据)
  if [ -f "$ROS_GZ_WS/install/setup.bash" ]; then
    source "$ROS_GZ_WS/install/setup.bash"
    echo ">>> 已加载 Harmonic 桥: $(which parameter_bridge 2>/dev/null)"
  else
    echo "!!! 警告: 未找到 $ROS_GZ_WS/install/setup.bash, 将使用 apt 的 Fortress bridge (可能 ROS 侧无数据)"
  fi
  set -u
  echo ">>> ros2 launch ... world=$WORLD_NAME record:=true headless:=$HEADLESS out_dir=$OUT_DIR"
  ros2 launch inspection_robot inspection_robot.launch.py \
    world:="$ROOT/worlds/$WORLD_NAME" \
    record:=true headless:="$HEADLESS" use_rviz:=false \
    out_dir:="$OUT_DIR" vx:="$VX" t_forward:="$T_FORWARD"
)
echo ">>> 录制完成, 落盘 $OUT_DIR"

step "5/5 属性图后处理 (点云 -> 稠密三通道)"
python3 "$ROOT/scripts/build_attrmaps.py" "$OUT_DIR"

step "离线几何/属性验证"
python3 "$ROOT/data_gen/validate_offline.py"

echo
echo "🎉 数据生成完毕: $OUT_DIR"
echo "    images/ labels/ (YOLO GT) clouds/ poses/ attrmaps/ + 标定/元数据"
