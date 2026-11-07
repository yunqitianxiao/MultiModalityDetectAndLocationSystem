#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# =====================================================================
# V100 服务器 · 多 seed 配对数据生成 (gz 原生, 无 ROS 桥)
# =====================================================================
# 在服务器仓库根运行 (需: python3 + numpy/cv2/scipy + gz-sim8/ogre2 + xacro + gz sdf)。
# 每 seed: 病害布置 -> 生成世界 -> gz 仿真 + gz_record 录制 ->
#          离线标签 + 属性图 -> .done 标记。
#
# 前置: 1) 本机已生成 textures_data1 并 scp 到服务器 (病集裁图+稀有类增强)
#       2) 仓库 scripts/data_gen/worlds/urdf 已 scp 到服务器
#
# 用法:
#   bash scripts/run_server_gen.sh              # seed 0..9
#   SEED_START=0 SEED_END=0 bash scripts/run_server_gen.sh   # 单 seed
# =====================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# xacro 需要 ROS PATH; source 会引用未绑定变量撞 set -u, 先临时关掉
if [ -f /opt/ros/humble/setup.bash ]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

TEXTURES="$ROOT/data_gen/textures_data1"
WORLD="$ROOT/worlds/defect_tunnel_data1.world"
DEFECTS="$ROOT/data_gen/defects.json"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-9}"
GZ_TIMEOUT="${GZ_TIMEOUT:-220}"      # 每 seed 仿真最大秒数 (80s 行驶 + 余量)

command -v gz >/dev/null || { echo "缺少 gz"; exit 1; }
command -v xacro >/dev/null || { echo "缺少 xacro"; exit 1; }
[ -d "$TEXTURES" ] || { echo "缺少 $TEXTURES, 请先 scp"; exit 1; }

run_seed() {
  local seed="$1"
  local out="$ROOT/data_gen/output_big_s${seed}"
  echo; echo "########## seed=$seed ##########"

  pkill -f 'gz sim' 2>/dev/null || true
  sleep 2

  echo "--- $seed: 病害布置 (seed=$seed, 铺满 textures)"
  python3 "$ROOT/data_gen/defect_placement.py" "$seed" 8 "$TEXTURES"

  echo "--- $seed: 生成世界"
  python3 "$ROOT/scripts/generate_defect_world.py" "$DEFECTS" "$WORLD" "$TEXTURES"

  echo "--- $seed: gz 仿真 + 录制"
  rm -rf "$out"; mkdir -p "$out"
  timeout "$GZ_TIMEOUT" gz sim -s -r --render-engine ogre2 -v 2 "$WORLD" \
      > "$out/gz.log" 2>&1 &
  local gzpid=$!
  sleep 9    # 让仿真/传感器起来, gz_record 再订阅
  # gz_record 若仿真卡死不会自行退出, 用 timeout 兜底 (略小于 GZ_TIMEOUT)
  timeout "$((GZ_TIMEOUT - 15))" python3 "$ROOT/scripts/gz_record.py" \
      --out_dir "$out" \
      || echo "⚠ gz_record 超时/异常退出 (seed=$seed, 日志见 $out/gz.log)"
  kill "$gzpid" 2>/dev/null || true
  wait "$gzpid" 2>/dev/null || true

  echo "--- $seed: 离线标签"
  python3 "$ROOT/data_gen/build_labels_offline.py" --out_dir "$out" --defects "$DEFECTS"

  echo "--- $seed: 属性图"
  python3 "$ROOT/scripts/build_attrmaps.py" "$out"

  touch "$out/.done"
  echo "✅ seed=$seed 完成: $(ls "$out/images" | wc -l) 帧 -> $out"
}

for s in $(seq "$SEED_START" "$SEED_END"); do
  run_seed "$s"
done

echo
echo "🎉 全部 seed ($SEED_START..$SEED_END) 生成完毕:"
for s in $(seq "$SEED_START" "$SEED_END"); do
  n=$(ls "$ROOT/data_gen/output_big_s${s}/images" 2>/dev/null | wc -l)
  echo "   output_big_s${s}: ${n} 帧"
done
