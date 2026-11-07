#!/bin/bash
# 同步代码（可带数据集）到训练服务器。
#
# 用法:
#   bash scripts/sync_to_server.sh                          # 只同步代码与脚本
#   bash scripts/sync_to_server.sh output_spalling_big      # 额外同步一份配对数据集
#   DATA_ROOT=~/workspace/inspection_robot/data_gen bash scripts/sync_to_server.sh output_spalling_big
#
# 数据集源目录默认是 inspection_robot/data_gen/<name>，落到服务器 $DEST/data/<name>。
# 密码不入库: 先 ssh-copy-id 免密，或 eval $(ssh-agent) 后 ssh-add。
set -euo pipefail

HOST=${HOST:?请通过环境变量 HOST 指定训练服务器}
PORT=${PORT:-22}
RUSER=${REMOTE_USER:-root}
DEST=${DEST:-/root/tunnel_crack_fusion}
DATA_ROOT=${DATA_ROOT:-$HOME/workspace/inspection_robot/data_gen}

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_CMD=(ssh -p "$PORT" "$RUSER@$HOST")

echo ">>> 代码: $SRC/ -> $RUSER@$HOST:$DEST/"
"${SSH_CMD[@]}" "mkdir -p '$DEST'"
rsync -az --delete --info=progress2 -e "ssh -p $PORT" \
  --exclude '__pycache__' --exclude '.idea' \
  --exclude 'runs' --exclude 'data' --exclude 'weights' \
  "$SRC/" "$RUSER@$HOST:$DEST/"

for name in "$@"; do
  src="$DATA_ROOT/$name"
  if [ ! -d "$src" ]; then
    echo "!!! 数据集不存在: $src" >&2
    exit 1
  fi
  echo ">>> 数据: $src/ -> $DEST/data/$name/"
  "${SSH_CMD[@]}" "mkdir -p '$DEST/data/$name'"
  rsync -az --info=progress2 -e "ssh -p $PORT" "$src/" "$RUSER@$HOST:$DEST/data/$name/"
done

echo ">>> 完成。服务器上与本地一致的入口: bash scripts/run_baseline.sh"
