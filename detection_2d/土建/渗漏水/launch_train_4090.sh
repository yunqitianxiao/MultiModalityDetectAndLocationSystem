#!/usr/bin/env bash
# 8×RTX4090 + yolo11l 渗漏水训练启动脚本
# =============================================
# 用法: 先把整个项目树(含 data/ 和 yolo11l.pt)拷到服务器, 然后:
#   bash 土建/渗漏水/launch_train_4090.sh [额外参数...]
#   e.g. bash 土建/渗漏水/launch_train_4090.sh --epochs 120 --imgsz 640
#
# 建议先跑(在服务器上):
#   python3 土建/渗漏水/yolo训练_8x4090.py --verify-only   # 数据自检
#   python3 土建/渗漏水/yolo训练_8x4090.py --probe          # 单卡显存探测
#
# 日志默认 $HOME/yolo11l_train_8x4090.log, 可用环境变量覆盖:
#   LOG=/path/x.log bash 土建/渗漏水/launch_train_4090.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_PY="$SCRIPT_DIR/yolo训练_8x4090.py"
LOG="${LOG:-$HOME/yolo11l_train_8x4090.log}"

if [ ! -f "$TRAIN_PY" ]; then
  echo "✗ 找不到 $TRAIN_PY, 请在正确相对位置执行本脚本" >&2
  exit 1
fi

# 落盘脚本 + bash 执行, 规避 SSH 内联 setsid nohup 断开被杀的问题
setsid nohup python3 "$TRAIN_PY" "$@" > "$LOG" 2>&1 &

echo "🚀 已启动训练: PID=$!   日志: $LOG"
echo "   查看进度: tail -f $LOG"
echo "   停止训练: pkill -f '[y]olo训练_8x4090.py'   (注意 [y] 避免自杀)"
