#!/usr/bin/env bash
# 8×RTX4090 + yolov13l 渗漏水训练启动脚本（独立于 11l）
# ======================================================
# 前置: 服务器装 iMoonLab/yolov13 fork(ultralytics 8.3.63);
#       把 yolov13l.pt、data/ 与整个项目树拷到服务器。
# 用法:
#   bash 土建/渗漏水/launch_train_4090_yolo13l.sh [额外参数...]
#   e.g. bash 土建/渗漏水/launch_train_4090_yolo13l.sh --epochs 60 --freeze 0
#
# yolov13 环境自动解析:
#   - fork 源码默认在 $SCRIPT_DIR/yolov13-yolov13(与脚本同目录解压 yolov13_fork.zip),
#     存在则自动加入 PYTHONPATH(纯源码覆盖, 不动 env 里装的 mainline ultralytics;
#     mainline ≥8.4.120 是 11l 脚本的要求, 不可被 8.3.63 fork 替换)。
#     可用环境变量 YOLOV13_SITE 覆盖为其他路径。
#   - 解释器优先 $YOLOV13_PYTHON, 其次 python3(yolo_cu128 已含 torch cu128 + 依赖)
#
# 建议先跑(在服务器上):
#   python3 土建/渗漏水/yolo13l训练_8x4090.py --verify-only   # 数据自检
#   python3 土建/渗漏水/yolo13l训练_8x4090.py --probe          # 单卡显存探测
#
# 日志默认 $HOME/yolo13l_train_8x4090.log, 可用环境变量覆盖:
#   LOG=/path/x.log bash 土建/渗漏水/launch_train_4090_yolo13l.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_PY="$SCRIPT_DIR/yolo13l训练_8x4090.py"
LOG="${LOG:-$HOME/yolo13l_train_8x4090.log}"

if [ ! -f "$TRAIN_PY" ]; then
  echo "✗ 找不到 $TRAIN_PY, 请在正确相对位置执行本脚本" >&2
  exit 1
fi

YOLOV13_SITE="${YOLOV13_SITE:-$SCRIPT_DIR/yolov13-yolov13}"
if [ -d "$YOLOV13_SITE" ]; then
  export PYTHONPATH="$YOLOV13_SITE${PYTHONPATH:+:$PYTHONPATH}"
  echo "  [env] PYTHONPATH 已加入 fork: $YOLOV13_SITE"
fi
if [ -z "${YOLOV13_PYTHON:-}" ] && [ -x /root/autodl-tmp/pro2venv/bin/python ]; then
  YOLOV13_PYTHON=/root/autodl-tmp/pro2venv/bin/python
fi
PY="${YOLOV13_PYTHON:-python3}"
echo "  [env] 解释器: $PY"

# 落盘脚本 + bash 执行, 规避 SSH 内联 setsid nohup 断开被杀的问题
setsid nohup "$PY" "$TRAIN_PY" "$@" > "$LOG" 2>&1 &

echo "🚀 已启动 yolov13l 训练: PID=$!   日志: $LOG"
echo "   查看进度: tail -f $LOG"
echo "   停止训练: pkill -f '[y]olo13l训练_8x4090.py'   (注意 [y] 避免自杀)"
