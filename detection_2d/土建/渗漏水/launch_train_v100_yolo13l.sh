#!/usr/bin/env bash
# V100 + yolov13l 渗漏水训练启动脚本（独立于 4090 版, 配合 yolo13l训练_v100.py）
# =====================================================
# 用法(服务器上, fork 目录与脚本同目录 yolov13-yolov13/):
#   bash 土建/渗漏水/launch_train_v100_yolo13l.sh [额外参数...]
#   e.g. bash 土建/渗漏水/launch_train_v100_yolo13l.sh --epochs 120 --imgsz 512
#
# 关键: YOLOV13_PYTHON 必须指向含 cu121 torch(含 sm_70) 的 python(服务器上):
#   export YOLOV13_PYTHON=/root/miniconda3/envs/yolo13l/bin/python
#
# 建议先跑(在服务器上, 每次 PYTHONPATH 都要指向 fork):
#   PYTHONPATH=$SCRIPT_DIR/yolov13-yolov13 $YOLOV13_PYTHON \
#       土建/渗漏水/yolo13l训练_v100.py --check-env     # torch/驱动/Volta(sm_70) 自检
#   ... --verify-only                                  # 数据自检
#   ... --probe                                        # 单卡显存探测
#
# 日志默认 $HOME/yolo13l_train_v100.log, 可用环境变量覆盖:
#   LOG=/path/x.log bash 土建/渗漏水/launch_train_v100_yolo13l.sh
set -u

# 共享服务器 DDP 常见 MKL/OpenMP 线程层冲突 → rank 进程 OOM / undefined symbol
# (4090 服务器实测踩过)。显式 GNU 线程层规避; DDP 子进程会继承此 env。
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 统一 CWD 到脚本目录(数据盘): fork 的 AMP 检查会 YOLO("yolov13n.pt") 并下载到 CWD,
# SCRIPT_DIR 里已预置 yolov13n.pt, cd 到这里可跳过网络下载(升级换机后同样适用)。
cd "$SCRIPT_DIR"
TRAIN_PY="$SCRIPT_DIR/yolo13l训练_v100.py"
FORK_DIR="$SCRIPT_DIR/yolov13-yolov13"
LOG="${LOG:-$HOME/yolo13l_train_v100.log}"

# YOLOV13_PYTHON: 显式指定含 cu121 torch 的 python; 否则按已知路径/系统 python 兜底
# (重建脚本 setup_yolo13l_env.sh 会把环境装在 /root/autodl-tmp/yolo13l_env)
if [ -n "${YOLOV13_PYTHON:-}" ]; then
  PY="$YOLOV13_PYTHON"
elif [ -x "/root/autodl-tmp/yolo13l_env/bin/python" ]; then
  PY="/root/autodl-tmp/yolo13l_env/bin/python"
else
  PY="python3"
fi

if [ ! -f "$TRAIN_PY" ]; then
  echo "✗ 找不到 $TRAIN_PY, 请在正确相对位置执行本脚本" >&2
  exit 1
fi

# fork 目录存在则用 PYTHONPATH 前置覆盖主线程 ultralytics(必须! 否则加载 yolov13l.pt 失败)
if [ -d "$FORK_DIR" ]; then
  export PYTHONPATH="$FORK_DIR:${PYTHONPATH:-}"
  echo "  [fork] PYTHONPATH 前置: $FORK_DIR"
else
  echo "⚠️  未找到 fork 目录 $FORK_DIR, 可能用的是主线程 ultralytics, 加载 yolov13l.pt 会失败" >&2
fi

echo "  [py] $PY"
if ! "$PY" -c "import sys; assert sys.version_info >= (3, 8)" 2>/dev/null; then
  echo "✗ python 不可用: $PY" >&2
  exit 1
fi

# 落盘脚本 + bash 执行, 规避 SSH 内联 setsid nohup 断开被杀的问题
setsid nohup "$PY" "$TRAIN_PY" "$@" > "$LOG" 2>&1 &

echo "🚀 已启动 yolov13l V100 训练: PID=$!   日志: $LOG"
echo "   查看进度: tail -f $LOG"
echo "   停止训练: pkill -f '[y]olo13l训练_v100.py'   (注意 [y] 避免自杀)"
