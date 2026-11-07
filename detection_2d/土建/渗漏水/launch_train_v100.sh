#!/usr/bin/env bash
# 8×V100 + yolo11x 渗漏水训练启动脚本（独立于 4090 版）
# =====================================================
# 用法: 先把整个项目树(含 data/ 和 yolo11x.pt)拷到服务器, 然后:
#   bash 土建/渗漏水/launch_train_v100.sh [额外参数...]
#   e.g. bash 土建/渗漏水/launch_train_v100.sh --epochs 120 --imgsz 512
#
# 若 torch 装在 conda 环境, 先 conda activate <env> 再执行本脚本。
#
# 建议先跑(在服务器上):
#   python3 土建/渗漏水/yolo训练_v100.py --check-env     # torch/驱动/Volta(sm_70) 自检, 不训练
#   python3 土建/渗漏水/yolo训练_v100.py --verify-only    # 数据自检
#   python3 土建/渗漏水/yolo训练_v100.py --probe           # 单卡显存探测
#
# 日志默认 $HOME/yolo11x_train_v100.log, 可用环境变量覆盖:
#   LOG=/path/x.log bash 土建/渗漏水/launch_train_v100.sh
set -u

# 共享服务器 DDP 常见 MKL/OpenMP 线程层冲突 → rank 进程 OOM / undefined symbol
# (4090 服务器实测踩过: DDP 启动期 dist.broadcast(self.amp) OOM)。
# 显式 GNU 线程层规避; DDP 子进程会继承此 env。V100 服务器即使不冲突也无副作用。
export MKL_THREADING_LAYER=GNU
export MKL_SERVICE_FORCE_INTEL=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_PY="$SCRIPT_DIR/yolo训练_v100.py"
LOG="${LOG:-$HOME/yolo11x_train_v100.log}"

if [ ! -f "$TRAIN_PY" ]; then
  echo "✗ 找不到 $TRAIN_PY, 请在正确相对位置执行本脚本" >&2
  exit 1
fi

# 落盘脚本 + bash 执行, 规避 SSH 内联 setsid nohup 断开被杀的问题
setsid nohup python3 "$TRAIN_PY" "$@" > "$LOG" 2>&1 &

echo "🚀 已启动 V100 训练: PID=$!   日志: $LOG"
echo "   查看进度: tail -f $LOG"
echo "   停止训练: pkill -f '[y]olo训练_v100.py'   (注意 [y] 避免自杀)"
