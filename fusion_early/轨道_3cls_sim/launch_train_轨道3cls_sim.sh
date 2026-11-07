#!/bin/bash
# 轨道3类(yolo11l)仿真数据集训练 后台启动器 —— 服务器上后台跑, 日志写 train.log
# ============================================================================
# 用前: 把 本脚本 + train_yolo11l_轨道3cls_sim.py + 轨道_3cls_sim/ 放到同一目录。
#   setsid nohup 后台运行, 关终端不中断; 训练完自动在外部 test 集上评估。
#
# 用法:
#   bash launch_train_轨道3cls_sim.sh                        # 默认超参
#   bash launch_train_轨道3cls_sim.sh --epochs 150 --imgsz 1024
#   bash launch_train_轨道3cls_sim.sh --device 0,1 --per_gpu_batch 8
# 可以把参数写在末尾, 会原样透传给训练脚本。
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# 预训练权重: 没有则让 ultralytics 自动下载(yolo11l.pt), 这里只提示不明退
[ -f yolo11l.pt ] || echo "[launch] 未找到 yolo11l.pt, 将由 ultralytics 自动下载(需联网)"

# 解释器: 默认 python, 可用环境变量 PY 覆盖(如 PY=/root/miniconda3/bin/python)
PY="${PY:-python}"

# setsid 脱离终端, nohup 防挂断; 日志写当前目录 train.log
setsid nohup "$PY" "$DIR/train_yolo11l_轨道3cls_sim.py" "$@" > train.log 2>&1 < /dev/null &
echo "launched pid $! ; log=$DIR/train.log"
echo "[launch] 如需指定GPU解释器:  PY=/root/miniconda3/bin/python bash $0 --device 0,1"
