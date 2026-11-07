#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# =====================================================================
# cut-paste 重建后 · 三支 服务器一键训练+test评估
# =====================================================================
# 在服务器上于本项目根运行 (需 scp 整个 pro/3 到服务器, 保持 目录结构).
#   cd /xx/yy/pro/3 && bash run_train_all.sh
#
# 每支: 用 train_<分支>_3cls.py / train_设备_2cls.py 训练, 训完自动对
#       test(病集留测真图) 用 --test-imgsz 1280 评估, 打印 mAP50/mAP50-95。
# find_data() 依赖"脚本同目录下有 <分支>_3cls_sim/data.yaml", 所以整目录搬到位即可。
#
# 前置: 服务器有 yolo11l.pt (或能联网自动下载)。AutoDL 无外网时软链本地权重:
#   ln -sf /path/to/yolo11l.pt  <服务器项目根>/yolo11l.pt
# =====================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[ -f 轨道_3cls_sim/data.yaml ] || { echo "缺 轨道_3cls_sim/data.yaml"; exit 1; }

run () {
  echo; echo "################ $1 ################"
  python3 "$1" "${@:2}" 2>&1 | grep -E "TRAIN_DONE|TEST|mAP50|mAP50-95|\[TEST|Error|Traceback" || true
}

run train_轨道_3cls.py
run train_土建_3cls.py
run train_设备_2cls.py

echo
echo "🎉 全部训练评估完成。若某支 TEST mAP50 仍≈0, 说明 cut-paste 未脱离0, 转真实数据路线。"
