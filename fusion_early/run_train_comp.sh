#!/usr/bin/env bash
# 三支"仅赛方病集·全幅贴满" 顺序训练 (单卡) + test评估
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python3}
MODEL=${MODEL:-$ROOT/../weights/yolo11l.pt}
cd "$ROOT"
EPOCHS="${EPOCHS:-200}"

run() {
  echo "======== $1  [epochs=$EPOCHS] ========"
  "$PY" "$1" --data "$2" --model "$MODEL" --epochs "$EPOCHS" \
    --imgsz 1280 --test-imgsz 1280 --per-gpu-batch 8 --name "$3" \
    > "$ROOT/comp_$3.log" 2>&1
}

run train_轨道_3cls.py "$ROOT/轨道_3cls_comp/data.yaml" 轨道_3cls_comp
run train_土建_3cls.py "$ROOT/土建_3cls_comp/data.yaml" 土建_3cls_comp
run train_设备_2cls.py "$ROOT/设备_2cls_comp/data.yaml" 设备_2cls_comp

echo ALL_TRAIN_DONE
