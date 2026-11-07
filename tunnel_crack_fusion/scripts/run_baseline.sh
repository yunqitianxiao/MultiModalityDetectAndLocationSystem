#!/bin/bash
# 服务器上跑 A/B 两档基线：训练 + 统一尺子评测
#   bash scripts/run_baseline.sh                          # 默认数据集 -> runs/{rgb,rgb_pc}
#   DATA=/root/tunnel_crack_fusion/data/output_spalling_big bash scripts/run_baseline.sh
#        -> runs/output_spalling_big/{rgb,rgb_pc}（TAG 未给时按数据集名分目录，不覆盖旧结果）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PY:-python3}
EPOCHS=${EPOCHS:-60}
BATCH=${BATCH:-8}
WORKERS=${WORKERS:-8}
DATA=${DATA:-}
TAG=${TAG:-}
if [ -z "$TAG" ] && [ -n "$DATA" ]; then
  TAG="$(basename "$DATA")/"
fi
RUNS="runs/${TAG}"
DATA_ARG=()
[ -n "$DATA" ] && DATA_ARG=(--data "$DATA")

for MOD in rgb rgb+pc; do
  OUT="${RUNS}${MOD//+/_}"
  echo "===== TRAIN $MOD -> $OUT ====="
  "$PY" -m tunnel_tri.train \
    --modalities "$MOD" --epochs "$EPOCHS" --batch "$BATCH" \
    --workers "$WORKERS" --split sequential --out "$OUT" "${DATA_ARG[@]}"
  echo "===== EVAL $MOD ====="
  "$PY" -m tunnel_tri.evaluate --ckpt "$OUT/last.pt"
done

echo "===== SUMMARY ====="
"$PY" - "$RUNS" <<'EOF'
import json, os, sys
runs = sys.argv[1]
rows = []
for name in ('rgb', 'rgb_pc'):
    p = f'{runs}{name}/eval.json'
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    c = d['classes']['0']
    rows.append((name, d['modalities'], d['mAP_continuous'], d['mAP_voc11'],
                 c['n_gt'], c['n_det'], c['n_tp'], c['opt_thresh'], c['opt_precision'], c['opt_recall']))
print(f"{'run':<8}{'mod':<8}{'AP_cont':>9}{'AP_voc11':>10}{'n_gt':>6}{'n_det':>7}{'TP':>6}{'th':>10}{'P':>7}{'R':>7}")
for r in rows:
    print(f"{r[0]:<8}{r[1]:<8}{r[2]:>9}{r[3]:>10}{r[4]:>6}{r[5]:>7}{r[6]:>6}{r[7]:>10}{r[8]:>7}{r[9]:>7}")
EOF
