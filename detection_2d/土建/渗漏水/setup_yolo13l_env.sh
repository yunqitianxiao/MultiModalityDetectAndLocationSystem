#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# 一条命令重建 渗漏水 yolov13l V100 训练环境(全部落在数据盘 /root/autodl-tmp)
# 用途: AutoDL 升级 GPU 配置会清系统盘, 只保留 /root/autodl-tmp(数据盘)。
#       系统盘被清后, 运行本脚本即可重建 miniconda + venv + torch cu121 + 依赖。
# ═══════════════════════════════════════════════════════════════════
# 用法: bash /root/autodl-tmp/setup_yolo13l_env.sh
# 依赖: /root/autodl-tmp/pro/2 下已有项目文件(脚本/fork/权重/数据, 数据盘保留)。
set -euo pipefail

MINICONDA_DIR=/root/autodl-tmp/miniconda3
VENV_DIR=/root/autodl-tmp/yolo13l_env
# pip 缓存放数据盘: 系统盘被清后无需重下 wheel
export PIP_CACHE_DIR=/root/autodl-tmp/.pip_cache

echo "== [1/5] miniconda(py310) → $MINICONDA_DIR =="
if [ -x "$MINICONDA_DIR/bin/python" ] && "$MINICONDA_DIR/bin/python" --version 2>/dev/null | grep -q "Python 3.10"; then
  echo "  已存在, 跳过"
else
  cd /root/autodl-tmp
  # 动态取最新 py310 安装包名(venv 是 py310, 系统盘被清后必须装同版本才能 --upgrade 保留已装包)
  PY310=$(curl -fsSL --max-time 20 https://repo.anaconda.com/miniconda/ | grep -oE 'Miniconda3-py310_[0-9._-]+-Linux-x86_64\.sh' | sort -u | tail -1)
  echo "  安装包: $PY310"
  [ -f "$PY310" ] || curl -fsSL -o "$PY310" "https://repo.anaconda.com/miniconda/$PY310"
  bash "$PY310" -b -p "$MINICONDA_DIR"
fi

echo "== [2/5] venv → $VENV_DIR =="
if [ -e "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" --version >/dev/null 2>&1; then
  echo "  venv 正常, 跳过"
elif [ -d "$VENV_DIR" ]; then
  echo "  venv 的 python 失效(系统盘被清?), 用数据盘 miniconda 修复(--upgrade 保留已装包)..."
  "$MINICONDA_DIR/bin/python" -m venv --upgrade "$VENV_DIR"
else
  "$MINICONDA_DIR/bin/python" -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip -q

echo "== [3/5] 运行所需本地资产(Arial 字体 + yolov13n, 从数据盘恢复到 ~/.config) =="
# 系统盘被清 → /root/.config/yolov13 被清空; 训练启动时会从 ultralytics.com 下载 Arial.ttf(被卡)和
# github 下载 yolov13n.pt(AMP 检查)。这里用数据盘上的副本恢复, 彻底绕开网络。
mkdir -p /root/.config/yolov13
[ -f /root/autodl-tmp/.fonts/Arial.ttf ] && cp -f /root/autodl-tmp/.fonts/Arial.ttf /root/.config/yolov13/Arial.ttf
[ -f /root/autodl-tmp/pro/2/土建/渗漏水/yolov13n.pt ] && cp -f /root/autodl-tmp/pro/2/土建/渗漏水/yolov13n.pt /root/.config/yolov13/yolov13n.pt
ls -la /root/.config/yolov13/ 2>/dev/null || true

echo "== [4/5] torch cu121(含 sm_70, V100 必需) =="
"$VENV_DIR/bin/pip" install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --progress-bar off

echo "== [5/5] 依赖(ultralytics==8.3.63 拉非 torch 依赖, 已装 torch 不被覆盖) =="
"$VENV_DIR/bin/pip" install ultralytics==8.3.63 huggingface-hub safetensors --progress-bar off

echo "== [6/5] 自检 =="
"$VENV_DIR/bin/python" - <<'PY'
import torch
print("  torch", torch.__version__, "build", torch.version.cuda)
print("  arch", torch.cuda.get_arch_list())
print("  cuda_available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
assert "sm_70" in torch.cuda.get_arch_list(), "✗ torch 不含 sm_70! V100 跑不了, 需 cu118/cu121 构建"
assert torch.cuda.is_available(), "✗ CUDA 不可用"
PY

echo "✅ 环境就绪: $VENV_DIR"
echo "   启动训练: export YOLOV13_PYTHON=$VENV_DIR/bin/python"
echo "             bash /root/autodl-tmp/pro/2/土建/渗漏水/launch_train_v100_yolo13l.sh"
