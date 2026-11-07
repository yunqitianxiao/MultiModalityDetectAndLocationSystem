#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
渗漏水 YOLOv13-L 检测训练 —— V100(Volta sm_70) 适配版(独立脚本, 不碰 4090 版)
================================================================
数据: data/土建_渗漏水_yolo  (单类 leakage, 420/74/55)
模型: yolov13l.pt —— YOLOv13 Hypergraph 版 L 级 (27.6M 参数, 88.4 GFLOPs, COCO 53.4 mAP)

本脚本是 yolo13l训练_8x4090.py 的 V100 独立版(沿用 yolo训练_v100.py 的
Volta 适配做法): 只针对 V100 硬件差异改几处, 逻辑与 4090 版一致。

  ⚠️ 先用 yolo13l训练_8x4090.py 头部说明理解 v13 与 11l 的差异(fork/权重名/冻结),
    本脚本只补充 V100 相关约束:

  1. torch 必须编译进 sm_70(Volta)。cu130/cu128 部分构建只到 sm_80+, 装了也能
     import, 但模型一加载就崩 → 用 --check-env 先自检。
  2. V100 无 bf16, 只有 fp16 tensor core → 混合精度固定 fp16(amp=True)。
  3. YOLOv13 官方代码是 iMoonLab/yolov13 的 ultralytics 8.3.63 fork, 不是主线程
     ultralytics! 权重文件是 yolov13l.pt(带 v)。fork 通过 PYTHONPATH 覆盖使用:
        PYTHONPATH=<fork目录>:... python yolo13l训练_v100.py
     ⚠️ fork 的 block.py 在 import 时调用 torch.cuda.get_device_capability(),
     会把 CUDA 上下文绑到默认物理卡(GPU 0)。本脚本已在任何 import ultralytics
     之前锁定 CUDA_VISIBLE_DEVICES(见 main 内注释), 勿改动该顺序。
  4. 空闲 GPU 阈值不再写死 20GB: 默认按"每卡总显存×85%"自适应(16GB/32GB V100
     都适用), 也可 --gpu-free-threshold 覆盖。卡数自适应(1 卡 ~ 8 卡通吃)。
  5. 默认 batch=0(自动): 探测到卡数与每卡显存后按"每卡24(32GB卡)/16(16GB卡) ×
     卡数"设定, 先 --probe 探显存再调。默认 nbs=256 → 只做梯度累积、不改生效 LR,
     任意卡数下有效 batch 都指向 256(5×32GB: batch120 + 累积2 ≈ 240)。
  6. V100 服务器 torch 装法(看 nvidia-smi 驱动版本选一个, 装完先 --check-env):
      - 驱动 ≥530 (CUDA 12.x 可用):
            pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
              --index-url https://download.pytorch.org/whl/cu121
        (cu121 构建含 sm_70; 国内服务器走 SJTU 上交镜像)
      - 驱动 520~530 (CUDA 11.x):
            pip install torch==2.1.2 torchvision==0.16.2 \
              --index-url https://download.pytorch.org/whl/cu118
      - 如果 torch 装在 conda 环境: 先 conda activate <env> 再跑脚本。

用法(V100 服务器上):
  0) 前置: 装 torch(含 sm_70) + 放好 iMoonLab yolov13 fork 目录;
           把 yolov13l.pt、data/ 与整个项目树拷到服务器(保持相对位置)
  1) 环境自检: PYTHONPATH=<fork目录> python3 土建/渗漏水/yolo13l训练_v100.py --check-env
  2) 校验数据: PYTHONPATH=<fork目录> python3 土建/渗漏水/yolo13l训练_v100.py --verify-only
  3) 看哪些卡空闲: nvidia-smi
  4) 显存探测:   PYTHONPATH=<fork目录> python3 土建/渗漏水/yolo13l训练_v100.py --probe
  5) 启动训练:   bash 土建/渗漏水/launch_train_v100_yolo13l.sh
  6) 只评估:     PYTHONPATH=<fork目录> python3 土建/渗漏水/yolo13l训练_v100.py \
                    --eval-only --weights <best.pt>

⚠️ ultralytics 8.3.63 fork 关键坑(与 4090 版相同, 勿回退):
  - lr0 按 batch/nbs 缩放是旧版行为 → 本脚本 nbs=0 跟随 --batch, 使生效 LR=lr0;
    要拉大有效 batch 用 --nbs(只影响梯度累积, 不改 LR)。
  - optimizer 务必显式传 "AdamW"! optimizer="auto" 会用 lr_fit=0.002(nc=1) 覆盖
    lr0, 必发散。
  - v13 层结构与 11l 不同: 11l 的 freeze=10 在此 ≠ 冻结 backbone → 默认 --freeze 0
    (全量微调, 420图 > 50图, 文献结论全量微调优于冻结)。
  - 共享服务器 DDP 常见 MKL/OpenMP 线程层冲突(rank 进程 OOM / undefined symbol),
    启动前 export MKL_THREADING_LAYER=GNU MKL_SERVICE_FORCE_INTEL=1
    (launch_train_v100_yolo13l.sh 已内置)。

建议: 训完在测试集用 predict+IoU 匹配算真 PR(conf=0.10 部署, 和 11l 同一套 conf
网格对比), 见 eval_true_pr.py 的写法。
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "土建_渗漏水_yolo")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "yolo_runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# 共享服务器自适应: 空闲判定阈值不写死。0 = 自动: 每卡按"总显存×85%"判定
# (V100 16GB/32GB 都合理); 可被 --gpu-free-threshold 覆盖为固定 MB 值。
GPU_FREE_MEM_MB = 0

# ============================================================


def find_weights(model_name: str) -> str:
    """按 --model 解析权重文件:
      1) --model 本身是存在的文件路径 → 直接用;
      2) --model 是纯文件名 → 依次在 脚本目录/项目根/当前目录 找同名文件;
      3) 都没找到 → 返回原 --model 值, 交给 ultralytics 尝试下载(GitHub 被墙则失败)。
    """
    if os.path.isfile(model_name):
        return model_name
    for p in (SCRIPT_DIR, PROJECT_ROOT, os.getcwd()):
        cand = os.path.join(p, model_name)
        if os.path.isfile(cand):
            return cand
    return model_name


def ensure_data_yaml(data_yaml=None):
    """把 data.yaml 的 path 修正为本机实际路径。

    data_yaml 为空时用默认推导路径(本地 dev 布局);
    显式传 --data 时(服务器布局), 只校验文件存在, 不做 path 修正。
    """
    data_yaml = data_yaml or DATA_YAML
    dataset_dir = os.path.dirname(data_yaml)
    if not os.path.isfile(data_yaml):
        sys.exit(
            f"\n✗ 找不到 data.yaml: {data_yaml}\n"
            f"  默认推导路径: {DATA_YAML}\n"
            f"  若数据在服务器别处, 用 --data <绝对路径 data.yaml> 覆盖。\n"
        )
    if data_yaml != DATA_YAML:
        print(f"  [data] 使用 --data 覆盖: {data_yaml}")
        return
    if not os.path.isdir(dataset_dir):
        sys.exit(f"\n✗ 找不到数据集目录: {dataset_dir}\n")
    new_path = f"path: {dataset_dir}"
    lines = Path(data_yaml).read_text(encoding="utf-8").splitlines()
    changed = False
    for i, ln in enumerate(lines):
        if ln.strip().startswith("path:"):
            if ln.strip() != new_path:
                print(f"  [data.yaml] path 修正: {ln.strip()} → {new_path}")
                lines[i] = new_path
                changed = True
            break
    if changed:
        Path(data_yaml).write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        print(f"  [data.yaml] path 正确: {dataset_dir}")


def parse_nc(data_yaml=None) -> int:
    nc = 1
    data_yaml = data_yaml or DATA_YAML
    if os.path.isfile(data_yaml):
        for line in Path(data_yaml).read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("nc:"):
                try:
                    nc = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    return nc


def parse_splits(data_yaml=None) -> dict:
    """解析 train/val/test 相对目录(兼容 train/images 与 images/train 两种布局)。"""
    splits = {}
    data_yaml = data_yaml or DATA_YAML
    if os.path.isfile(data_yaml):
        for line in Path(data_yaml).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            for key in ("train", "val", "test"):
                if line.startswith(key + ":"):
                    splits[key] = line.split(":", 1)[1].strip()
    return splits


def driver_version() -> str | None:
    """读第一块卡的驱动版本(如 '530.41.03'); 读不到返回 None。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        )
        first = out.strip().splitlines()[0].strip()
        return first or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def check_env() -> bool:
    """V100(Volta sm_70) 环境自检: torch 是否含 CUDA + 编译 arch 是否含 sm_70。

    必须在 CUDA_VISIBLE_DEVICES 锁定之后调用(见 main 内顺序), 否则 torch.cuda.*
    会先把 CUDA 上下文绑到默认物理卡。
    """
    try:
        import torch
    except ImportError:
        print("\n✗ 未安装 torch。V100(Volta sm_70) 装法见脚本头部说明:")
        print("    驱动≥530 → pip install torch==2.4.1 ... --index-url .../cu121")
        print("    驱动520~530 → pip install torch==2.1.2 ... --index-url .../cu118")
        print("    装完先跑本脚本 --check-env 复核。")
        return False

    print(f"\n  torch: {torch.__version__}  build={torch.version.cuda or 'cpu'}")
    dv = driver_version()
    print(f"  驱动: {dv if dv else '读不到(nvidia-smi 不可用?)'}")
    if not torch.cuda.is_available():
        print("  ✗ torch.cuda.is_available()=False → 构建不含 GPU 支持, 或驱动/CUDA 不匹配。")
        print("    nvidia-smi 看驱动版本, 按头部说明换 cu118/cu121 构建(cu130 对老驱动不兼容)。")
        return False

    archs = torch.cuda.get_arch_list()
    print(f"  torch 编译 arch: {archs}")
    has_volta = any(a in ("sm_70", "compute_70") for a in archs)
    if not has_volta:
        print("  ✗ torch 编译 arch 不含 sm_70(Volta) → V100 无法跑, 换 cu118/cu121 构建。")
        print("    cu121: pip install torch==2.4.1 torchvision==0.19.1 "
              "--index-url https://download.pytorch.org/whl/cu121")
        return False

    n = torch.cuda.device_count()
    print(f"  ✅ torch 含 Volta(sm_70), 当前 CUDA_VISIBLE_DEVICES 下可见 {n} 块卡:")
    for i in range(min(n, 8)):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        mark = "  ← V100" if cap == (7, 0) else ""
        print(f"    GPU{i}: {name}  SM{cap[0]}.{cap[1]}{mark}")
        if cap != (7, 0):
            print(f"    ⚠ GPU{i} 不是 Volta(SM7.0) — 本脚本是 V100 版。"
                  f"4090/Turing+ 请用 yolo13l训练_8x4090.py。")
    print("  ✅ 环境通过。V100 无 bf16 → 混合精度固定 fp16(amp=True)。")
    return True


def detect_free_gpus(threshold_mb: int) -> list | None:
    """用 nvidia-smi 探测空闲 GPU, 避开被其他用户占用的卡。

    空闲判定: threshold_mb>0 → 剩余显存≥阈值; ==0(自动) → 每卡总显存×85%。
    返回升序 GPU 索引列表; nvidia-smi 不可用时返回 None。
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        print(f"  ⚠️ nvidia-smi 探测失败({type(e).__name__}), 无法自动选卡。"
              f"请手动 --device 指定。")
        return None
    free = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            idx, used, total = map(int, parts)
        except ValueError:
            continue
        thr = threshold_mb if threshold_mb > 0 else int(total * 0.85)
        if total - used >= thr:
            free.append(idx)
    return sorted(free)


def _thr_desc(mb: int) -> str:
    """空闲阈值的可读描述: 固定值 vs 自动(每卡总显存×85%)。"""
    return f"≥{mb/1024:.0f}GB" if mb > 0 else "每卡总显存×85%(按卡自适应)"


def resolve_device_list(device_arg: str, threshold_mb: int) -> list:
    """解析最终 GPU 设备列表: 手动 --device 优先, 否则自动探测空闲卡。"""
    if device_arg:
        import re
        toks = [x for x in re.split(r"[\s,]+", str(device_arg).strip()) if x]
        try:
            return [int(x) for x in toks]
        except ValueError:
            sys.exit(f"\n✗ --device 解析失败: '{device_arg}'\n"
                     f"  期望逗号/空格分隔的数字索引, 如 '0,1,2,4,5,6,7' 或 '0 1 2'。\n")
    print(f"  🔍 自动探测空闲 GPU (空闲判定: {_thr_desc(threshold_mb)})…")
    free = detect_free_gpus(threshold_mb)
    if free is None:
        sys.exit(
            f"\n✗ 无法自动探测 GPU。\n"
            f"  请用 --device 手动指定(如 --device 0,1,2,4,5,6,7),\n"
            f"  或先执行 nvidia-smi 确认哪些卡空闲。\n"
        )
    if not free:
        sys.exit(
            f"\n✗ 没有探测到空闲 GPU (均未达空闲判定: {_thr_desc(threshold_mb)})。\n"
            f"  共享服务器可能有其他任务占卡。\n"
            f"  → nvidia-smi 查看占用; 可 --device 强行指定自己的卡, 或调低 --gpu-free-threshold。\n"
        )
    print(f"  ✅ 空闲 GPU: {free}")
    return free


def verify_dataset(data_yaml=None) -> bool:
    """数据自检: 图片/标签一一对应, label 行格式/坐标范围/类别编号合法。"""
    data_yaml = data_yaml or DATA_YAML
    dataset_dir = os.path.dirname(data_yaml)
    nc = parse_nc(data_yaml)
    splits = parse_splits(data_yaml)
    print(f"  数据: {dataset_dir}")
    print(f"  data.yaml nc = {nc}  布局 = {splits}")
    ok = True
    for split, rel in splits.items():
        img_dir = os.path.join(dataset_dir, rel)
        lbl_dir = os.path.join(dataset_dir, rel.replace("images", "labels"))
        imgs = sorted(os.listdir(img_dir))
        lbls = sorted(os.listdir(lbl_dir))
        n_lbl = sum(1 for x in lbls if x.endswith(".txt"))
        print(f"  [{split}] images={len(imgs)}  labels={n_lbl}", end="")
        if len(imgs) != n_lbl:
            ok = False
            print("  ✗ 图片/标签数量不一致", end="")
        print()

        bad, empty = 0, 0
        for fn in lbls:
            if not fn.endswith(".txt"):
                continue
            lines = [ln for ln in Path(os.path.join(lbl_dir, fn)).read_text().strip().splitlines() if ln.strip()]
            if not lines:
                empty += 1
                continue
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    bad += 1
                    continue
                try:
                    c, cx, cy, w, h = map(float, parts)
                except ValueError:
                    bad += 1
                    continue
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and w > 0 and h > 0) or c != int(c) or not (0 <= int(c) < nc):
                    bad += 1
        print(f"      非法行={bad}  空标签={empty}")
        if bad:
            ok = False
    print("  数据集校验:", "✓ 通过" if ok else "✗ 有问题, 先修数据再训练")
    return ok


def min_gpu_total_mb(devices: list) -> int | None:
    """取选定卡中最小的总显存(MiB), 用于自动 batch 按每卡容量定档; 失败返回 None。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=30,
        )
        total = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                try:
                    total[int(parts[0])] = int(parts[1])
                except ValueError:
                    pass
        vals = [total[d] for d in devices if d in total]
        return min(vals) if vals else None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def check_model_supported(weights: str):
    """加载前先校验 ultralytics 版本/环境是否支持 yolov13, 给出明确错误而非闷声崩。

    必须在 CUDA_VISIBLE_DEVICES 锁定之后调用(内部 import ultralytics/fork)。
    """
    try:
        import ultralytics
    except ImportError:
        sys.exit("\n✗ 本机未安装 ultralytics。\n"
                 f"  YOLOv13 需要 iMoonLab/yolov13 的 ultralytics 8.3.63 fork\n"
                 f"  (https://github.com/iMoonLab/yolov13), 不是主线程 ultralytics。\n"
                 f"  主线程版本可能报 'unknown task type' / 加载 yolov13l.pt 失败。\n"
                 f"  确认启动时 PYTHONPATH 指向 fork 目录。\n")
    ver = tuple(int(x) for x in ultralytics.__version__.split(".")[:2])
    print(f"  ultralytics 版本: {ultralytics.__version__}")
    if ver < (8, 3):
        sys.exit(f"\n✗ ultralytics {ultralytics.__version__} 过旧, YOLOv13 需要 ≥8.3.63。\n"
                 f"  请装 iMoonLab/yolov13 fork(8.3.63)。\n")
    # 真加载一次, 提前暴露架构不匹配
    from ultralytics import YOLO
    try:
        YOLO(weights)
    except Exception as e:
        sys.exit(f"\n✗ 加载 {weights} 失败: {e}\n"
                 f"  YOLOv13 权重(yolov13l.pt)与 11l 权重(yolo11l.pt)结构不同, "
                 f"且需要 iMoonLab 的 yolov13 fork。\n"
                 f"  确认服务器装的是 fork 而不是主线程 ultralytics。\n")


def main():
    ap = argparse.ArgumentParser(description="渗漏水 YOLOv13-L 检测训练(V100 独立脚本, 自适应卡数)")
    ap.add_argument("--model", default="yolov13l.pt",
                    help="预训练权重名或路径。默认=yolov13l.pt(带v! 不是 yolo13l.pt)。"
                         "下载: https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt"
                         "(服务器 GitHub 被墙, 需本地下载后 rsync)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=512, help="图像尺寸(数据为512², 对齐512; 640会放大1.25x, 显存更高)")
    ap.add_argument("--batch", type=int, default=0,
                    help="总 batch。0(默认)=自动: 探测到卡数+每卡显存后按 "
                         "每卡24(32GB卡)/16(16GB卡)×卡数 设定(先 --probe 探显存再调)。"
                         "也可显式给(须被实际卡数整除, 每卡=batch/卡数)")
    ap.add_argument("--device", default="",
                    help="GPU 组合, 如 '0,1,2,4,5,6,7'(手动指定, 任意卡数)。"
                         "默认空串=自动探测空闲卡(nvidia-smi 查显存, 避开被占用的卡)")
    ap.add_argument("--gpu-free-threshold", type=int, default=GPU_FREE_MEM_MB,
                    help=f"空闲 GPU 判定阈值(MB)。默认 {GPU_FREE_MEM_MB}=自动, 每卡按总显存×85%判定"
                         f"(16/32GB V100 都适用); 传固定值则用固定阈值")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr0", type=float, default=0.00012,
                    help="初始学习率。配合 nbs=跟随batch, 生效LR=lr0(任何版本行为一致)。"
                         "实测 1.2e-4 小数据集稳定收敛")
    ap.add_argument("--optimizer", default="AdamW",
                    help="务必显式! optimizer='auto' 会用 lr_fit 覆盖 lr0 直接发散")
    ap.add_argument("--freeze", type=int, default=0,
                    help="冻结前N层。⚠️ v13 层结构与 11l 不同, 11l 的 freeze=10 在此不适用! "
                         "默认 0=全量微调(420图>50图, 文献结论全量微调优于冻结)。要试再 --freeze 10/20")
    ap.add_argument("--nbs", type=int, default=256,
                    help="nominal batch, 只影响梯度累积(不缩放 LR)。默认 256 → 任意卡数下有效 batch "
                         "都指向 256(1卡batch16+累积16 / 5卡batch80+累积3, 见 fork accumulate=max(round(nbs/batch),1))。"
                         "0=跟随 --batch(生效LR=lr0 恒等)")
    ap.add_argument("--amp", type=lambda x: x.lower() != "false", default=True,
                    help="混合精度。V100 无 bf16, 只有 fp16 → amp=True 即 fp16(本脚本默认, 也是唯一正确设置); "
                         "'--amp false' 可关")
    ap.add_argument("--cache", type=lambda x: x.lower() != "false", default=True,
                    help="RAM 缓存图片(数据小, 开缓存省 PCIe 读盘)")
    ap.add_argument("--workers", type=int, default=8, help="每卡 dataloader 进程数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="yolov13l_imgsz512_v100", help="runs 子目录名, 区分实验")
    ap.add_argument("--data", default="",
                    help="覆盖 data.yaml 绝对路径。默认=脚本上级两级/data/土建_渗漏水_yolo/data.yaml。"
                         "服务器上数据若在别处, 显式传此参数")
    ap.add_argument("--project", default="",
                    help="覆盖 runs 根目录。默认=项目根/yolo_runs")
    ap.add_argument("--check-env", action="store_true",
                    help="只做 V100 环境自检(torch/CUDA/sm_70), 不训练。装好 torch 后第一件事")
    ap.add_argument("--verify-only", action="store_true", help="只校验数据, 不训练(可本地/服务器跑)")
    ap.add_argument("--probe", action="store_true",
                    help="单卡显存探测: 用选中的第一张卡跑1epoch/全量数据, 测每卡batch峰值显存后退出")
    ap.add_argument("--eval-only", action="store_true", help="只跑测试集评估(可 CPU)")
    ap.add_argument("--weights", default="", help="eval-only 时的权重路径")
    ap.add_argument("--tta", action="store_true",
                    help="val/test 开启 test-time augmentation(TTA); 11l@512 实测 TTA 有害, 默认别开")
    args = ap.parse_args()

    # ---------- 数据与权重前置检查 ----------
    # nbs 语义: 默认 256=只做梯度累积(不缩放 LR, fork 8.3.63 的 lr 就是 lr0);
    #           0=跟随 batch(下方探测到卡数后置为 batch, 关闭一切 LR 缩放)。
    data_yaml = args.data or DATA_YAML
    project = args.project or RUNS_DIR
    ensure_data_yaml(data_yaml)
    print(f"  项目根: {PROJECT_ROOT}")
    print(f"  data: {data_yaml}")
    print(f"  project: {project}")

    if args.verify_only:
        sys.exit(0 if verify_dataset(data_yaml) else 1)

    weights = find_weights(args.model)
    if not os.path.isfile(weights):
        print(f"⚠️  未找到权重文件 {args.model}, 尝试自动下载(GitHub 被墙则失败)…")
    else:
        print(f"  ✅ 使用预训练权重: {weights}")

    # ---------- GPU 选择 + 环境锁(共享服务器自适应) ----------
    # 必须在任何 CUDA 调用 / import ultralytics 之前完成! fork 的 block.py 模块级
    # 检查会调用 torch.cuda.get_device_capability(), 把 CUDA 上下文绑到默认物理卡
    # (GPU 0); 一旦上下文建立, 之后设 CUDA_VISIBLE_DEVICES 无效 → 训练实际跑在
    # GPU 0 而非探测选中的卡。注意: check_model_supported / check_env 内部都会触发
    # CUDA, 所以锁必须放在它们前面。
    if not args.eval_only:
        devices = resolve_device_list(args.device, args.gpu_free_threshold)
        n_gpu = len(devices)
        device_str = ",".join(str(d) for d in devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        print(f"  [env] CUDA_VISIBLE_DEVICES={device_str} (在任何 CUDA/ultralytics 调用前锁定)")
    else:
        devices, n_gpu, device_str = None, 0, ""

    # 探测到卡数后: batch=0(自动) → 按每卡显存定档(32GB→24/卡, 16GB→16/卡);
    # nbs=0(跟随) → 置为 batch(关闭一切 LR 缩放)
    if not args.eval_only:
        if args.batch == 0:
            tot = min_gpu_total_mb(devices)
            is_32gb = (tot or 0) >= 28672
            per_card = 24 if is_32gb else 16
            args.batch = per_card * n_gpu
            print(f"  [batch] --batch 未指定 → 自动 = 每卡{per_card}({'32GB' if is_32gb else '16GB'}卡) × "
                  f"{n_gpu}卡 = {args.batch}")
        if args.nbs == 0:
            args.nbs = args.batch
            print(f"  [nbs] --nbs 0(跟随) → nbs={args.batch}, 生效LR=lr0 恒等")

    # V100 环境自检 + YOLOv13 fork 校验 —— 均在 CUDA_VISIBLE_DEVICES 锁定之后
    if not args.eval_only and not check_env():
        sys.exit(1)
    check_model_supported(weights)

    from ultralytics import YOLO  # 延迟导入(此时 CUDA 上下文将落在已锁定的物理卡上)

    # ---------- 显存探测 ----------
    if args.probe:
        per_card = max(1, args.batch // n_gpu)
        probe_dev = devices[0]
        print(f"\n🔬 显存探测: 单卡 batch={per_card} @ imgsz{args.imgsz}, 1 epoch, 全量数据, device={probe_dev}")
        print("   (注意: DDP 峰值比单卡 probe 略高, 建议留 ~3GB 余量)")
        try:
            model = YOLO(weights)
            model.train(
                data=data_yaml, epochs=1, imgsz=args.imgsz, batch=per_card,
                device=str(probe_dev), project=project, name="probe_tmp", exist_ok=True,
                fraction=1.0, amp=args.amp, optimizer=args.optimizer,
                # fraction=1.0: 全量数据, 小数据集(420张)下 fraction=0.1 只剩42张, 测不出真实每卡显存
                freeze=args.freeze, lr0=args.lr0, nbs=args.nbs, seed=args.seed,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️ 原始报错(看 Tried to allocate / already allocated / free 三处):\n{e}\n")
                sys.exit(f"\n✗ 单卡 batch={per_card} 已 OOM。请减小 --batch(如 96/64)后再试。\n")
            raise
        print("\n✅ 探测完成。若日志末尾出现 'Using X.XG / YGB' 且留有余量, 可放心用当前 batch 启动训练。")
        return

    # ---------- 评估 ----------
    if args.eval_only:
        if not args.weights:
            args.weights = os.path.join(project, args.name, "weights", "best.pt")
        if not os.path.isfile(args.weights):
            sys.exit(f"\n✗ 找不到权重: {args.weights}\n")
        print(f"\n🔍 在测试集上评估: {args.weights}  tta={args.tta}")
        model = YOLO(args.weights)
        metrics = model.val(data=data_yaml, split="test", imgsz=args.imgsz, augment=args.tta)
        print(f"\n  mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}  Recall: {metrics.box.mr:.4f}")
        return

    # ---------- 训练 ----------
    if args.batch % n_gpu != 0:
        sys.exit(f"\n✗ batch={args.batch} 不能被 {n_gpu}(卡数)整除, DDP 每卡 batch 需为整数\n")
    per_card = args.batch // n_gpu

    print("=" * 72)
    print(f"🚀 YOLOv13-L 渗漏水检测训练 (V100 服务器, 自适应选卡, Volta sm_70)  实际模型: {os.path.basename(weights)}")
    print(f"  数据: {data_yaml}")
    print(f"  权重文件: {weights}   epochs={args.epochs}  imgsz={args.imgsz}")
    print(f"  总 batch={args.batch} (每卡 {per_card})  device={device_str}")
    print(f"  optimizer={args.optimizer}  lr0={args.lr0}(生效LR=lr0, 已关闭batch/nbs缩放)  "
          f"freeze={args.freeze}(v13 默认全量微调, 勿套 11l 的 10)  nbs={args.nbs}  amp={args.amp}(fp16)")
    print("=" * 72)

    model = YOLO(weights)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,          # 多卡自动均分(每卡 = batch/卡数)
        device=device_str,
        patience=args.patience,
        lr0=args.lr0,
        nbs=args.nbs,
        freeze=args.freeze,
        optimizer=args.optimizer,  # 显式 AdamW, 防 optimizer='auto' 覆盖 lr0 发散
        amp=args.amp,
        cache=args.cache,          # RAM 缓存小数据集, 省 PCIe 读盘
        project=project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        seed=args.seed,
        workers=args.workers,
        verbose=True,
    )

    # 训练完在测试集上评估
    print(f"\n📊 测试集评估: tta={args.tta}")
    metrics = model.val(data=data_yaml, split="test", imgsz=args.imgsz, augment=args.tta)
    print(f"  mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}  Recall: {metrics.box.mr:.4f}")
    print(f"\n✅ 完成! 权重: {os.path.join(project, args.name, 'weights')}/")
    print("   部署建议: 用 predict+IoU 匹配算真 PR(conf=0.10 已定为部署阈值), 别只看 val 的默认 conf=0.25 点。")


if __name__ == "__main__":
    main()
