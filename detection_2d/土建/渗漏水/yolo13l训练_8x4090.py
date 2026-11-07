#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
渗漏水 YOLOv13-L 检测训练 —— 8×RTX4090 版（独立脚本，不碰 11l）
================================================================
数据: data/土建_渗漏水_yolo  (单类 leakage, 420/74/55)
模型: yolov13l.pt —— YOLOv13 Hypergraph 版 L 级 (27.6M 参数, 88.4 GFLOPs, COCO 53.4 mAP)

⚠️ 与 11l 的关键差异（务必先读，别拿 11l 的习惯直接套）:
  - YOLOv13 官方代码是 iMoonLab/yolov13 的 ultralytics 8.3.63 fork，
    不是主线程 ultralytics！服务器 yolo_cu128 环境若版本不匹配会加载失败。
    权重文件名为 yolov13l.pt（带 v），不是 yolo13l.pt。
  - 权重下载: https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt
    服务器 GitHub 被墙 → 必须在本地下载后 rsync 过去，放 脚本目录/项目根 均可。
  - 层结构与 11l 不同（v13 有 Hypergraph 模块），11l 的 freeze=10 在这里
    **不再等于"冻结 backbone"**，可能冻结到网络中间/头部。所以本脚本默认
    --freeze 0（全量微调，420 图 > 50 图，文献结论全量微调优于冻结）。
    若要试冻结，用 --freeze 10/20 自测，别信 11l 的 10。
  - 8.3.63 是旧版 ultralytics，lr0 会按 batch/nbs 缩放；本脚本保持
    nbs=0 → 跟随 batch 的做法，使生效 LR 恒等于 lr0（新旧版本行为一致）。

硬件差异(vs 旧 7×V100-32GB 服务器):
  - 共享服务器 8×RTX4090 24GB, SM8.9(Ada); 但 GPU 可能被其他用户占用
  - 4090 支持原生 fp16 AMP; 8.3.63 无 bf16 训练开关 → amp=True(fp16)
  - 数据仅 549 张 512², PCIe 多卡 DDP 无 I/O 瓶颈(开 cache 减少读盘)

共享服务器 GPU 自适应(沿用 11l 脚本已验证的基建):
  - 默认自动探测空闲卡: nvidia-smi 查每卡剩余显存, ≥20GB(可 --gpu-free-threshold 调)
    才算空闲, 自动避开被其他用户占用的卡; 实际卡数动态推导
  - 也可 --device 手动指定任意组合: --device 0,1,2,4,5,6,7
  - batch 整除校验按实际卡数: --batch 需能被选中卡数整除

用法(共享 8×4090 服务器上):
  0) 前置: 服务器装 iMoonLab yolov13 fork(ultralytics 8.3.63)；
          把 yolov13l.pt、data/ 与整个项目树拷到服务器(保持相对位置)
  1) 校验数据: python3 土建/渗漏水/yolo13l训练_8x4090.py --verify-only
  2) 看哪些卡空闲: nvidia-smi
  3) 显存探测:   python3 土建/渗漏水/yolo13l训练_8x4090.py --probe
  4) 启动训练:   bash 土建/渗漏水/launch_train_4090_yolo13l.sh
  5) 只评估:     python3 土建/渗漏水/yolo13l训练_8x4090.py --eval-only --weights <best.pt>

服务器路径差异(数据不在脚本默认推导位置时):
  默认 DATASET_DIR = 脚本上级两级/data/土建_渗漏水_yolo(本地 dev 布局)。
  服务器上数据若在别处(如 /root/autodl-tmp/leakage_train/data/土建_渗漏水_yolo_enh),
  显式传 --data <绝对路径 data.yaml> 和 --project <runs 根目录> 覆盖。

建议: 训完在测试集用 predict+IoU 匹配算真 PR 曲线(和 11l 同一套 conf 网格对比)，
     见 eval_true_pr.py 的写法。
"""
import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "土建_渗漏水_yolo")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "yolo_runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# 共享服务器自适应: 不再硬编码 8 卡。
GPU_FREE_MEM_MB = 20000       # 空闲判定阈值(MB): 剩余显存≥20GB 才算空闲; 可被 --gpu-free-threshold 覆盖

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
    显式传 --data 时(服务器布局), 只校验文件存在, 不做 path 修正
    (服务器上的 data.yaml 已带正确绝对路径)。
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


def detect_free_gpus(threshold_mb: int) -> list | None:
    """用 nvidia-smi 探测空闲 GPU, 避开被其他用户占用的卡。"""
    import subprocess
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
        if total - used >= threshold_mb:
            free.append(idx)
    return sorted(free)


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
    print(f"  🔍 自动探测空闲 GPU (剩余显存 ≥{threshold_mb/1024:.0f}GB 视为空闲)…")
    free = detect_free_gpus(threshold_mb)
    if free is None:
        sys.exit(
            f"\n✗ 无法自动探测 GPU。\n"
            f"  请用 --device 手动指定(如 --device 0,1,2,4,5,6,7),\n"
            f"  或先执行 nvidia-smi 确认哪些卡空闲。\n"
        )
    if not free:
        sys.exit(
            f"\n✗ 没有探测到空闲 GPU (剩余显存均 <{threshold_mb/1024:.0f}GB)。\n"
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


def check_model_supported(weights: str):
    """加载前先校验 ultralytics 版本/环境是否支持 yolov13, 给出明确错误而非闷声崩。"""
    try:
        import ultralytics
    except ImportError:
        sys.exit("\n✗ 本机未安装 ultralytics。\n"
                 f"  YOLOv13 需要 iMoonLab/yolov13 的 ultralytics 8.3.63 fork\n"
                 f"  (https://github.com/iMoonLab/yolov13), 不是主线程 ultralytics。\n"
                 f"  主线程版本可能报 'unknown task type' / 加载 yolov13l.pt 失败。\n")
    ver = tuple(int(x) for x in ultralytics.__version__.split(".")[:2])
    print(f"  ultralytics 版本: {ultralytics.__version__}")
    if ver < (8, 3):
        sys.exit(f"\n✗ ultralytics {ultralytics.__version__} 过旧, YOLOv13 需要 ≥8.3.63。\n"
                 f"  请装 iMoonLab/yolov13 fork(8.3.63):\n"
                 f"    wget https://github.com/iMoonLab/yolov13/archive/refs/tags/yolov13.zip\n"
                 f"    unzip yolov13.zip && cd yolov13-yolov13 && pip install -e .\n")
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
    ap = argparse.ArgumentParser(description="渗漏水 YOLOv13-L 检测训练(8×RTX4090, 独立脚本)")
    ap.add_argument("--model", default="yolov13l.pt",
                    help="预训练权重名或路径。默认=yolov13l.pt(带v! 不是 yolo13l.pt)。"
                         "下载: https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13l.pt"
                         "(服务器 GitHub 被墙, 需本地下载后 rsync)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=512, help="图像尺寸(数据为512², 对齐512; 640会放大1.25x, 显存更高)")
    ap.add_argument("--batch", type=int, default=128, help="总 batch(须被实际卡数整除, 每卡=batch/卡数)")
    ap.add_argument("--device", default="",
                    help="GPU 组合, 如 '0,1,2,4,5,6,7'(手动指定, 任意卡数)。"
                         "默认空串=自动探测空闲卡(nvidia-smi 查显存, 避开被占用的卡)")
    ap.add_argument("--gpu-free-threshold", type=int, default=GPU_FREE_MEM_MB,
                    help=f"空闲 GPU 判定阈值(MB), 剩余显存≥该值才算空闲。"
                         f"默认 {GPU_FREE_MEM_MB}(≈{GPU_FREE_MEM_MB/1024:.0f}GB, 4090 24GB 上被占卡基本会被排除)")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr0", type=float, default=0.00012,
                    help="初始学习率。配合 nbs=跟随batch, 生效LR=lr0(新旧 ultralytics 行为一致)。"
                         "实测 1.2e-4 小数据集稳定收敛")
    ap.add_argument("--optimizer", default="AdamW",
                    help="务必显式! optimizer='auto' 会用 lr_fit 覆盖 lr0 直接发散")
    ap.add_argument("--freeze", type=int, default=0,
                    help="冻结前N层。⚠️ v13 层结构与 11l 不同, 11l 的 freeze=10 在此不适用! "
                         "默认 0=全量微调(420图>50图, 文献结论全量微调优于冻结)。要试再 --freeze 10/20")
    ap.add_argument("--nbs", type=int, default=0,
                    help="nominal batch, 只影响梯度累积。0(默认)=跟随 --batch, "
                         "使生效LR恒等于lr0(任何 ultralytics 版本都不被 batch/nbs 缩放)")
    ap.add_argument("--amp", type=lambda x: x.lower() != "false", default=True,
                    help="混合精度(fp16, 4090 原生支持); '--amp false' 可关")
    ap.add_argument("--cache", type=lambda x: x.lower() != "false", default=True,
                    help="RAM 缓存图片(数据小, 开缓存省 PCIe 读盘)")
    ap.add_argument("--workers", type=int, default=8, help="每卡 dataloader 进程数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="yolov13l_imgsz512_8x4090", help="runs 子目录名, 区分实验")
    ap.add_argument("--data", default="",
                    help="覆盖 data.yaml 绝对路径。默认=脚本上级两级/data/土建_渗漏水_yolo/data.yaml(本地dev布局)。"
                         "服务器上数据若在 leakage_train/data/土建_渗漏水_yolo_enh/data.yaml, 显式传此参数")
    ap.add_argument("--project", default="",
                    help="覆盖 runs 根目录。默认=项目根/yolo_runs; 服务器基线在 leakage_train/runs_leakage, 传 --project 覆盖")
    ap.add_argument("--verify-only", action="store_true", help="只校验数据, 不训练(可本地/服务器跑)")
    ap.add_argument("--probe", action="store_true",
                    help="单卡显存探测: 用选中的第一张卡跑1epoch/10%数据, 测每卡batch峰值显存后退出")
    ap.add_argument("--eval-only", action="store_true", help="只跑测试集评估")
    ap.add_argument("--weights", default="", help="eval-only 时的权重路径")
    ap.add_argument("--tta", action="store_true",
                    help="val/test 开启 test-time augmentation(TTA); 11l@512 实测 TTA 有害, 默认别开")
    args = ap.parse_args()

    # ---------- 数据与权重前置检查 ----------
    if args.nbs == 0:
        args.nbs = args.batch  # 跟随 batch: 关闭任意版本的 LR 缩放, 生效LR=lr0
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
    # 必须在任何 import ultralytics 之前完成! fork 的 block.py 模块级检查会调用
    # torch.cuda.get_device_capability(), 就此把 CUDA 上下文绑到默认物理卡(GPU 0);
    # 一旦上下文建立, 之后设 CUDA_VISIBLE_DEVICES 无效 → 训练实际跑在 GPU 0 而非
    # 探测选中的卡。注意: check_model_supported 内部就会 import ultralytics, 所以
    # 锁必须放在它前面。
    if not args.eval_only:
        devices = resolve_device_list(args.device, args.gpu_free_threshold)
        n_gpu = len(devices)
        device_str = ",".join(str(d) for d in devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        print(f"  [env] CUDA_VISIBLE_DEVICES={device_str} (在任何 import ultralytics 前锁定, 防 fork 误绑 GPU0)")
    else:
        devices, n_gpu, device_str = None, 0, ""

    # YOLOv13 环境校验(fork 版本/架构不匹配提前报错) —— 此时 CUDA_VISIBLE_DEVICES 已锁定
    check_model_supported(weights)

    from ultralytics import YOLO  # 延迟导入(此时 CUDA 上下文将落在已锁定的物理卡上)

    # ---------- 显存探测 ----------
    if args.probe:
        per_card = max(1, args.batch // n_gpu)
        probe_dev = devices[0]
        print(f"\n🔬 显存探测: 单卡 batch={per_card} @ imgsz{args.imgsz}, 1 epoch, 10% 数据, device={probe_dev}")
        print("   (注意: DDP 峰值比单卡 probe 略高, 建议留 ~3GB 余量)")
        try:
            model = YOLO(weights)
            model.train(
                data=data_yaml, epochs=1, imgsz=args.imgsz, batch=per_card,
                device=str(probe_dev), project=project, name="probe_tmp", exist_ok=True,
                fraction=1.0, amp=args.amp, optimizer=args.optimizer,  # 探测用全量: fraction=0.1 在小数据集(<50张)下只剩4张, 测不出真实每卡显存
                freeze=args.freeze, lr0=args.lr0, nbs=args.nbs, seed=args.seed,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️ 原始报错(看 Tried to allocate / already allocated / free 三处):\n{e}\n")
                sys.exit(f"\n✗ 单卡 batch={per_card} 已 OOM。请减小 --batch(如 96/64)后再试。\n")
            raise
        print("\n✅ 探测完成。若日志末尾出现 'Using X.XG / 24GB' 且留有余量, 可放心用当前 batch 启动 DDP 训练。")
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
    print(f"🚀 YOLOv13-L 渗漏水检测训练 (共享 4090 服务器, 自适应选卡)  实际模型: {os.path.basename(weights)}")
    print(f"  数据: {data_yaml}")
    print(f"  权重文件: {weights}   epochs={args.epochs}  imgsz={args.imgsz}")
    print(f"  总 batch={args.batch} (每卡 {per_card})  device={device_str}")
    print(f"  optimizer={args.optimizer}  lr0={args.lr0}(生效LR=lr0, 已关闭batch/nbs缩放)  "
          f"freeze={args.freeze}(v13 默认全量微调, 勿套 11l 的 10)  nbs={args.nbs}  amp={args.amp}")
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
    metrics = model.val(data=DATA_YAML, split="test", imgsz=args.imgsz, augment=args.tta)
    print(f"  mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}  Recall: {metrics.box.mr:.4f}")
    print(f"\n✅ 完成! 权重: {os.path.join(RUNS_DIR, args.name, 'weights')}/")
    print("   部署建议: 用 predict+IoU 匹配算真 PR(和 11l 同一套 conf 网格对比), 别只看 val 的默认 conf=0.25 点。")


if __name__ == "__main__":
    main()
