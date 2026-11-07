#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
渗漏水 YOLO 检测训练 —— 8×RTX4090 + yolo11x 适配版
=====================================================
数据: data/土建_渗漏水_yolo  (由 土建三病害掩膜转YOLO框.py 从掩膜生成, 单类 leakage, 420/74/55)
模型: yolo11x.pt (357 层, 25.37M 参数, 87.9 GFLOPs, 与 yolo8m 同量级)

硬件差异(vs 旧 7×V100-32GB 服务器):
  - 共享服务器 8×RTX4090 24GB, SM8.9(Ada); 但 GPU 可能被其他用户占用
  - 4090 支持原生 fp16 AMP; ultralytics 8.4.120 无 bf16 训练开关 → amp=True(fp16)
  - 数据仅 549 张 512², PCIe 多卡 DDP 无 I/O 瓶颈(开 cache 减少读盘)

⚠️ 8.4.120 与旧版 ultralytics 的关键差异(踩坑记录, 勿回退):
  - lr0 不再按 batch/nbs 自动缩放 → 生效 LR = lr0 本身(旧版 "×1.75" 的坑不适用)
  - optimizer 务必显式传 "AdamW"! optimizer="auto" 会用 lr_fit=0.002(nc=1) 覆盖 lr0, 必发散
  - nbs 只影响梯度累积, 不影响 LR; 默认跟随 --batch, 使生效LR=lr0(兼容任意 ultralytics 版本的缩放行为)
  - DDP 下 OOM 不会自动减半 batch(会直接崩), 先用 --probe 探显存

共享服务器 GPU 自适应(本版核心改动):
  - 默认自动探测空闲卡: nvidia-smi 查每卡剩余显存, ≥20GB(可 --gpu-free-threshold 调)
    才算空闲, 自动避开被其他用户占用的卡(如 gpu3); 实际卡数动态推导, 不再硬编码 8
  - 也可 --device 手动指定任意组合: --device 0,1,2,4,5,6,7 (或 "0 1 2")
  - batch 整除校验按实际卡数: --batch 需能被选中卡数整除
  - 无空闲卡/显存不足时给出明确报错提示, 不会闷声崩

用法(共享 8×4090 服务器上):
  0) 前置: pip install -U ultralytics; 把 yolo11x.pt、data/ 与整个项目树拷到服务器
          (保持相对位置; 服务器 GitHub 被墙, yolo11x.pt 需本地 rsync 过去)
  1) 校验数据:   python3 土建/渗漏水/yolo训练_8x4090.py --verify-only
  2) 看哪些卡空闲: nvidia-smi
  3) 显存探测:   python3 土建/渗漏水/yolo训练_8x4090.py --probe
     (单卡探存, 自动选第一张空闲卡; 或 --probe --device 3 指定卡)
  4) 启动训练:   bash 土建/渗漏水/launch_train_4090.sh        # 自动选空闲卡
     (手动指定:  bash 土建/渗漏水/launch_train_4090.sh --device 0,1,2,4,5,6,7)
  5) 只评估:     python3 土建/渗漏水/yolo训练_8x4090.py --eval-only --weights <best.pt>

建议: 训完在测试集用 predict+IoU 匹配算真 PR 曲线(conf=0.10 部署, 已实测 R93.2/P78.1), 见
      /root/autodl-tmp/leakage_train/compare_pr.py 的写法。
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
# 默认自动探测空闲 GPU, 避开被其他用户占用的卡(如 gpu3), 实际卡数由探测结果推导。
GPU_FREE_MEM_MB = 20000       # 空闲判定阈值(MB): 剩余显存≥20GB 才算空闲(4090 共 24GB, 留 4GB 余量); 可被 --gpu-free-threshold 覆盖

# ============================================================


def find_weights(model_name: str) -> str:
    """按 --model 解析权重文件(不再写死 yolo11x.pt):
      1) --model 本身是存在的文件路径 → 直接用;
      2) --model 是纯文件名(如 yolo11x.pt) → 依次在 脚本目录/项目根/当前目录 找同名文件;
      3) 都没找到 → 返回原 --model 值, 交给 ultralytics 尝试下载(GitHub 被墙则失败)。
    """
    if os.path.isfile(model_name):
        return model_name
    for p in (SCRIPT_DIR, PROJECT_ROOT, os.getcwd()):
        cand = os.path.join(p, model_name)
        if os.path.isfile(cand):
            return cand
    return model_name


def ensure_data_yaml():
    """把 data.yaml 的 path 修正为本机实际路径。

    旧版转框脚本生成的是 /home/ylf 绝对路径; 服务器上拷贝后必须指向服务器路径。
    """
    if not os.path.isdir(DATASET_DIR):
        sys.exit(
            f"\n✗ 找不到数据集目录: {DATASET_DIR}\n"
            f"  请把整个项目树(含 data/)拷到服务器并保持相对位置,\n"
            f"  或在服务器上重跑 土建三病害掩膜转YOLO框.py 重新生成 data.yaml。\n"
        )
    if not os.path.isfile(DATA_YAML):
        sys.exit(f"\n✗ 找不到 {DATA_YAML}\n")
    new_path = f"path: {DATASET_DIR}"
    lines = Path(DATA_YAML).read_text(encoding="utf-8").splitlines()
    changed = False
    for i, ln in enumerate(lines):
        if ln.strip().startswith("path:"):
            if ln.strip() != new_path:
                print(f"  [data.yaml] path 修正: {ln.strip()} → {new_path}")
                lines[i] = new_path
                changed = True
            break
    if changed:
        Path(DATA_YAML).write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        print(f"  [data.yaml] path 正确: {DATASET_DIR}")


def parse_nc() -> int:
    nc = 1
    if os.path.isfile(DATA_YAML):
        for line in Path(DATA_YAML).read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("nc:"):
                try:
                    nc = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    return nc


def parse_splits() -> dict:
    """解析 train/val/test 相对目录(兼容 train/images 与 images/train 两种布局)。"""
    splits = {}
    if os.path.isfile(DATA_YAML):
        for line in Path(DATA_YAML).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            for key in ("train", "val", "test"):
                if line.startswith(key + ":"):
                    splits[key] = line.split(":", 1)[1].strip()
    return splits


def detect_free_gpus(threshold_mb: int) -> list | None:
    """用 nvidia-smi 探测空闲 GPU, 避开被其他用户占用的卡。

    空闲判定: 剩余显存 >= threshold_mb 才视为"空闲"(默认 20000MB,
    即 4090 24GB 上被占用的卡基本都会被排除)。
    返回升序 GPU 索引列表; nvidia-smi 不可用时返回 None(调用方需提示)。
    """
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
    """解析最终 GPU 设备列表: 手动 --device 优先, 否则自动探测空闲卡。

    - device_arg 非空 → 用户手动指定, 原样解析为 int 列表。
    - device_arg 为空 → nvidia-smi 自动探测空闲卡; 无可用的卡则报错退出。
    """
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


def verify_dataset() -> bool:
    """数据自检: 图片/标签一一对应, label 行格式/坐标范围/类别编号合法。"""
    nc = parse_nc()
    splits = parse_splits()
    print(f"  数据: {DATASET_DIR}")
    print(f"  data.yaml nc = {nc}  布局 = {splits}")
    ok = True
    for split, rel in splits.items():
        img_dir = os.path.join(DATASET_DIR, rel)
        lbl_dir = os.path.join(DATASET_DIR, rel.replace("images", "labels"))
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


def main():
    ap = argparse.ArgumentParser(description="渗漏水 YOLO 检测训练(8×RTX4090 + yolo11x)")
    ap.add_argument("--model", default="yolo11x.pt",
                    help="预训练权重名或路径。⚠️ 默认=yolo11x.pt(基线)! 跑 YOLO11x 必须显式加 --model yolo11x.pt, "
                         "且服务器上必须已有 yolo11x.pt 文件(无则下载, GitHub 被墙会失败)")
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
                    help="初始学习率。8.4.120 不再按 batch/nbs 缩放, 生效LR=lr0。"
                         "实测生效 ~1.2e-4 稳定收敛(小数据集微调)")
    ap.add_argument("--optimizer", default="AdamW",
                    help="务必显式! optimizer='auto' 会用 lr_fit 覆盖 lr0 直接发散")
    ap.add_argument("--freeze", type=int, default=10, help="冻结前N层(10≈backbone), 防微调发散; 0=全不冻结")
    ap.add_argument("--nbs", type=int, default=0,
                    help="nominal batch, 只影响梯度累积。0(默认)=跟随 --batch, "
                         "使生效LR恒等于lr0(任何 ultralytics 版本都不被 batch/nbs 缩放)")
    ap.add_argument("--amp", type=lambda x: x.lower() != "false", default=True,
                    help="混合精度(fp16, 4090 原生支持); '--amp false' 可关")
    ap.add_argument("--cache", type=lambda x: x.lower() != "false", default=True,
                    help="RAM 缓存图片(数据小, 开缓存省 PCIe 读盘)")
    ap.add_argument("--workers", type=int, default=8, help="每卡 dataloader 进程数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="yolo11x_imgsz512_8x4090", help="runs 子目录名, 区分实验")
    ap.add_argument("--verify-only", action="store_true", help="只校验数据, 不训练(可本地/服务器跑)")
    ap.add_argument("--probe", action="store_true",
                    help="单卡显存探测: 用选中的第一张卡跑1epoch/10%数据, 测每卡batch峰值显存后退出")
    ap.add_argument("--eval-only", action="store_true", help="只跑测试集评估")
    ap.add_argument("--weights", default="", help="eval-only 时的权重路径")
    ap.add_argument("--tta", action="store_true",
                    help="val/test 开启 test-time augmentation(TTA), 推理增强可能再提几个点; "
                         "8.4.120 的开关是 model.val(augment=True), 评估/训练完的测试集评估都会生效")
    args = ap.parse_args()

    # ---------- 数据与权重前置检查 ----------
    if args.nbs == 0:
        args.nbs = args.batch  # 跟随 batch: 关闭任意版本的 LR 缩放, 生效LR=lr0
    ensure_data_yaml()
    print(f"  项目根: {PROJECT_ROOT}")

    if args.verify_only:
        sys.exit(0 if verify_dataset() else 1)

    weights = find_weights(args.model)
    if not os.path.isfile(weights):
        print(f"⚠️  未找到权重文件 {args.model}, 尝试自动下载(GitHub 被墙则失败)…")
    else:
        print(f"  ✅ 使用预训练权重: {weights}")

    from ultralytics import YOLO  # 延迟导入, 便于本地 py_compile 语法检查

    # ---------- GPU 选择(共享服务器自适应) ----------
    # probe / train 前解析实际设备: 手动 --device 优先, 否则自动探测空闲卡。
    # eval-only 不需要 GPU 探测(可 CPU 跑), 跳过解析避免误报"无空闲卡"。
    if not args.eval_only:
        devices = resolve_device_list(args.device, args.gpu_free_threshold)
        n_gpu = len(devices)
        device_str = ",".join(str(d) for d in devices)
    else:
        devices, n_gpu, device_str = None, 0, ""

    # ---------- 显存探测 ----------
    if args.probe:
        per_card = max(1, args.batch // n_gpu)
        probe_dev = devices[0]
        print(f"\n🔬 显存探测: 单卡 batch={per_card} @ imgsz{args.imgsz}, 1 epoch, 10% 数据, device={probe_dev}")
        print("   (注意: DDP 峰值比单卡 probe 略高, 建议留 ~3GB 余量)")
        try:
            model = YOLO(weights)
            model.train(
                data=DATA_YAML, epochs=1, imgsz=args.imgsz, batch=per_card,
                device=str(probe_dev), project=RUNS_DIR, name="probe_tmp", exist_ok=True,
                fraction=0.1, amp=args.amp, optimizer=args.optimizer,
                freeze=args.freeze, lr0=args.lr0, nbs=args.nbs, seed=args.seed,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                sys.exit(f"\n✗ 单卡 batch={per_card} 已 OOM。请减小 --batch(如 96/64)后再试。\n")
            raise
        print("\n✅ 探测完成。若日志末尾出现 'Using X.XG / 24GB' 且留有余量, 可放心用当前 batch 启动 DDP 训练。")
        return

    # ---------- 评估 ----------
    if args.eval_only:
        if not args.weights:
            args.weights = os.path.join(RUNS_DIR, args.name, "weights", "best.pt")
        if not os.path.isfile(args.weights):
            sys.exit(f"\n✗ 找不到权重: {args.weights}\n")
        print(f"\n🔍 在测试集上评估: {args.weights}  tta={args.tta}")
        model = YOLO(args.weights)
        metrics = model.val(data=DATA_YAML, split="test", imgsz=args.imgsz, augment=args.tta)
        print(f"\n  mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}  Recall: {metrics.box.mr:.4f}")
        return

    # ---------- 训练 ----------
    if args.batch % n_gpu != 0:
        sys.exit(f"\n✗ batch={args.batch} 不能被 {n_gpu}(卡数)整除, DDP 每卡 batch 需为整数\n")
    per_card = args.batch // n_gpu

    print("=" * 72)
    print(f"🚀 YOLO 渗漏水检测训练 (共享 4090 服务器, 自适应选卡)  实际模型: {os.path.basename(weights)}")
    print(f"  数据: {DATA_YAML}")
    print(f"  权重文件: {weights}   epochs={args.epochs}  imgsz={args.imgsz}")
    print(f"  总 batch={args.batch} (每卡 {per_card})  device={device_str}")
    print(f"  optimizer={args.optimizer}  lr0={args.lr0}(生效LR=lr0, 已关闭batch/nbs缩放)  "
          f"freeze={args.freeze}  nbs={args.nbs}(仅梯度累积)  amp={args.amp}")
    print("=" * 72)

    model = YOLO(weights)

    model.train(
        data=DATA_YAML,
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
        project=RUNS_DIR,
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
    print("   部署建议: 用 predict+IoU 匹配算真 PR(conf=0.10 已定为部署阈值), 别只看 val 的默认 conf=0.25 点。")


if __name__ == "__main__":
    main()
