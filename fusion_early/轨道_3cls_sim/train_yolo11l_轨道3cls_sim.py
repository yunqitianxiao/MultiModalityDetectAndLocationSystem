#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨道扣件 3类仿真数据集 (断裂0/丢失1/松动歪斜2) yolo11l 训练 —— ★可移植版★(任意服务器/本机)
================================================================================
数据集: 轨道_3cls_sim  (pro/3/轨道_3cls_sim)
  由仿真 多seed 8类 -> 轨道3类, 另加病集留测真图 test(防泄漏)。
  train/val: 仿真帧;  test: 病集留测真图(训练时未用于贴图, 独立外部评测)。
  类别(读自 data.yaml): 0=断裂  1=丢失  2=松动歪斜   (正常扣件不框, 作负样本)

【上传后怎么跑】(若服务器无GPU, 自动退CPU; 无 yolo11l.pt 自动从 ultralytics 下载)
  把 3 个东西放到服务器同一目录:
     本脚本        train_yolo11l_轨道3cls_sim.py
     数据集        轨道_3cls_sim/        (解压 tar 得到, 内含 data.yaml/images/labels)
     预训练权重    yolo11l.pt           (没有则自动下载)
  然后:
     python train_yolo11l_轨道3cls_sim.py
  或自定义:
     python train_yolo11l_轨道3cls_sim.py --epochs 150 --imgsz 1024 --batch 32 --device 0
     python train_yolo11l_轨道3cls_sim.py --skip-test-eval   # 训练完不自动评外部 test

【做了什么】
  1) 自动定位数据集 & 把 data.yaml 重写为"相对路径"(去掉写死绝对 path), 不再报 FileNotFoundError。
  2) 自动探测 GPU: 有卡用全部卡(多卡逗号分隔), 无卡退 CPU。
  3) batch 自动 = 每卡batch(默认8) × 检测到的GPU数 —— 自适应 4090 卡数。
  4) 训练结束用【最优权重】在外部 test 集上评估并打印 mAP/逐类结果 (data.yaml 不含 test 则跳过)。

【超参】沿用土建/轨道模板: AdamW / lr0=5.5e-5 / freeze=10 / imgsz=1024 / amp=True
  例: 2×4090 -> batch 16;  4×4090 -> batch 32。
  轨道扣件是小目标, 默认 imgsz=1024 更准; 显存不够降到 640。
"""
import os, sys, argparse, glob
import torch
from ultralytics import YOLO

DEFAULT = dict(
    epochs=100, imgsz=1024, batch=None, patience=50,
    freeze=10, optimizer="AdamW", lr0=5.5e-5, workers=8, amp=True,
)
PER_GPU_BATCH = 8   # 单张 4090(24GB) @ imgsz1024 的适配值; 换卡/换尺寸可调此值或 --per_gpu_batch


def find_data_yaml(dataset_dir):
    """从 数据目录 或 显式 data.yaml 路径 里找到 data.yaml; 返回绝对路径。"""
    if os.path.isfile(dataset_dir):
        return os.path.abspath(dataset_dir)
    for cand in ("data.yaml",):
        p = os.path.join(dataset_dir, cand)
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"在 {dataset_dir}/ 下没找到 data.yaml。请确认解压位置或 --data 指对。")


def make_portable_data_yaml(yaml_path):
    """把 data.yaml 重写为相对路径(删掉绝对 path)并返回类别名。ultralytics 在无 path 时
    自动以 yaml 所在目录作数据集根, 因此数据集挪到任何机器都能训。"""
    import yaml
    with open(yaml_path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    d.pop("path", None)
    keep = {k: d[k] for k in ("train", "val", "test", "names", "nc") if k in d}
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(keep, f, allow_unicode=True, sort_keys=False)
    return d.get("names")


def _bool(v):
    """命令行 --amp 的 bool 解析: 正确识别 False。"""
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y")


def detect_device(override):
    """返回 (device字符串, gpu数)。override 非 auto 时按手写值走, 否则自适应全部卡/USB。"""
    if override and override != "auto":
        return override, len(override.split(","))
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        return ",".join(str(i) for i in range(n)), n   # 自适应: 全部卡
    return "cpu", 0


def evaluate_test(best, data_yaml, imgsz):
    """用最优权重在外部 test 集上评估并打印整体+逐类指标。"""
    print(f"[评估] 用最优权重 {best} 在外部 test 集上评估 ...")
    m = YOLO(best).val(data=data_yaml, split="test", imgsz=imgsz, batch=16, plots=True)
    box = getattr(m, "box", None)
    if box is None:
        print("[TEST结果] 已输出, 但未读到 box 指标字段。")
        return
    print(f"[TEST结果] mAP50={box.map50:.4f}  mAP50-95={box.map:.4f}")
    try:
        names = getattr(m, "names", None) or m.names
        for i, n in names.items():
            print(f"  {n}: P={box.p[i]:.3f}  R={box.r[i]:.3f}  "
                  f"mAP50={box.ap50[i]:.3f}  mAP50-95={box.ap[i]:.3f}")
    except Exception as e:
        print(f"  (逐类打印跳过: {e})")


def main():
    ap = argparse.ArgumentParser(description="轨道3类 yolo11l 仿真数据集训练(可移植)")
    ap.add_argument("--data", default=None, help="数据集目录或 data.yaml 路径")
    ap.add_argument("--model", default="yolo11l.pt", help="预训练权重路径或名称")
    for k, v in DEFAULT.items():
        if k == "batch":
            continue  # batch 单独处理(自动)
        ap.add_argument(f"--{k}",
                        type=int if k in ("epochs", "imgsz", "patience", "freeze", "workers")
                        else (float if k == "lr0" else (_bool if k == "amp" else str)),
                        default=v)
    ap.add_argument("--batch", type=int, default=None, help="总batch; 不填=自动(每卡×卡数)")
    ap.add_argument("--per_gpu_batch", type=int, default=PER_GPU_BATCH, help="单卡batch(自动时用)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--project", default=None, help="runs 输出根目录, 默认=数据集目录")
    ap.add_argument("--name", default="yolo11l_轨道_3cls_sim", help="run名")
    ap.add_argument("--skip-test-eval", action="store_true",
                    help="训练结束后不自动评估外部 test 集")
    args = ap.parse_args()

    # 1) 定位数据集 + data.yaml: 默认取脚本所在目录下的 轨道_3cls_sim / data / .
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.data:
        data_root = os.path.abspath(args.data)
    else:
        cand = [d for d in ("轨道_3cls_sim", "data", ".")
                if glob.glob(os.path.join(script_dir, d, "data.yaml"))]
        data_root = os.path.join(script_dir, cand[0]) if cand else script_dir
    yaml_path = find_data_yaml(data_root)
    print(f"[数据集] data.yaml = {yaml_path}")
    names = make_portable_data_yaml(yaml_path)
    print(f"[数据集] 类别 = {names}")
    print("[数据集] 已重写为相对路径 data.yaml(可移植)")

    # 2) 设备自动探测
    device, gpu_count = detect_device(args.device)
    print(f"[设备] device = {device}  (GPU数={gpu_count} / "
          f"机器可用={torch.cuda.device_count() if torch.cuda.is_available() else 0})")

    # 3) batch: 自动 = 每卡batch × 卡数 (自适应 4090 卡数)
    if args.batch is not None:
        batch = args.batch
        src = "手动指定"
    else:
        per_gpu = max(1, args.per_gpu_batch)
        batch = per_gpu * gpu_count if gpu_count > 0 else per_gpu
        src = f"自动 = {per_gpu}/卡 × {gpu_count}卡"
    print(f"[batch] {batch}  ({src})")

    # 4) 训练
    model = YOLO(args.model)
    kwargs = {k: getattr(args, k) for k in DEFAULT}
    kwargs["batch"] = batch
    if isinstance(kwargs["amp"], str):          # 命令行字符串 -> bool
        kwargs["amp"] = kwargs["amp"].lower() in ("1", "true", "yes")
    project = args.project or os.path.dirname(yaml_path)
    print(f"[训练] 模型={args.model}  数据集类别={names}  batch={batch}  project={project}")

    model.train(
        data=yaml_path, **kwargs, device=device,
        project=project, name=args.name, exist_ok=True, val=True, plots=True,
    )
    print("TRAIN_DONE")

    # 5) 最优权重 -> 外部 test 集评估
    if not args.skip_test_eval:
        best = os.path.join(project, args.name, "weights", "best.pt")
        if os.path.exists(best):
            evaluate_test(best, yaml_path, args.imgsz)
        else:
            print(f"[评估] 未找到最优权重 {best}, 跳过外部 test 集评估。")


if __name__ == "__main__":
    main()
