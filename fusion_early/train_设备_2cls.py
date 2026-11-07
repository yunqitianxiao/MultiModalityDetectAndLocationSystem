#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设备2类 (异物入侵0 / 管线支架松脱1) yolo11l 训练 —— 防泄漏版
================================================================================
数据: 设备_2cls_sim  (train/val = 仿真帧 1280x720,  test = 病集留测真图, 零重叠)
  类序: 0=异物入侵 1=管线支架松脱
  仿真图 1280x720 -> 训练 --imgsz 640; test 为百万像素病集 -> 单测 --test-imgsz 1280。

直接无参即跑(用默认数据), 或:
  python3 train_设备_2cls.py --epochs 200 --imgsz 640 --per-gpu-batch 8 --device 0

超参沿用土建/轨道模板:  AdamW / lr0=5.5e-5 / freeze=10 / amp=True。
batch 自动 = 每卡batch × 检测到的GPU卡数 (无卡退 CPU)。
"""
import os, sys, argparse
import torch
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(_HERE, "设备_2cls_sim", "data.yaml")
DEFAULT_MODEL = "yolo11l.pt"
CLASSES = ["异物入侵", "管线支架松脱"]


def find_data():
    """自主探测 data.yaml: 优先脚本同目录 ./设备_2cls_sim/data.yaml, 否则回退 DEFAULT_DATA。
    本地(pro/3) 与服务器(~ /linfengyao/3) 均自动定位, 免改 --data。"""
    return DEFAULT_DATA


def parse_args():
    a = argparse.ArgumentParser(description="设备2类 yolo11l 训练(防泄漏)")
    a.add_argument("--data", default=find_data(), help="data.yaml 路径 (内含 train/val/test)")
    a.add_argument("--model", default=DEFAULT_MODEL, help="预训练权重路径或名称(缺则自动下载)")
    a.add_argument("--imgsz", type=int, default=640, help="训练尺寸(仿真1280x720, 默认640)")
    a.add_argument("--test-imgsz", type=int, default=1280, help="测试集评估尺寸(百万像素, 默认1280)")
    a.add_argument("--per-gpu-batch", type=int, default=8)
    a.add_argument("--epochs", type=int, default=200)
    a.add_argument("--patience", type=int, default=50)
    a.add_argument("--freeze", type=int, default=10)
    a.add_argument("--workers-per-gpu", type=int, default=4)
    a.add_argument("--lr0", type=float, default=5.5e-5)
    a.add_argument("--device", default="auto", help="0/1/2/... 或 auto")
    a.add_argument("--name", default="yolo11l_设备_2cls")
    a.add_argument("--project", default=None, help="runs 输出根目录(默认数据目录下 runs)")
    a.add_argument("--skip-test-eval", action="store_true", help="训练完不测 test")
    return a.parse_args()


def auto_device(override):
    if override and override != "auto":
        return override, len(override.split(","))
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        return ",".join(str(i) for i in range(n)), n
    return "cpu", 0


def main():
    a = parse_args()
    if not os.path.isfile(a.data):
        sys.exit(f"找不到 data.yaml: {a.data}\n 用 --data 指定.")

    dev, ng = auto_device(a.device)
    print(f"[GPU] {ng} 张可用卡 -> device={dev}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    batch = max(1, a.per_gpu_batch) * ng if ng > 0 else a.per_gpu_batch
    workers = min(max(1, a.workers_per_gpu) * max(1, ng), os.cpu_count() or 8)
    print(f"[batch] {batch}   [workers] {workers}   [imgsz] {a.imgsz}   [模型] {a.model}")

    project = a.project or os.path.join(os.path.dirname(os.path.abspath(a.data)), "runs")
    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=batch,
        patience=a.patience, device=dev, workers=workers,
        freeze=a.freeze, optimizer="AdamW", lr0=a.lr0, amp=True,
        project=project, name=a.name, exist_ok=True, val=True, plots=True,
    )
    print("TRAIN_DONE")

    best = os.path.join(project, a.name, "weights", "best.pt")
    if not a.skip_test_eval and os.path.isfile(best):
        r = YOLO(best).val(data=a.data, split="test", imgsz=a.test_imgsz, batch=16, plots=True)
        b = getattr(r, "box", None)
        if b is not None:
            print(f"{'='*50}\n[TEST(病集留测)] imgsz={a.test_imgsz}  "
                  f"mAP50={b.map50:.4f}  mAP50-95={b.map:.4f}\n{'='*50}")


if __name__ == "__main__":
    main()
