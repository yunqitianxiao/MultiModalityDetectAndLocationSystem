#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单类裂缝图像检测器 (yolo11l, nc=1) —— 三模态融合的"图像支"来源
================================================================
数据: crack_1cls_big/data.yaml (由 data_gen/build_tujian_3cls_from_sim.py
      从合并后的配对数据 output_crack_big 抽出的单类 train/val, 零重叠)。
仿真图 1280x720 -> --imgsz 640 (与点云辅助融合模型的 YOLO_SIZE=640 同尺度,
且省显存; 640 下裂缝框中位约 29x59 px, 对 YOLO 足够)。

产物 best.pt 交给 fusion/train_assist.py 的 --img-ckpt 当图像主干。

用法:
  python train_image_crack.py \
      --data ../inspection_robot/data_gen/crack_1cls_big/data.yaml \
      --model ../weights/yolo11l.pt --epochs 150 --name yolo11l_crack_1cls
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_IR = os.environ.get('IR_ROOT', os.path.abspath(os.path.join(_HERE, '..', 'inspection_robot')))
if _IR not in sys.path:
    sys.path.insert(0, _IR)

# 本机 torchvision 是 0.28.0+cpu: CUDA 上缺 torchvision::nms。训练循环不碰 NMS,
# 但每个 epoch 末的内部 validate() 会对 CUDA 框做 NMS -> NotImplementedError。
# 必须在 import ultralytics 之前打补丁(它按属性访问 torchvision.ops.nms)。
from fix_nms import patch_torchvision_nms  # noqa: E402

patch_torchvision_nms()

import torch            # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULT_DATA = os.path.join(_IR, 'data_gen', 'crack_1cls_big', 'data.yaml')
DEFAULT_MODEL = os.path.abspath(os.path.join(_HERE, '..', 'weights', 'yolo11l.pt'))


def parse_args():
    a = argparse.ArgumentParser(description="单类裂缝 yolo11l 训练 (nc=1)")
    a.add_argument("--data", default=DEFAULT_DATA)
    a.add_argument("--model", default=DEFAULT_MODEL)
    a.add_argument("--imgsz", type=int, default=640)
    a.add_argument("--per-gpu-batch", type=int, default=8)
    a.add_argument("--epochs", type=int, default=150)
    a.add_argument("--patience", type=int, default=40)
    a.add_argument("--freeze", type=int, default=10, help="冻结前 N 层(小数据防过拟合)")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--lr0", type=float, default=1e-3)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--device", default="auto")
    a.add_argument("--project", default=None)
    a.add_argument("--name", default="yolo11l_crack_1cls")
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
        sys.exit(f"找不到 data.yaml: {a.data}")

    dev, ng = auto_device(a.device)
    batch = max(1, a.per_gpu_batch) * ng if ng > 0 else a.per_gpu_batch
    project = a.project or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    print(f"[GPU] {ng} 张卡 device={dev} | batch={batch} imgsz={a.imgsz} | {a.model}", flush=True)

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=batch,
        patience=a.patience, device=dev, workers=a.workers,
        freeze=a.freeze, optimizer="AdamW", lr0=a.lr0, amp=True,
        seed=a.seed, project=project, name=a.name, exist_ok=True,
        val=True, plots=True,
    )
    print("TRAIN_DONE", flush=True)

    best = os.path.join(project, a.name, "weights", "best.pt")
    if os.path.isfile(best):
        r = YOLO(best).val(data=a.data, split="val", imgsz=a.imgsz, batch=16, plots=True,
                           project=project, name=f"{a.name}_val")   # 不传 project 会写到 cwd 推的 runs/
        b = getattr(r, "box", None)
        if b is not None:
            print(f"{'=' * 56}\n[VAL] nc=1 crack  mAP50={b.map50:.4f}  mAP50-95={b.map:.4f}"
                  f"  P={b.mp:.4f}  R={b.mr:.4f}\n{'=' * 56}")
        print(f"[best] {best}")


if __name__ == "__main__":
    main()
