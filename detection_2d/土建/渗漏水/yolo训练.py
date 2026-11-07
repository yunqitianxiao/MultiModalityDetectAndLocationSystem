#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
渗漏水 YOLO 检测训练 (ultralytics YOLOv8)
数据: data/土建_渗漏水_yolo  (由 土建三病害掩膜转YOLO框.py 从掩膜生成, 单类 leakage)
目标: 7×V100-32GB 分布式训练

用法(远程 AutoDL, pro2venv):
  # 首次需安装 ultralytics
  pip install ultralytics
  # 训练
  python3 土建/渗漏水/yolo训练.py --device 0,1,2,3,4,5,6 --batch 112
  # 只测测试集(已有权重)
  python3 土建/渗漏水/yolo训练.py --eval-only --weights <runs/.../best.pt>
"""
import argparse
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_YAML = os.path.join(PROJECT_ROOT, "data/土建_渗漏水_yolo/data.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "yolo_runs")
os.makedirs(RUNS_DIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_YAML)
    ap.add_argument("--model", default="yolov8m.pt", help="预训练权重或已有 best.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=512, help="图像尺寸(数据为512², 对齐512; 640会放大1.25x)")
    ap.add_argument("--batch", type=int, default=112, help="总 batch(多卡均分, 7卡×16=112)")
    ap.add_argument("--device", default="0,1,2,3,4,5,6", help="7×V100 全卡, 单卡调试用 0")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr0", type=float, default=0.0001,
                    help="初始学习率。注意 ultralytics 按 batch/nbs=64 自动缩放, 7卡batch112 时实际生效 = lr0×1.75。"
                         "实测: 生效~1.2e-4 训练稳定收敛(mAP50 0.88), 生效1.75e-3 会发散崩到预测不到框")
    ap.add_argument("--freeze", type=int, default=10, help="冻结前N层(10=backbone), 防微调发散; 0=全不冻结")
    ap.add_argument("--name", default="leakage_yolo", help="runs 子目录名, 区分实验")
    ap.add_argument("--nbs", type=int, default=64, help="nominal batch size(ultralytics 用于 LR 缩放), 传实际 batch 可关闭缩放")
    ap.add_argument("--eval-only", action="store_true", help="只跑测试集评估")
    ap.add_argument("--weights", default="", help="eval-only 时的权重")
    args = ap.parse_args()

    from ultralytics import YOLO  # 延迟导入, 便于 py_compile 语法检查

    if args.eval_only:
        if not args.weights:
            args.weights = os.path.join(RUNS_DIR, "leakage_yolo", "weights", "best.pt")
        model = YOLO(args.weights)
        print(f"🔍 在测试集上评估: {args.weights}")
        metrics = model.val(data=args.data, split="test", imgsz=args.imgsz)
        print(f"\nmAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}  Recall: {metrics.box.mr:.4f}")
        return

    print("=" * 70)
    print("🚀 YOLOv8 渗漏水检测训练")
    print(f"  数据: {args.data}")
    print(f"  模型: {args.model}   epochs={args.epochs}  imgsz={args.imgsz}")
    print(f"  总 batch={args.batch}  device={args.device}")
    print("=" * 70)

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,          # 多卡自动均分(每卡 = batch/卡数)
        device=args.device,
        patience=args.patience,
        lr0=args.lr0,              # 小数据集降 LR 防发散
        nbs=args.nbs,              # 默认64: batch112 时 LR 实际 ×1.75; 传 --nbs 112 可关闭缩放
        freeze=args.freeze,        # 冻结 backbone 防发散
        project=RUNS_DIR,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        cache=True,
        verbose=True,
        workers=8,
    )

    # 训练完在测试集上评估
    print("\n📊 测试集评估:")
    metrics = model.val(data=args.data, split="test", imgsz=args.imgsz)
    print(f"  mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}  Recall: {metrics.box.mr:.4f}")
    print(f"\n✅ 完成! 权重: {RUNS_DIR}/leakage_yolo/weights/")


if __name__ == "__main__":
    main()
