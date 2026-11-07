#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨道3类仿真数据集 det 模型评估 —— 在外部 test 集(病集留测真图)上验 mAP/精度/召回
================================================================================
类别自动从 data.yaml 读取; 权重默认用同目录 runs/yolo11l_轨道_3cls_sim/weights/best.pt。
独立于训练脚本, 训练完/换权重后都能单独跑。

用法(服务器):
  python test_轨道3cls_sim.py                            # 自动用最新 best.pt 评 test
  python test_轨道3cls_sim.py --weights my.pt            # 指定权重
  python test_轨道3cls_sim.py --split val --weights my.pt # 评 val 集
  python test_轨道3cls_sim.py --predict                  # 额外推理存标注图供人工看框
"""
import argparse, os
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def default_dirs():
    """返回 (data.yaml路径, best.pt路径) —— 都优先取脚本所在目录下的默认位置。"""
    data_yaml = os.path.join(SCRIPT_DIR, "轨道_3cls_sim", "data.yaml")
    if not os.path.exists(data_yaml):
        data_yaml = os.path.join(SCRIPT_DIR, "data.yaml")
    best = os.path.join(SCRIPT_DIR, "runs", "yolo11l_轨道_3cls_sim", "weights", "best.pt")
    return data_yaml, best


def main():
    ap = argparse.ArgumentParser(description="轨道3类仿真数据集模型评估")
    d_yaml, d_best = default_dirs()
    ap.add_argument("--weights", default=d_best, help=f"权重 .pt 路径 (默认 {d_best})")
    ap.add_argument("--data", default=d_yaml, help=f"data.yaml 路径 (默认 {d_yaml})")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"),
                    help="评估哪个 split (默认 test, 即病集留测真图)")
    ap.add_argument("--device", default="0", help="运行设备: 0/1/2/... 或 cpu")
    ap.add_argument("--predict", action="store_true", help="额外对 split 图推理并存标注图")
    a = ap.parse_args()

    print(f"[权重] {os.path.abspath(a.weights)}")
    print(f"[数据] {os.path.abspath(a.data)}  split={a.split}  imgsz={a.imgsz}")

    model = YOLO(a.weights)

    # ---- 1. 验证(指标) ----
    met = model.val(data=a.data, split=a.split, imgsz=a.imgsz,
                    conf=a.conf, device=a.device, verbose=True)
    print("\n===== 整体指标 =====")
    print(f"  mAP50-95 : {met.box.map:.4f}")
    print(f"  mAP50    : {met.box.map50:.4f}")
    print(f"  平均精度 : {met.box.mp:.4f}")
    print(f"  平均召回 : {met.box.mr:.4f}")

    print("\n===== 逐类指标 =====")
    names = model.names if hasattr(model, "names") else met.names
    try:
        for i, n in names.items():
            p = met.box.p[i] if hasattr(met.box.p, "__getitem__") else None
            r = met.box.r[i] if hasattr(met.box.r, "__getitem__") else None
            print(f"  类{i} {n}: P={p:.3f}" if p is not None else f"  类{i} {n}: (无)")
    except Exception as e:
        print(f"  (逐类打印跳过: {e})")
    # 逐类 mAP 用 ap50/ap 字段
    try:
        for i, n in names.items():
            print(f"  - {n}: mAP50={met.box.ap50[i]:.3f}  mAP50-95={met.box.ap[i]:.3f}")
    except Exception as e:
        print(f"  (逐类mAP打印跳过: {e})")

    # ---- 2. 可选: 推理存标注图 ----
    if a.predict:
        src = os.path.join(os.path.dirname(a.data), "images", a.split)
        print(f"\n[推理] source={src}  conf={a.conf}  ...")
        model.predict(source=src, imgsz=a.imgsz, conf=a.conf, save=True)
        print("[推理] 标注图已存到 runs/ 下")

    print("TEST_DONE")


if __name__ == "__main__":
    main()
