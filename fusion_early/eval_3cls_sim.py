#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三分支仿真数据集 —— 测试集检出率评估 (全指标)
================================================================================
用训练好的 best.pt 在【病集留测真图 test】上跑 YOLO val, 输出:
  - 整体:  mAP50 / mAP50-95 / Precision / Recall / F1
  - 逐类:  每类 P / R / mAP50 / mAP50-95 / GT框数 / 检测数 / 检出率
  - 混淆矩阵 + PR / F1 曲线 (保存到 runs/eval_<集>_test/)
  - 逐类一致性: test 各类框数(自 label 统计) vs 预测框数, 用于核对"每类检出率"
核心是 Recall(检出率)=TP/(TP+FN); 其它指标一并给出。

与训练脚本同构: 数据路径用 find_data() 按脚本同目录自动定位(本地/服务器通用),
  test 用 --test-imgsz 1280(百万像素病集) 提升小缺陷检出。

用法:
  python3 eval_3cls_sim.py                          # 测全部三集(默认 weights=best.pt)
  python3 eval_3cls_sim.py --data 土建_3cls_sim     # 只测土建
  python3 eval_3cls_sim.py --weights runs/yolo11l_土建_3cls/weights/best.pt --data 土建
  # 权重查找优先级: --weights > --name(runs/<name>/weights/best.pt) > ./best.pt
"""
import os, sys, argparse, glob

# 按 名称 -> 类别数 -> 类名 (顺序即打印顺序)
DEFAULTS = [
    ("土建_3cls_sim", 3, ["裂缝", "渗漏水", "管片掉块"]),
    ("轨道_3cls_sim", 3, ["断裂", "丢失", "松动歪斜"]),
    ("设备_2cls_sim", 2, ["异物入侵", "管线支架松脱"]),
]


def ds_path(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def resolve_weights(a, ds):
    """权重查找: --weights > runs/<name>/weights/best.pt > <数据集>/runs/<name>/weights/best.pt > ./best.pt"""
    cands = []
    if a.weights:
        cands.append(a.weights)
    else:
        # 训练脚本 project 默认 = 数据集目录/runs, name = yolo11l_<分支>_<cls>
        label = {"土建_3cls_sim": "yolo11l_土建_3cls",
                 "轨道_3cls_sim": "yolo11l_轨道_3cls",
                 "设备_2cls_sim": "yolo11l_设备_2cls"}.get(ds, "")
        cands.append(os.path.join(ds_path(ds), "runs", label, "weights", "best.pt"))
        cands.append(os.path.join("runs", label, "weights", "best.pt"))
        cands.append("best.pt")
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    sys.exit(f"❌ 找不到权重。试过:\n  " + "\n  ".join(cands) +
             "\n 用 --weights 指定, 或先训练(或 --name)。")


def test_gt_boxes(ds):
    """从 test 标签统计每类 GT 框数(核对基准)。"""
    import yaml
    cnt = {}
    txts = glob.glob(os.path.join(ds_path(ds), "labels", "test", "*.txt"))
    for t in txts:
        for ln in open(t, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                cl = int(float(ln.split()[0]))
                cnt[cl] = cnt.get(cl, 0) + 1
    return cnt


def main():
    ap = argparse.ArgumentParser(description="三分支测试集检出率评估(全指标)")
    ap.add_argument("--data", default=None, help="只评估某数据集(缺省=全部3)")
    ap.add_argument("--weights", default=None, help="best.pt 路径(缺省自动找)")
    ap.add_argument("--imgsz", type=int, default=1280, help="test 评估尺寸(百万像素病集, 默认1280)")
    ap.add_argument("--conf", type=float, default=0.001, help="conf 阈值(默认0.001, 求mAP用低阈值)")
    ap.add_argument("--iou", type=float, default=0.6, help="val 的 NMS IoU(默认0.6)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default=None, help="runs/<name>/weights/best.pt (缺省按分支猜测)")
    a = ap.parse_args()

    sel = [x for x in DEFAULTS if a.data is None or x[0] == a.data]
    if not sel:
        sys.exit(f"未知数据集: {a.data}\n可选: {[x[0] for x in DEFAULTS]}")
    try:
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(f"需要 ultralytics: {e}")

    for ds, nc, cn in sel:
        cfg = None
        data_yml = os.path.join(ds_path(ds), "data.yaml")
        w = resolve_weights(a, ds)
        print(f"\n{'='*72}\n【 评估 {ds} 】 权重: {w}\n{'='*72}")
        model = YOLO(w)

        r = model.val(data=data_yml, split="test", imgsz=a.imgsz,
                      conf=a.conf, iou=a.iou, batch=a.batch, plots=True,
                      project=os.path.join(ds_path(ds), "runs"),
                      name=f"eval_{ds}_test", exist_ok=True)
        b = getattr(r, "box", None) if not isinstance(r, (int, float)) else None
        if b is None:
            rd0 = r if isinstance(r, dict) else {}
            print("  ⚠ 无 box 指标" + (
                ", 退化为 results_dict: " + ", ".join(
                    f"{k}={v}" for k, v in rd0.items()) if rd0 else " (空)"))
            continue

        print(f"\n--- 整体 (test {a.imgsz}px) ---")
        print(f"  mAP50    = {b.map50:.4f}")
        print(f"  mAP50-95 = {b.map:.4f}")
        # per-class p/r (可能为数组或标量)
        try:
            mp = b.mp if hasattr(b, "mp") else r.results_dict.get("metrics/precision", None)
            mr = b.mr if hasattr(b, "mr") else r.results_dict.get("metrics/recall", None)
        except Exception:
            mp = mr = None

        def safe_mean(arr):
            try:
                import numpy as np
                a = np.asarray(arr, dtype=float)
                return float(a.mean()) if a.size else float("nan")
            except Exception:
                return float("nan")

        print(f"  Precision= {safe_mean(mp):.4f}   Recall= {safe_mean(mr):.4f}")

        # 逐类: 优先用 per-class 数组(可能因类数少而为标量, 需逐项安全取)
        ap50arr = b.ap50 if hasattr(b, "ap50") and getattr(b, "ap50", None) is not None else None
        aparr = b.ap if hasattr(b, "ap") and b.ap is not None else None

        def safe_at(arr, k, n):
            """从数组/标量安全取第 k 项; 标量视为单个值, 越界/非法回 NaN。"""
            if arr is None:
                return float("nan")
            try:
                import numpy as np
                a = np.asarray(arr, dtype=float)
                if a.ndim == 0:
                    return float(a)
                return float(a[k]) if k < a.size else float("nan")
            except Exception:
                return float("nan")

        gt = test_gt_boxes(ds)
        print(f"\n--- 逐类 (GT框核对 -> 检出率) ---")
        print(f"  {'类别':<12}{'GT框':>6}{'P':>8}{'R':>8}{'F1':>8}{'mAP50':>8}{'mAP50-95':>10}")
        for k in range(nc):
            p = safe_at(mp, k, nc)
            r = safe_at(mr, k, nc)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            a50 = safe_at(ap50arr, k, nc)
            a95 = safe_at(aparr, k, nc)
            g = gt.get(k, 0)
            print(f"  {cn[k]:<12}{g:>6}{p:>8.3f}{r:>8.3f}{f1:>8.3f}{a50:>8.3f}{a95:>10.3f}")

        # 用 results_dict 再补一次整体(更标准口径) —— 防御: 版本/返回结构差异
        rd = getattr(r, "results_dict", None)
        if not isinstance(rd, dict):
            rd = {}
        keys = {k: v for k, v in rd.items() if "map" in k or "precision" in k or "recall" in k}
        if keys:
            print(f"\n--- ultralytics results_dict ---")
            for k in sorted(keys):
                v = keys[k]
                if isinstance(v, (list, tuple)):
                    print(f"  {k}: {', '.join(f'{x:.3f}' for x in v)}")
                else:
                    print(f"  {k}: {v if isinstance(v, str) else f'{v:.4f}'}")
        print(f"  (混淆矩阵/PR/F1曲线 -> {ds_path(ds)}/runs/eval_{ds}_test/)")


if __name__ == "__main__":
    main()
