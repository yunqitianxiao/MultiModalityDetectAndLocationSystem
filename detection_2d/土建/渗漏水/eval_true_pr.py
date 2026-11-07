#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真 Precision/Recall 评估: predict + IoU 匹配 (与 yolo训练_8x4090.py 同一套路径解析)
================================================================================
背景: 测试集 recall @ conf=0.25 已达 92.5%, 但硬指标要 95% 检出率。
本脚本把推理 conf 降到 0.02, 用 predict + IoU 匹配算"真实能达到的 P/R" (部署 conf 参考)。

做法:
  - 在 data.yaml 的 test 划分上跑 model.predict (单类 leakage)
  - 对每个 conf 阈值做置信度过滤 → 与 GT 框做贪心 IoU 匹配 (IoU≥--iou 算 TP, 一一对应)
  - 输出表格: conf → P / R / TP / FP / FN
  - 再列出"最低 conf 下仍漏检"的图片名 + 漏检框数 (方便定位哪几张检不出)

用法 (服务器上, conda activate yolo_cu128):
  python3 土建/渗漏水/eval_true_pr.py --weights <best.pt 路径>
  # 自定义: --conf "0.25 0.10 0.05 0.02"  --iou 0.5  --imgsz 512  --device 0 (0=用 GPU, 默认 cpu)
  # 只自检(加载+数据, 不推理): --dry-run

示例:
  python3 土建/渗漏水/eval_true_pr.py \
    --weights /home/b311/data/zhangzixuan/linfengyao/2/2/yolo_runs/<run名>/weights/best.pt \
    --conf "0.25 0.10 0.05 0.02" --iou 0.5 --imgsz 512 --device 0
"""
import argparse
import os
import sys
from pathlib import Path

# ---------- 路径解析: 与 yolo训练_8x4090.py 完全一致 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "土建_渗漏水_yolo")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")


def ensure_data_yaml():
    """把 data.yaml 的 path 修正为 DATASET_DIR (服务器拷贝后指向服务器路径)。"""
    if not os.path.isdir(DATASET_DIR):
        sys.exit(
            f"\n✗ 找不到数据集目录: {DATASET_DIR}\n"
            f"  请保持脚本在 <项目根>/土建/渗漏水/ 下、数据在 <项目根>/data/土建_渗漏水_yolo。\n"
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


def parse_splits() -> dict:
    """解析 train/val/test 相对目录。"""
    splits = {}
    if os.path.isfile(DATA_YAML):
        for line in Path(DATA_YAML).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            for key in ("train", "val", "test"):
                if line.startswith(key + ":"):
                    splits[key] = line.split(":", 1)[1].strip()
    return splits


def iou_xyxy(a, b):
    """xyxy 两个框的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def parse_gt_txt(lbl_path: str, w: int, h: int) -> list:
    """读 YOLO 标签 txt, 归一化(cx,cy,w,h) → 绝对 xyxy。"""
    boxes = []
    if not os.path.isfile(lbl_path):
        return boxes
    for line in Path(lbl_path).read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            _, cx, cy, bw, bh = map(float, parts)
        except ValueError:
            continue
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append((x1, y1, x2, y2))
    return boxes


def match_tp_fp_fn(pred_boxes, gt_boxes, iou_thresh: float):
    """贪心 IoU 匹配: 返回 (tp, fp, fn)。
    pred_boxes: [(x1,y1,x2,y2,conf)], gt_boxes: [(x1,y1,x2,y2)]
    预测按 conf 降序, 每个 GT 至多匹配一个预测。"""
    gt_used = [False] * len(gt_boxes)
    tp = 0
    for pb in sorted(pred_boxes, key=lambda b: -b[4]):
        best_iou, best_gi = iou_thresh, -1
        for gi, gb in enumerate(gt_boxes):
            if gt_used[gi]:
                continue
            ov = iou_xyxy(pb[:4], gb)
            if ov > best_iou:
                best_iou, best_gi = ov, gi
        if best_gi >= 0:
            gt_used[best_gi] = True
            tp += 1
        # 未匹配上的预测 = FP (在汇总时按 tp/fp 计算)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser(
        description="真 Precision/Recall: predict + IoU 匹配 (多 conf 阈值)")
    ap.add_argument("--weights", required=True, help="要评估的 best.pt 路径(必填)")
    ap.add_argument("--conf", default="0.25 0.10 0.05 0.02",
                    help="多个 conf 阈值, 空格分隔(默认 '0.25 0.10 0.05 0.02')")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU 匹配阈值, ≥该值算 TP(默认 0.5)")
    ap.add_argument("--imgsz", type=int, default=512, help="推理尺寸(默认 512, 与训练一致)")
    ap.add_argument("--device", default="cpu", help="推理设备: cpu / 0 / 0,1(默认 cpu)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只加载权重+数据自检, 不跑推理(本地快速验证用)")
    args = ap.parse_args()

    try:
        confs = [float(x) for x in args.conf.split()]
    except ValueError:
        sys.exit(f"\n✗ --conf 解析失败: '{args.conf}'  期望空格分隔数字, 如 '0.25 0.10 0.02'\n")
    if not confs:
        sys.exit("\n✗ --conf 为空\n")

    if not os.path.isfile(args.weights):
        sys.exit(f"\n✗ 找不到权重: {args.weights}\n")
    print(f"  ✅ 权重: {args.weights}")

    ensure_data_yaml()
    splits = parse_splits()
    if "test" not in splits:
        sys.exit(f"\n✗ data.yaml 没有 test 划分(当前: {splits})\n")
    test_rel = splits["test"]                     # 如 test/images
    test_img_dir = os.path.join(DATASET_DIR, test_rel)
    test_lbl_dir = os.path.join(DATASET_DIR, test_rel.replace("images", "labels"))
    if not os.path.isdir(test_img_dir):
        sys.exit(f"\n✗ 找不到测试图片目录: {test_img_dir}\n")
    imgs = sorted(f for f in os.listdir(test_img_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not imgs:
        sys.exit(f"\n✗ {test_img_dir} 里没有图片\n")
    print(f"  测试集: {test_img_dir}  共 {len(imgs)} 张")

    # ---- 读 GT(需图片尺寸, 用 PIL 读) ----
    from PIL import Image
    gt_by_img, img_paths = {}, []
    for fn in imgs:
        ip = os.path.join(test_img_dir, fn)
        img_paths.append(ip)
        try:
            with Image.open(ip) as im:
                w, h = im.size
        except Exception as e:
            sys.exit(f"\n✗ 读取图片失败 {ip}: {e}\n")
        lbl = os.path.join(test_lbl_dir, os.path.splitext(fn)[0] + ".txt")
        gt_by_img[fn] = parse_gt_txt(lbl, w, h)
    n_gt = sum(len(v) for v in gt_by_img.values())
    print(f"  GT 框总数: {n_gt}")

    if args.dry_run:
        from ultralytics import YOLO
        YOLO(args.weights)
        print("\n✅ dry-run 通过: 权重可加载、数据自检 OK。")
        print("   (正式评估去掉 --dry-run 即可)")
        return

    # ---- 推理: 用最低 conf 出一批候选, 各阈值再过滤(避免多次推理) ----
    from ultralytics import YOLO
    min_conf = min(confs)
    print(f"\n🔬 predict 推理 (imgsz={args.imgsz}, device={args.device}, 候选 conf≥{min_conf})…")
    model = YOLO(args.weights)
    results = model.predict(
        source=img_paths,
        conf=min_conf,
        iou=0.5,                 # NMS IoU, 与匹配阈值解耦(标准做法)
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    # 每个 conf 阈值的统计
    stats = {}
    for conf in confs:
        tp = fp = fn = 0
        for fn_img, res in zip(imgs, results):
            pb = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                c = res.boxes.conf.cpu().numpy()
                for bx, cc in zip(xyxy, c):
                    if cc >= conf:
                        pb.append((float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]), float(cc)))
            t, f_p, f_n = match_tp_fp_fn(pb, gt_by_img[fn_img], args.iou)
            tp += t; fp += f_p; fn += f_n
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        stats[conf] = dict(tp=tp, fp=fp, fn=fn, p=p, r=r)

    # ---- 输出表格 ----
    print("\n" + "=" * 66)
    print(f"真 PR 对比  (IoU≥{args.iou} 算 TP, 测试集 {len(imgs)} 张 / GT {n_gt} 框, imgsz={args.imgsz})")
    print("=" * 66)
    print(f"  {'conf':>6} | {'Precision':>9} | {'Recall':>7} | {'TP':>4} {'FP':>4} {'FN':>4}")
    print("  " + "-" * 58)
    for conf in confs:
        s = stats[conf]
        print(f"  {conf:>6.3f} | {s['p']*100:>7.2f}% | {s['r']*100:>5.2f}% | "
              f"{s['tp']:>4} {s['fp']:>4} {s['fn']:>4}")
    print("=" * 66)

    # ---- 最低 conf 下仍漏检的图片 ----
    low = min(confs)
    print(f"\n📋 最低 conf={low:.3f} 下仍漏检的图片 (共 {len(imgs)} 张中漏检 GT 框):")
    missed = []
    for fn_img, res in zip(imgs, results):
        pb = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            c = res.boxes.conf.cpu().numpy()
            for bx, cc in zip(xyxy, c):
                if cc >= low:
                    pb.append((float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]), float(cc)))
        _, _, fn_n = match_tp_fp_fn(pb, gt_by_img[fn_img], args.iou)
        if fn_n > 0:
            missed.append((fn_img, fn_n))
    if missed:
        for name, n in sorted(missed, key=lambda x: -x[1]):
            print(f"  - {name}  漏检 {n} 框")
    else:
        print("  (无漏检 — 全部 GT 框都被检出)")
    print(f"\n建议: 按 R 是否≥95% 决定部署 conf。Recall 不足时看上面漏检图定位原因。\n")


if __name__ == "__main__":
    main()
