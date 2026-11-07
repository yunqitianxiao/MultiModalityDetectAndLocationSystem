#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真 Precision/Recall 评估 (按类别匹配, 支持多类) —— crack / leakage
=================================================================
在单类版 eval_true_pr.py 基础上改造:
  - parse_gt_txt 返回 (class_id, box) 元组
  - pred 框带 class id (res.boxes.cls)
  - 匹配时**同类才匹配** (crack 预测只能对 crack 的 GT, 避免跨类虚高 TP)
  - 每个 conf 阈值输出: 总 P/R + 逐类 P/R/F1/TP/FP/FN

用法 (服务器, PYTHONPATH 指向 yolov13 fork):
  python3 eval_true_pr_2cls.py --weights <2cls best.pt> \
    --conf "0.30 0.20 0.15 0.12 0.10 0.05 0.02" --iou 0.5 --imgsz 512 --device 0
"""
import argparse
import os
import sys
from pathlib import Path

# ---------- 路径: 2 类数据集 ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "土建_裂缝渗漏水_yolo")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")


def ensure_data_yaml():
    if not os.path.isdir(DATASET_DIR):
        sys.exit(f"\n✗ 找不到数据集目录: {DATASET_DIR}\n")
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
    splits = {}
    if os.path.isfile(DATA_YAML):
        for line in Path(DATA_YAML).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            for key in ("train", "val", "test"):
                if line.startswith(key + ":"):
                    splits[key] = line.split(":", 1)[1].strip()
    return splits


def parse_names() -> list:
    """解析 data.yaml 的 names, 返回 ['crack','leakage']。
    兼容两种格式: names: [a, b] 或 names:\\n  0: a\\n  1: b"""
    if not os.path.isfile(DATA_YAML):
        return []
    lines = Path(DATA_YAML).read_text(encoding="utf-8").splitlines()
    names = []
    in_block = False
    for ln in lines:
        s = ln.strip()
        if not in_block:
            if s == "names:":
                in_block = True
            elif s.startswith("names:"):
                rest = s[len("names:"):].strip()
                if rest.startswith("["):
                    import re
                    m = re.search(r"\[([^\]]*)\]", rest)
                    if m:
                        names = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
                    break
            continue
        # 字典块: "0: crack" / "- crack"
        if s and s[0].isdigit() and ":" in s:
            names.append(s.split(":", 1)[1].strip().strip("'\""))
        elif s.startswith("-"):
            names.append(s[1:].strip().strip("'\""))
        else:
            break
    return names


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def parse_gt_txt(lbl_path: str, w: int, h: int) -> list:
    """读 YOLO 标签 txt → [(class_id, (x1,y1,x2,y2)), ...]"""
    boxes = []
    if not os.path.isfile(lbl_path):
        return boxes
    for line in Path(lbl_path).read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls, cx, cy, bw, bh = map(float, parts)
        except ValueError:
            continue
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append((int(cls), (x1, y1, x2, y2)))
    return boxes


def match_tp_fp_fn(pred_boxes, gt_boxes, iou_thresh: float):
    """贪心 IoU 匹配 (同类内): pred=[(x1,y1,x2,y2,conf,cls)], gt=[(cls,(box))]。
    返回 (tp, fp, fn)。"""
    gt_used = [False] * len(gt_boxes)
    tp = 0
    for pb in sorted(pred_boxes, key=lambda b: -b[4]):
        pcls = pb[5]
        best_iou, best_gi = iou_thresh, -1
        for gi, (gcls, gb) in enumerate(gt_boxes):
            if gt_used[gi] or gcls != pcls:      # ★ 同类才匹配
                continue
            ov = iou_xyxy(pb[:4], gb)
            if ov > best_iou:
                best_iou, best_gi = ov, gi
        if best_gi >= 0:
            gt_used[best_gi] = True
            tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser(description="真 Precision/Recall (按类匹配)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", default="0.30 0.20 0.15 0.12 0.10 0.05 0.02")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    confs = [float(x) for x in args.conf.split()]
    if not confs:
        sys.exit("\n✗ --conf 为空\n")
    if not os.path.isfile(args.weights):
        sys.exit(f"\n✗ 找不到权重: {args.weights}\n")
    print(f"  ✅ 权重: {args.weights}")

    ensure_data_yaml()
    names = parse_names()
    n_cls = len(names)
    print(f"  类别: {names} (nc={n_cls})")
    splits = parse_splits()
    if "test" not in splits:
        sys.exit(f"\n✗ data.yaml 没有 test 划分(当前: {splits})\n")
    test_rel = splits["test"]
    test_img_dir = os.path.join(DATASET_DIR, test_rel)
    test_lbl_dir = os.path.join(DATASET_DIR, test_rel.replace("images", "labels"))
    if not os.path.isdir(test_img_dir):
        sys.exit(f"\n✗ 找不到测试图片目录: {test_img_dir}\n")
    imgs = sorted(f for f in os.listdir(test_img_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not imgs:
        sys.exit(f"\n✗ {test_img_dir} 里没有图片\n")
    print(f"  测试集: {test_img_dir}  共 {len(imgs)} 张")

    from PIL import Image
    gt_by_img, img_paths = {}, []
    gt_cnt = [0] * n_cls
    for fn in imgs:
        ip = os.path.join(test_img_dir, fn)
        img_paths.append(ip)
        with Image.open(ip) as im:
            w, h = im.size
        lbl = os.path.join(test_lbl_dir, os.path.splitext(fn)[0] + ".txt")
        boxes = parse_gt_txt(lbl, w, h)
        gt_by_img[fn] = boxes
        for cls, _ in boxes:
            if cls < n_cls:
                gt_cnt[cls] += 1
    print(f"  GT 框总数: {sum(gt_cnt)}  {dict(zip(names, gt_cnt))}")

    if args.dry_run:
        from ultralytics import YOLO
        YOLO(args.weights)
        print("\n✅ dry-run 通过\n")
        return

    from ultralytics import YOLO
    min_conf = min(confs)
    print(f"\n🔬 predict 推理 (imgsz={args.imgsz}, device={args.device}, 候选 conf≥{min_conf})…")
    model = YOLO(args.weights)
    results = model.predict(
        source=img_paths, conf=min_conf, iou=0.5,
        imgsz=args.imgsz, device=args.device, verbose=False,
    )

    # 每个 conf: 总 stats + 逐类 stats
    stats = {}
    for conf in confs:
        tot = [0, 0, 0]                     # tp, fp, fn
        per = [[0, 0, 0] for _ in range(n_cls)]
        for fn_img, res in zip(imgs, results):
            pb = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                cc = res.boxes.conf.cpu().numpy()
                cl = res.boxes.cls.cpu().numpy()
                for bx, c_, cid in zip(xyxy, cc, cl):
                    if c_ >= conf:
                        pb.append((float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]),
                                   float(c_), int(cid)))
            t, f_p, f_n = match_tp_fp_fn(pb, gt_by_img[fn_img], args.iou)
            tot[0] += t; tot[1] += f_p; tot[2] += f_n
            # 逐类统计: tp/fp/fn 分摊到预测/GT 所属类
            for cid in range(n_cls):
                pp = [x for x in pb if x[5] == cid]
                gg = [g for g in gt_by_img[fn_img] if g[0] == cid]
                if pp or gg:
                    a, b, c = match_tp_fp_fn(pp, gg, args.iou)
                    per[cid][0] += a; per[cid][1] += b; per[cid][2] += c
        stats[conf] = (tot, per)

    # ---- 输出 ----
    print("\n" + "=" * 78)
    print(f"真 PR (同类匹配, IoU≥{args.iou}, 测试集 {len(imgs)} 张, imgsz={args.imgsz})")
    print("=" * 78)
    hdr = f"  {'conf':>6} | {'P':>7} | {'R':>7} | {'TP':>4} {'FP':>4} {'FN':>4}"
    print(hdr)
    print("  " + "-" * 40)
    for conf in confs:
        (tp, fp, fn) = stats[conf][0]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  {conf:>6.3f} | {p*100:>6.2f}% | {r*100:>6.2f}% | {tp:>4} {fp:>4} {fn:>4}")
    print("=" * 78)

    for ci, cname in enumerate(names):
        print(f"\n  —— {cname} (类{ci}) ——")
        print(f"  {'conf':>6} | {'P':>7} | {'R':>7} | {'F1':>6} | {'TP':>4} {'FP':>4} {'FN':>4}")
        print("  " + "-" * 46)
        for conf in confs:
            tp, fp, fn = stats[conf][1][ci]
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            print(f"  {conf:>6.3f} | {p*100:>6.2f}% | {r*100:>6.2f}% | {f1*100:>5.2f}% | "
                  f"{tp:>4} {fp:>4} {fn:>4}")
    print("\n" + "=" * 78)

    # ---- 最低 conf 下仍漏检的图片 (逐类) ----
    low = min(confs)
    print(f"\n📋 最低 conf={low:.3f} 下仍漏检 (按类):")
    for ci, cname in enumerate(names):
        missed = []
        for fn_img, res in zip(imgs, results):
            pb = []
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                cc = res.boxes.conf.cpu().numpy()
                cl = res.boxes.cls.cpu().numpy()
                for bx, c_, cid in zip(xyxy, cc, cl):
                    if c_ >= low:
                        pb.append((float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]),
                                   float(c_), int(cid)))
            pp = [x for x in pb if x[5] == ci]
            gg = [g for g in gt_by_img[fn_img] if g[0] == ci]
            _, _, fn_n = match_tp_fp_fn(pp, gg, args.iou)
            if fn_n > 0:
                missed.append((fn_img, fn_n))
        if missed:
            print(f"  {cname}: {len(missed)} 张漏检")
            for name, n in sorted(missed, key=lambda x: -x[1])[:10]:
                print(f"    - {name}  漏检 {n} 框")
        else:
            print(f"  {cname}: (无漏检)")
    print()


if __name__ == "__main__":
    main()
