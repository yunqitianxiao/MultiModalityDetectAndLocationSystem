"""
伪标注自训练 —— 无标注数据扩充, 把整条 PR 曲线往上抬 (数据杠杆)
================================================================
teacher(已训检测) 对"无标注/新采集"帧预测高置信框 -> 置信度阈值 + NMS +
坐标合理性过滤 -> 写成 YOLO 标签, 并入训练集让学生重训。这是标注未到位时
最直接抬 AP 的数据杠杆(稀有类更需要大量正例)。

对 output_v7(本就有自动标注)也支持 --check 模式: 把真值当"答案", 看 teacher
高置信预测与真值的一致性(伪标签准确率), 用来标定 conf_thresh。

用法:
    # 1) 老师给一批帧生成伪标签 (真值缺失/未知的帧用 teacher 补)
    python fusion/pseudo_label.py --teacher runs/baseline_v1/last.pt \
        --data <root> --out runs/pseudo --conf 0.5
    # 2) 质量自检: 有真值但假装没有, 看 teacher 高置信预测对到多少
    python fusion/pseudo_label.py --teacher runs/baseline_v1/last.pt \
        --data output_v7 --check --conf 0.5
产物: {out}/labels/{i:06d}.txt  (YOLO 归一化 [cls,cx,cy,w,h])
      {out}/pseudo_report.json (--check 时: 每 conf 下的伪标签精确率/召回)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from fusion.config import OUTPUT_V7, YOLO_SIZE  # noqa: E402
from fusion.evaluate_pr import infer, load_model, train_val_split  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='伪标注自训练: teacher 给无标注帧生成 YOLO 标签')
    p.add_argument('--teacher', required=True, help='已训检测 {out}/last.pt')
    p.add_argument('--data', default=OUTPUT_V7)
    p.add_argument('--out', default='runs/pseudo')
    p.add_argument('--conf', type=float, default=0.5, help='伪标签置信度阈值(teacher 预测)')
    p.add_argument('--iou-thres', type=float, default=0.5, help='伪标签质量评估的匹配 IoU')
    p.add_argument('--nms-iou', type=float, default=0.45)
    p.add_argument('--min-size', type=int, default=8, help='过滤过小框(像素, 640 空间)')
    p.add_argument('--frames', default='', help='逗号分隔帧号; 空=全部帧')
    p.add_argument('--check', action='store_true', help='质量自检(有真值但作为未知)')
    return p.parse_args()


def boxes_to_yolo(preds, conf, min_size):
    """(M,6)=x1,y1,x2,y2,conf,cls (@640) -> [(cls,cx,cy,w,h) 归一化], 过滤 conf/尺寸。"""
    if len(preds) == 0:
        return []
    xyxy = preds[:, :4]; s = preds[:, 4]; c = preds[:, 5]
    keep = (s >= conf) & ((xyxy[:, 2] - xyxy[:, 0]) >= min_size) & ((xyxy[:, 3] - xyxy[:, 1]) >= min_size)
    rows = []
    for k in np.where(keep)[0]:
        x1, y1, x2, y2 = xyxy[k]
        cx, cy, w, h = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        rows.append(f"{int(c[k])} {cx/float(YOLO_SIZE):.5f} {cy/float(YOLO_SIZE):.5f} "
                    f"{w/float(YOLO_SIZE):.5f} {h/float(YOLO_SIZE):.5f}")
    return rows


def load_gt(root, i):
    """读真值 YOLO 标签 -> (cls, cxcywh) 归一化; 无则空。"""
    path = os.path.join(root, 'labels', f'{i:06d}.txt')
    if os.path.exists(path) and os.path.getsize(path) > 0:
        a = np.loadtxt(path, dtype=np.float32).reshape(-1, 5)
        return a[:, 0], a[:, 1:]
    return np.zeros((0,)), np.zeros((0, 4))


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ck = load_model(args.teacher, device)

    n_img = len([f for f in os.listdir(os.path.join(args.data, 'images')) if f.endswith('.jpg')])
    if args.frames:
        frames = [int(x) for x in args.frames.split(',')]
    else:
        frames = list(range(n_img))

    print(f"[pseudo] teacher={os.path.basename(args.teacher)}, conf={args.conf}, "
          f"frames={len(frames)} ({'check' if args.check else 'generate'}), device {device}")

    os.makedirs(os.path.join(args.out, 'labels'), exist_ok=True)

    preds_all = infer(model, frames, args.data, device, 0.001, args.nms_iou)

    n_written = 0
    tp_sum, fp_sum, gt_sum = 0, 0, 0        # --check: 伪标签 vs 真值 的框级一致性
    for k, i in enumerate(frames):
        rows = boxes_to_yolo(preds_all[k], args.conf, args.min_size)
        if not args.check:
            with open(os.path.join(args.out, 'labels', f'{i:06d}.txt'), 'w') as f:
                f.write('\n'.join(rows) + ('\n' if rows else ''))
            n_written += len(rows)
        if args.check:
            gcls, gxywh = load_gt(args.data, i)
            if len(gcls) == 0:
                continue
            # 伪标签框 == preds_all[k] (640 像素), 真值归一化 -> 像素
            pred = preds_all[k]
            pxyxy = pred[:, :4].cpu().numpy() if pred.is_cuda else pred[:, :4].numpy()
            pcls = pred[:, 5].long().cpu().numpy() if pred.is_cuda else pred[:, 5].numpy()
            pconf = pred[:, 4].cpu().numpy() if pred.is_cuda else pred[:, 4].numpy()
            gxyxy = gxywh.copy()
            gxyxy[:, 0] = (gxywh[:, 0] - gxywh[:, 2] / 2) * YOLO_SIZE
            gxyxy[:, 1] = (gxywh[:, 1] - gxywh[:, 3] / 2) * YOLO_SIZE
            gxyxy[:, 2] = (gxywh[:, 0] + gxywh[:, 2] / 2) * YOLO_SIZE
            gxyxy[:, 3] = (gxywh[:, 1] + gxywh[:, 3] / 2) * YOLO_SIZE
            gt_sum += len(gcls)
            # 每个伪标签能否匹配到同类别真值(仅计入 conf>=args.conf 的)
            from torchvision.ops import box_iou
            for t in range(len(pred)):
                if pconf[t] < args.conf:
                    continue
                c = pcls[t]
                cand = np.where(gcls == c)[0]
                if len(cand) == 0:
                    fp_sum += 1; continue
                iou = box_iou(torch.from_numpy(pxyxy[t:t + 1]),
                              torch.from_numpy(gxyxy[cand])).numpy()[0]
                if iou.max() >= args.iou_thres:
                    tp_sum += 1
                else:
                    fp_sum += 1

    if not args.check:
        print(f"[write] {n_written} 个伪标签框 -> {os.path.join(args.out, 'labels')}/xxx.txt")
    else:
        precision = tp_sum / max(tp_sum + fp_sum, 1)
        recall = tp_sum / max(gt_sum, 1)
        report = {'conf': args.conf, 'iou_thres': args.iou_thres,
                  'pseudo_TP': tp_sum, 'pseudo_FP': fp_sum, 'GT': gt_sum,
                  'pseudo_precision': round(precision, 4),
                  'pseudo_recall': round(recall, 4)}
        with open(os.path.join(args.out, 'pseudo_report.json'), 'w') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[check] 伪标签(conf>={args.conf}) vs 真值: precision={precision:.3f} "
              f"recall对真值={recall:.3f}, TP={tp_sum} FP={fp_sum} GT={gt_sum}")
        print(f"[save] {os.path.join(args.out, 'pseudo_report.json')}")


if __name__ == '__main__':
    main()
