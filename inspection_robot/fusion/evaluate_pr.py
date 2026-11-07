"""
检测 PR 曲线评估与阈值标定 —— 量化"整条 PR 曲线有没有往上抬"
================================================================
点云辅助检测(PointCloudAssistedYOLO)在保留验证集上按 class@IoU0.5 扫置信度阈值,
逐类算 Precision/Recall, 出 PR 曲线 + AP@0.5 + mAP@0.5, 并按"最大 F1"或"目标召回"
给出推荐工作阈值。这就是拿"整个 PR 曲线"做对照的测量棒: 同一验证集, 基线 vs
加了预训练/伪标注/门控/跟踪之后, AP 与工作点都可比。

与 train_assist 的 val 划分保持一致(seed 42 + val_ratio), 避免评估集泄漏。

用法:
    .venv/bin/python fusion/evaluate_pr.py --ckpt runs/baseline_v1/last.pt
    .venv/bin/python fusion/evaluate_pr.py --ckpt runs/baseline_v1/last.pt --target-recall 0.8
    .venv/bin/python fusion/evaluate_pr.py --ckpt ... --plot            # 存 plt PR 图
产物: {out}/pr_results.json + {out}/pr_threshold_sweep.csv (+ {out}/pr_curve.png)
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from fusion.config import NC, OUTPUT_V7  # noqa: E402
from fusion.pointcloud_assist_detector import (  # noqa: E402
    PointCloudAssistedYOLO, collate_batch, load_sample,
)

def parse_args():
    p = argparse.ArgumentParser(description='点云辅助检测 PR 评估与阈值标定')
    p.add_argument('--ckpt', required=True, help='train_assist.py 产物 {out}/last.pt')
    p.add_argument('--data', default=OUTPUT_V7)
    p.add_argument('--conf-min', type=float, default=0.001, help='NMS 最小值, 低于此丢弃(往上扫)')
    p.add_argument('--iou-thres', type=float, default=0.5, help='匹配 IoU 阈值')
    # 必须与 ultralytics val 默认 (conf=0.001, iou=0.7) 一致, 否则数字与本工程
    # 之外的评测/与 train 期间记录的 mAP 不可比。0.45 会把相邻目标框合并,
    # 在密集目标(如多裂缝帧 gt=13)下造成漏检, 使 recall 与 AP 系统性偏低
    # (实测单支在此 val 上 recall 上限 0.877@0.45 vs 0.978@0.7)。
    p.add_argument('--nms-iou', type=float, default=0.7)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--target-recall', type=float, default=0.0, help='>0 时推荐达到该召回的最大精度阈值')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--out', default='runs/pr_eval')
    return p.parse_args()


# ================================================================
# 1. 模型装载 + 确定性验证集划分 (与 train_assist 一致)
# ================================================================
def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    nc = ck.get('nc', NC)
    # 微调过的图像主干(如裂缝 best.pt, nc=1)与 COCO 主干(nc=80)的 cv3 分类层通道数不同,
    # 必须用同一主干重建, 否则 image_encoder.*.cv3.* 形状不匹配。ckpt 的 args 里存了它。
    img_ckpt = (ck.get('args') or {}).get('img_ckpt') or None
    model = PointCloudAssistedYOLO(use_point=ck.get('use_point', True), nc=nc,
                                   device=device, ckpt=img_ckpt)
    miss, unexp = model.load_state_dict(ck['model'], strict=False)
    if miss:
        print(f"⚠ model 缺键(预期仅冻结主干): {len(miss)} 个, {list(miss)[:5]}")
    model.eval()
    return model, ck


def train_val_split(n_img, val_ratio, seed):
    idx = list(range(n_img))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(n_img * val_ratio))
    return idx[:n_val], idx[n_val:]


# ================================================================
# 2. 推理: 逐帧 -> 框 (640 空间 xyxy, 已 NMS)
# ================================================================
@torch.no_grad()
def infer(model, frames, root, device, conf_min, nms_iou, target_class=None):
    from ultralytics.utils.nms import non_max_suppression
    preds = []                                     # 每帧 (M,6)=x1,y1,x2,y2,conf,cls
    for bi in range(0, len(frames), 8):
        batch = frames[bi:bi + 8]
        imgs, attrmaps, poses, _ = collate_batch(
            [load_sample(i, root=root, target_class=target_class) for i in batch])
        imgs = imgs.to(device); attrmaps = attrmaps.to(device); poses = poses.to(device)
        out = model(imgs, attrmaps, poses)         # eval -> (pred_grid, raw)
        grid = out[0] if isinstance(out, tuple) else out   # (B, 4+nc, N)
        # 本环境 torchvision 未编 CUDA nms, 移到 CPU 以稳定跑(N=8400, CPU 也快)。
        # non_max_suppression 全程只在 CPU 上做, 避免 NotRegistered CUDA 报错。
        nms = non_max_suppression(grid.detach().cpu(), conf_thres=conf_min,
                                  iou_thres=nms_iou, max_det=300)
        for d in nms:
            preds.append(d if len(d) else torch.zeros((0, 6)))
    return preds


# ================================================================
# 3. 逐类 PR: 阈值扫描, IoU 贪心匹配
# ================================================================
def match_one(pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls, iou_thres, num_cls):
    """帧级匹配 (只处理无阈值): 返回 pred 的 (match_gt_idx(N,), is_tp(N,bool)) 及 FN 计数。
    用 torchvision.ops.box_iou; 每框匹配未占用 GT 里 IoU 最大者。"""
    from torchvision.ops import box_iou
    tp = np.zeros(len(pred_boxes), dtype=bool)
    order = np.argsort(-pred_scores)               # 高分先匹配, 尽量多 TP
    used = np.zeros(len(gt_boxes), dtype=bool)
    for k in order:
        c = pred_cls[k]
        cand = np.where(gt_cls == c)[0]
        if len(cand) == 0:
            continue
        b = pred_boxes[k:k + 1]
        g = gt_boxes[cand]
        iou = box_iou(torch.from_numpy(b), torch.from_numpy(g)).numpy()[0]
        idx = int(np.argmax(iou))
        if iou[idx] >= iou_thres and not used[cand[idx]]:
            tp[k] = True
            used[cand[idx]] = True
    n_gt = len(gt_boxes)
    n_fn = int((~used).sum())
    return tp, n_fn, n_gt


def per_class_pr(frames_preds, frames_gt, num_cls, iou_thres):
    """收集每类所有检测(带分数)+ 每类 GT 数, 阈值扫描得 PR 曲线。"""
    det = {c: {'s': [], 'tp': [], 'n_gt': 0} for c in range(num_cls)}
    for preds, (gt_boxes, gt_cls) in zip(frames_preds, frames_gt):
        if len(preds):
            p_boxes = preds[:, :4].cpu().numpy()
            p_scores = preds[:, 4].cpu().numpy()
            p_cls = preds[:, 5].long().cpu().numpy()
        else:
            p_boxes = np.zeros((0, 4)); p_scores = np.zeros((0,)); p_cls = np.zeros((0,), int)
        tp, n_fn, n_gt = match_one(p_boxes, p_scores, p_cls, gt_boxes, gt_cls, iou_thres, num_cls)
        for c in range(num_cls):
            det[c]['n_gt'] += int((gt_cls == c).sum())
            det[c]['s'].extend(p_scores[p_cls == c].tolist())
            det[c]['tp'].extend(tp[p_cls == c].tolist())
    return det


def curve_for_class(d, n_bins=101):
    """按分数降序累积 TP/FP -> (thresholds, precision, recall)。"""
    if not d['s'] or d['n_gt'] == 0:
        return None
    s = np.array(d['s']); tp = np.array(d['tp'])
    order = np.argsort(-s)
    s, tp = s[order], tp[order]
    cs = np.concatenate([[0.0], np.cumsum(tp)])            # 前 i 个的 TP 累积
    cfp = np.concatenate([[0.0], np.cumsum(1 - tp)])
    recall = cs / d['n_gt']
    precision = cs / (cs + cfp).clip(min=1)
    # 阈值 = 每步的 scores (降序)
    th = np.concatenate([[s[0] + 1e-6], s])
    return th, precision[1:], recall[1:]


def ap_from_curve(recall, precision):
    """11 点插值 VOC-style AP。"""
    if len(recall) == 0:
        return 0.0
    ap = 0.0
    for t in np.arange(0.0, 1.0 + 1e-9, 0.1):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max() / 11.0
    return ap


def optimal_threshold(th, precision, recall, target_recall=0.0):
    """max-F1 或目标召回下的最佳阈值。"""
    if target_recall > 0:
        mask = recall >= target_recall
        if mask.any():
            j = np.where(mask)[0][-1]
            return th[j], precision[j], recall[j]
        return th[-1], precision[-1], recall[-1]
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    j = int(np.argmax(f1))
    return th[j], precision[j], recall[j]


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ck = load_model(args.ckpt, device)
    with open(os.path.join(args.data, 'data.yaml')) as f:
        names = next(l for l in f if 'names:' in l)
    nc = model.nc
    # 渗漏水单类: 训练时 class id 已被重映射为 0, 显示名取原数据里目标类名
    target_class = ck.get('args', {}).get('target_class')

    n_img = len([f for f in os.listdir(os.path.join(args.data, 'images')) if f.endswith('.jpg')])
    val_idx, _ = train_val_split(n_img, args.val_ratio, args.seed)
    print(f"[eval] {os.path.basename(args.ckpt)} | nc={nc} | val={len(val_idx)} 帧 | "
          f"device {device}, iou={args.iou_thres}")

    # GT per frame (640 空间 xyxy) —— 与训练同协议: target_class 过滤
    frames_gt = []
    for i in val_idx:
        _, _, _, cls, xyxy = load_sample(i, root=args.data, target_class=target_class)
        frames_gt.append((xyxy.astype(np.float32), cls.astype(np.int64)))

    t0 = time.time()
    frames_preds = infer(model, val_idx, args.data, device, args.conf_min, args.nms_iou,
                         target_class=target_class)
    print(f"[infer] 推理完成 ({time.time()-t0:.0f}s)")

    det = per_class_pr(frames_preds, frames_gt, nc, args.iou_thres)
    class_names = None
    try:
        import yaml
        class_names = yaml.safe_load(open(os.path.join(args.data, 'data.yaml')))['names']
    except Exception:
        class_names = {c: str(c) for c in range(nc)}
    if target_class is not None:            # 单类: 重映射后 0 = 原目标类
        name = class_names.get(target_class, str(target_class))
        class_names = {0: name}

    os.makedirs(args.out, exist_ok=True)
    results = {'mAP@0.5': None, 'classes': {}}
    aps = []
    rows = []
    for c in range(nc):
        cur = curve_for_class(det[c])
        if cur is None:
            results['classes'][c] = {'name': class_names.get(c, c), 'ap': None,
                                     'n_gt': det[c]['n_gt'], 'n_det': len(det[c]['s'])}
            rows.append([c, class_names.get(c, c), det[c]['n_gt'], len(det[c]['s']), 'nan', 'nan', 'nan'])
            continue
        th, prec, rec = cur
        ap = ap_from_curve(rec, prec)
        aps.append(ap)
        t_opt, p_opt, r_opt = optimal_threshold(th, prec, rec, args.target_recall)
        results['classes'][c] = {'name': class_names.get(c, c), 'ap': round(float(ap), 4),
                                 'n_gt': det[c]['n_gt'], 'n_det': len(det[c]['s']),
                                 'opt_thresh': round(float(t_opt), 4),
                                 'opt_precision': round(float(p_opt), 4),
                                 'opt_recall': round(float(r_opt), 4)}
        rows.append([c, class_names.get(c, c), det[c]['n_gt'], len(det[c]['s']),
                     round(float(ap), 4), round(float(p_opt), 4), round(float(t_opt), 4)])
        if args.plot:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.plot(rec, prec, label=f'{class_names.get(c,c)} AP={ap:.3f}')
    results['mAP@0.5'] = round(float(np.mean(aps)), 4) if aps else None

    # ---- 写 CSV ----
    with open(os.path.join(args.out, 'pr_threshold_sweep.csv'), 'w') as f:
        f.write('cls,name,gt_count,det_count,AP@0.5,opt_precision,opt_thresh\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')

    if args.plot:
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR curve @IoU0.5')
        plt.legend(loc='lower left'); plt.grid(alpha=0.3)
        plt.savefig(os.path.join(args.out, 'pr_curve.png'), dpi=120)
        plt.close()

    with open(os.path.join(args.out, 'pr_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n=== mAP@0.5: {results['mAP@0.5']} ===")
    for c, r in results['classes'].items():
        if r['ap'] is None:
            print(f"  cls {r['name']}: 无GT/无检测 (gt={r['n_gt']})")
        else:
            print(f"  cls {r['name']:24s} AP={r['ap']:.4f}  n_gt={r['n_gt']:3d}  det={r['n_det']:4d} "
                  f"  工作点(th={r['opt_thresh']:.3f}, P={r['opt_precision']:.3f}, R={r['opt_recall']:.3f})")
    print(f"\n[save] {os.path.join(args.out, 'pr_results.json')}")


if __name__ == '__main__':
    main()
