# -*- coding: utf-8 -*-
"""
三模态融合 · 病集召回评测器
================================================================
在 output_big_s3(病集留测) 上跑 fusion_<支>_* 权重, 对齐 GT 算逐类 Precision/Recall/mAP50。
这是 A/B 共用的"尺子": 回答"召回差多少" —— R=TP/(TP+FN), 短差 = 1-R 或 FN 框数。

与分支检测器(单模态 RGB)不同, 这是 3 模态(image+attrmap+nav)。用【与训练一致的 imgsz】
建 Detector 头(否则 640/1280 错配压低召回, 见 pro3-scale-mismatch-fix)。

指标口径:
  conf 层: 按 --conf 工作点给 P/R/F1 + GT/TP/FP/FN(逐类)。
  mAP50 : 随 conf 阈值扫 PR 曲线(每类 AP, IoU>=0.5)。

用法(服务器, venv):
  python -u eval_fusion_recall.py --branch 土建 --ckpt runs/fusion_ddp/fusion_土建_1280_neg.pt \
      --data pcdata/output_big_s3 --imgsz 1280 --conf 0.25 --out runs/evalfusion_土建.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_P = os.path.dirname(os.path.abspath(__file__))
if _P not in sys.path:
    sys.path.insert(0, _P)
import train_fusion as T  # noqa: E402

torch.set_grad_enabled(False)


def _nms(xyxy, conf, cls, iou_thr=0.45):
    """按 conf 降序, 抑制同 class 且 IoU>thr 的框。返回保留索引。"""
    order = np.argsort(-conf)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        same = cls[rest] == cls[i]
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0]); yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2]); yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ai = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        ar = (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        iou = inter / (ai + ar - inter + 1e-9)
        order = rest[~(same & (iou > iou_thr))]
    return keep


def unletterbox(xyxy_imgsz, g):
    """letterbox@imgsz 的 xyxy -> 原生像素。g=(r,nw,nh,dw,dh,imgsz)。"""
    r, nw, nh, dw, dh, imgsz = g
    x = (xyxy_imgsz[:, 0] - dw) / r
    y = (xyxy_imgsz[:, 1] - dh) / r
    x2 = (xyxy_imgsz[:, 2] - dw) / r
    y2 = (xyxy_imgsz[:, 3] - dh) / r
    return np.stack([x, y, x2, y2], 1)


def iou(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def _op_metrics(gts, dets, match_iou):
    """@{conf 过滤后}的 dets -> (tp,fp,fn)。一对一贪心。"""
    matched = [False] * len(gts)
    tp = 0
    for conf, b in dets:
        best_j, best_i = -1, match_iou
        for j, g in enumerate(gts):
            if matched[j]:
                continue
            v = iou(b, g)
            if v >= best_i:
                best_i, best_j = v, j
        if best_j >= 0:
            matched[best_j] = True
            tp += 1
    return tp, len(dets) - tp, len(gts) - tp


def conf_sweep(det_records, gt_boxes, nc, names, match_iou, confs):
    """给定已收集 det/GT, 在多个 conf 阈值下算 逐类+整体 P/R/F1。返回 rows 列表。"""
    rows = []
    for conf in confs:
        per = {}
        ogt = otp = ofp = ofn = 0
        for ci in range(nc):
            gts = gt_boxes[ci]
            dets = [(c, b) for (cl, c, b) in det_records if cl == ci and c >= conf]
            dets.sort(key=lambda x: -x[0])
            tp, fp, fn = _op_metrics(gts, dets, match_iou)
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
            per[names[ci]] = {'tp': tp, 'fp': fp, 'fn': fn, 'p': round(p, 4),
                              'r': round(r, 4), 'f1': round(f1, 4)}
            ogt += len(gts); otp += tp; ofp += fp; ofn += fn
        P = otp / (otp + ofp) if otp else 0.0
        R = otp / (otp + ofn) if otp else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        rows.append({'conf': conf, 'classes': per,
                     'overall': {'gt': ogt, 'tp': otp, 'fp': ofp, 'fn': ofn,
                                 'p': round(P, 4), 'r': round(R, 4), 'f1': round(F1, 4),
                                 'shortfall': ofn}})
    return rows


def ap_from_pr(rec, prec):
    """Voc2007 AP。rec/prec 已按 conf 降序(rec 单调不减), 直接累加包络, 勿重排。
    重排(按 rec 降序)会把曲线反向导致 AP≈0。"""
    rec = np.asarray(rec, float); prec = np.asarray(prec, float)
    if len(prec) == 0:
        return 0.0
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branch', default='土建')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default=os.path.join(_P, 'pcdata', 'output_big_s3'))
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou-thr', type=float, default=0.45)
    ap.add_argument('--match-iou', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--sweep', default='0.05,0.10,0.15,0.20,0.25,0.30',
                    help='conf 扫描阈值(逗号分隔)复算 P/R/F1; 传空串跳过')
    ap.add_argument('--cond', default='full', choices=['full', 'ablate'],
                    help="full=真实 attrmap+nav; ablate=清零 attrmap+nav(图像单模态对照, 同 arch 共享 backbone/Detect)")
    ap.add_argument('--limit', type=int, default=0, help='>0 时只取前 N 帧(探路/VRAM)')
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    conf_b = _BRANCH = args.branch
    br = T._BRANCHES[conf_b]
    nc = br['nc']
    class_names = list(br['sim_cls'].keys())      # 仅用于命名? 下面用逆映射
    # 类名: sim 类 id -> (分支局部idx, 名)。抓取分支层名称(若无则用 sim id)
    # 用 DEFAULTS 对齐过命名; 这里按 sim 到局部映射表 + 手写名
    names = {'土建': ['裂纹', '渗漏', '剥落'], '轨道': ['断裂', '丢失', '松动歪斜'],
             '设备': ['异物入侵', '管线支架松脱']}[conf_b]
    print(f'[eval] 分支={conf_b} nc={nc} 类名={names} imgsz={args.imgsz}', flush=True)

    wrapper, _ = T.build_model(conf_b, img_imgsz=args.imgsz, device='cuda')
    wrapper.eval()
    wrapper.load_state_dict(torch.load(args.ckpt, map_location='cuda', weights_only=False))
    print('[load] ckpt 载入:', args.ckpt, flush=True)

    sim_cls = br['sim_cls']
    stems = T.resolve_label_paths(args.data)
    if args.limit > 0:
        stems = stems[:args.limit]
    print(f'[data] {len(stems)} 帧 @ {args.data}  cond={args.cond}', flush=True)

    # 逐类收集: gt 框 & conf 排序后的 det 候选取出最TOP的每类框
    gt_boxes = {k: [] for k in range(nc)}         # class -> list[xyxy]
    det_records = []                              # (cls, conf, xyxy)

    for k, stem in enumerate(stems):
        s = T.load_sim_frame(args.data, stem, args.imgsz, sim_cls, nc)
        if s is None:
            continue
        img, attr, nav = s[0], s[1], s[2]
        gt_cls, gt_xyxy_imgsz = s[3], s[4]        # GT 已过滤+remap, @imgsz
        dmap = np.load(os.path.join(args.data, 'attrmaps', f'{stem}.npz'))['depth']
        g = T.build_letterbox(dmap.shape, args.imgsz)   # 原生几何
        if gt_xyxy_imgsz is not None and len(gt_xyxy_imgsz):
            gt_nat = unletterbox(np.asarray(gt_xyxy_imgsz, float), g)
            for ci, b in zip(np.asarray(gt_cls, int), gt_nat):
                gt_boxes[int(ci)].append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        if args.cond == 'ablate':
            a_in = torch.zeros(1, 4, args.imgsz, args.imgsz, device='cuda')
            n_in = torch.zeros(1, 5, device='cuda')
        else:
            a_in = attr.unsqueeze(0).cuda()
            n_in = torch.from_numpy(np.asarray(nav, np.float32)).unsqueeze(0).cuda()
        y, _ = wrapper(img.unsqueeze(0).cuda(), a_in, n_in)
        y = y[0].cpu()
        xywh = y[:4].t(); cs = y[4:]
        cc, cid = cs.max(0)
        m = cc >= 0.001                              # 收集全置信度(供 mAP PR 曲线)
        if not m.any():
            continue
        cc2, cid2 = cc[m].numpy(), cid[m].long().numpy()
        xywh2 = xywh[m].numpy()
        xc, yc, w, h = xywh2[:, 0], xywh2[:, 1], xywh2[:, 2], xywh2[:, 3]
        xyxy_imgsz = np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1)
        xyxy_nat = unletterbox(xyxy_imgsz, g)
        keep = _nms(xyxy_nat, cc2, cid2, args.iou_thr)
        for i in keep:
            det_records.append((int(cid2[i]), float(cc2[i]),
                                [float(xyxy_nat[i, 0]), float(xyxy_nat[i, 1]),
                                 float(xyxy_nat[i, 2]), float(xyxy_nat[i, 3])]))
        if (k + 1) % 40 == 0:
            print(f'  {k+1}/{len(stems)} 帧... det={len(det_records)}', flush=True)

    # ---------------- 逐类指标(@conf 工作点) ----------------
    print('\n===== 逐类指标 (IoU≥%.2f, conf≥%.2f) =====' % (args.match_iou, args.conf))
    header = f"  {'类别':<10}{'GT':>6}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>7}{'R':>7}{'F1':>7}{'mAP50':>7}"
    print(header)
    summary = {'branch': conf_b, 'imgsz': args.imgsz, 'conf': args.conf, 'cond': args.cond,
               'test_set': os.path.basename(os.path.normpath(args.data)),
               'classes': {}, 'total': {'gt': 0, 'tp': 0, 'fp': 0, 'fn': 0}}
    overall_gt = overall_tp = overall_fp = overall_fn = 0
    for ci in range(nc):
        gts = gt_boxes[ci]
        all_dets = sorted([(c, b) for (cl, c, b) in det_records if cl == ci],
                          key=lambda x: -x[0])                    # conf 降序
        op_dets = [(c, b) for (c, b) in all_dets if c >= args.conf]
        dets = op_dets
        matched_gt = [False] * len(gts)
        tp = 0
        for conf, b in dets:
            best_j, best_i = -1, args.match_iou
            for j, g in enumerate(gts):
                if matched_gt[j]:
                    continue
                v = iou(b, g)
                if v >= best_i:
                    best_i, best_j = v, j
            if best_j >= 0:
                matched_gt[best_j] = True
                tp += 1
        fp = len(dets) - tp
        fn = len(gts) - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        # mAP50: 用全部 det(含低置信) 扫 PR 曲线(conf 降序); 一对一贪心匹配
        rec_pts, prec_pts = [], []
        total_gt = len(gts)
        running_tp = 0
        matched_ap = [False] * total_gt
        for rank, (conf, b) in enumerate(all_dets, start=1):
            best_j, best_i = -1, args.match_iou
            for j, g in enumerate(gts):
                if matched_ap[j]:
                    continue
                v = iou(b, g)
                if v >= best_i:
                    best_i, best_j = v, j
            if best_j >= 0:
                matched_ap[best_j] = True
                running_tp += 1
            rec_pts.append(running_tp / total_gt if total_gt else 0.0)
            prec_pts.append(running_tp / rank if rank else 0.0)
        ap50 = ap_from_pr(rec_pts, prec_pts) if total_gt else 0.0
        overall_gt += len(gts); overall_tp += tp; overall_fp += fp; overall_fn += fn
        summary['classes'][names[ci]] = {
            'gt': len(gts), 'tp': tp, 'fp': fp, 'fn': fn,
            'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4), 'mAP50': round(ap50, 4),
            'shortfall': fn,   # 漏检框数
        }
        print(f"  {names[ci]:<10}{len(gts):>6}{tp:>6}{fp:>6}{fn:>6}{p:>7.3f}{r:>7.3f}"
              f"{f1:>7.3f}{ap50:>7.3f}")

    P = overall_tp / (overall_tp + overall_fp) if overall_tp else 0.0
    R = overall_tp / (overall_tp + overall_fn) if overall_tp else 0.0
    summary['total'] = {'gt': overall_gt, 'tp': overall_tp, 'fp': overall_fp, 'fn': overall_fn,
                        'p': round(P, 4), 'r': round(R, 4)}
    print(f"\n  {'整体':<10}{overall_gt:>6}{overall_tp:>6}{overall_fp:>6}{overall_fn:>6}"
          f"{P:>7.3f}{R:>7.3f}  <- 召回短差={overall_fn/(overall_fn+overall_tp):.3f}"
          f"({overall_fn} 漏检/{overall_gt} GT)")
    # ---------------- conf 扫描 (复算, 不重跑模型) ----------------
    if args.sweep:
        confs = [float(x) for x in args.sweep.split(',') if x.strip()]
        rows = conf_sweep(det_records, gt_boxes, nc, names, args.match_iou, confs)
        summary['conf_sweep'] = rows
        print('\n===== conf 扫描: P/R/F1 (IoU≥%.2f) =====' % args.match_iou)
        print(f"  {'conf':>5} | {'GT':>5}{'TP':>6}{'FP':>6}{'FN':>6} | {'P':>6}{'R':>6}{'F1':>6} | 短差(漏检)")
        for r in rows:
            o = r['overall']
            print(f"  {r['conf']:>5.2f} | {o['gt']:>5}{o['tp']:>6}{o['fp']:>6}{o['fn']:>6} | "
                  f"{o['p']:>6.3f}{o['r']:>6.3f}{o['f1']:>6.3f} | {o['shortfall']}")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print('[存入]', args.out)


if __name__ == '__main__':
    main()
