# -*- coding: utf-8 -*-
"""
三分支 · RGB 单模态基线召回评测器 (与 eval_fusion_recall 同口径)
================================================================
在 output_big_s3 / output_rail_s1 (融合 test_set) 上跑【分支检测器 best_<支>.pt (RGB-only)】,
对齐 GT 算逐类 Precision/Recall/mAP50 —— 这是"融合相对基线"的对照脚。

与 eval_fusion_recall.py 完全同口径:
  - 同样的 frame 枚举 (T.resolve_label_paths) / 同一 GT 过滤+remap (sim_cls)
  - 同样的 letterbox->native 解码 (E.unletterbox) / 同样 _nms(IoU_thr) / 同样 P/R/F1/mAP50 + conf 扫描
唯一差别: 前向是 分支检测器 RGB-only (YOLO predict), 而非 融合 wrapper(image+attrmap+nav)。

【为什么类直接可比】 best_<支>.pt 输出的是分支局部类 (土建 0/1/2 = 裂纹/渗漏/剥落),
而 load_sim_frame 的 GT 已按 sim_cls remap 到同一套分支局部 0/1/2 —— 无二次映射, 天然对齐。

用法(本地, inspection_robot/.venv = torch cu130 8GB 4060):
  python -u baseline_fusion_recall.py --branch 土建 --ckpt weights/best_土建.pt \
      --data pcdata/output_big_s3 --imgsz 1280 --conf 0.25 --out runs/evalbase_土建.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from ultralytics import YOLO

_P = os.path.dirname(os.path.abspath(__file__))
if _P not in sys.path:
    sys.path.insert(0, _P)
_IR = os.environ.get(
    'IR_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inspection_robot')),
)
if _IR not in sys.path:
    sys.path.append(_IR)          # 追加, 不抢占 pro/3 的 `fusion` 模块
from fix_nms import patch_torchvision_nms   # 本机 torchvision 纯CPU缺CUDA nms, predict 需patch
patch_torchvision_nms()                     # 必须在 ultralytics 走 NMS 前调用
import train_fusion as T          # noqa: E402
import eval_fusion_recall as E    # noqa: E402  复用它 iou/ap_from_pr/conf_sweep

torch.set_grad_enabled(False)

_NAMES = {'土建': ['裂纹', '渗漏', '剥落'], '轨道': ['断裂', '丢失', '松动歪斜'],
          '设备': ['异物入侵', '管线支架松脱']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--branch', default='土建')
    ap.add_argument('--ckpt', required=True)          # best_<支>.pt
    ap.add_argument('--data', default=os.path.join(_P, 'pcdata', 'output_big_s3'))
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.25)      # 工作点
    ap.add_argument('--iou-thr', type=float, default=0.45)   # NMS IoU, 与 fusion 一致
    ap.add_argument('--match-iou', type=float, default=0.5)  # 匹配 IoU, 与 fusion 一致
    ap.add_argument('--sweep', default='0.05,0.10,0.15,0.20,0.25,0.30,0.40')
    ap.add_argument('--limit', type=int, default=0, help='>0 时只取前 N 帧(探路/VRAM 用)')
    ap.add_argument('--img-scale', type=float, default=255.0,
                    help='喂给分支检测器的像素上限: 单模态 YOLO 原生=255([0,255]); 融合训练用 [0,1]')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    conf_b = args.branch
    br = T._BRANCHES[conf_b]
    nc = br['nc']
    names = _NAMES[conf_b]
    print(f'[base] 分支={conf_b} nc={nc} 类名={names} imgsz={args.imgsz} '
          f'ckpt={args.ckpt}', flush=True)

    model = YOLO(args.ckpt)
    model.eval()
    sim_cls = br['sim_cls']
    stems = T.resolve_label_paths(args.data)
    if args.limit > 0:
        stems = stems[:args.limit]
    print(f'[data] {len(stems)} 帧 @ {args.data}', flush=True)

    gt_boxes = {k: [] for k in range(nc)}
    det_records = []                                  # (cls, conf, xyxy@native)

    for k, stem in enumerate(stems):
        s = T.load_sim_frame(args.data, stem, args.imgsz, sim_cls, nc)
        if s is None:
            continue
        _, _, _, gt_cls, gt_xyxy_imgsz = s
        dmap = np.load(os.path.join(args.data, 'attrmaps', f'{stem}.npz'))['depth']
        g = T.build_letterbox(dmap.shape, args.imgsz)
        if gt_xyxy_imgsz is not None and len(gt_xyxy_imgsz):
            gt_nat = E.unletterbox(np.asarray(gt_xyxy_imgsz, float), g)
            for ci, b in zip(np.asarray(gt_cls, int), gt_nat):
                gt_boxes[int(ci)].append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        # ---- 分支检测器 RGB-only 前向(ultralytics predict: letterbox+decode+NMS -> 原生框) ----
        ip = T.img_path_for(args.data, stem)
        res = model.predict(ip, imgsz=args.imgsz, conf=0.001, iou=args.iou_thr,
                            max_det=30000, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        cid = res.boxes.cls.detach().cpu().numpy().astype(int)
        cc = res.boxes.conf.detach().cpu().numpy()
        xyxy = res.boxes.xyxy.detach().cpu().numpy()   # 原生 1280x720 px
        for i in range(len(cc)):
            det_records.append((int(cid[i]), float(cc[i]),
                                [float(xyxy[i, 0]), float(xyxy[i, 1]),
                                 float(xyxy[i, 2]), float(xyxy[i, 3])]))
        if (k + 1) % 40 == 0:
            print(f'  {k+1}/{len(stems)} 帧... det={len(det_records)}', flush=True)

    # ---------------- 逐类指标 (@conf 工作点) + mAP50 ----------------
    print('\n===== 基线逐类指标 (IoU≥%.2f, conf≥%.2f) =====' % (args.match_iou, args.conf))
    header = f"  {'类别':<10}{'GT':>6}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>7}{'R':>7}{'F1':>7}{'mAP50':>7}"
    print(header)
    summary = {'branch': conf_b, 'kind': 'baseline', 'imgsz': args.imgsz, 'conf': args.conf,
               'classes': {}, 'total': {'gt': 0, 'tp': 0, 'fp': 0, 'fn': 0}}
    ogt = otp = ofp = ofn = 0
    for ci in range(nc):
        gts = gt_boxes[ci]
        all_dets = sorted([(c, b) for (cl, c, b) in det_records if cl == ci],
                          key=lambda x: -x[0])
        op_dets = [(c, b) for (c, b) in all_dets if c >= args.conf]
        matched_gt = [False] * len(gts)
        tp = 0
        for conf, b in op_dets:
            best_j, best_i = -1, args.match_iou
            for j, g in enumerate(gts):
                if matched_gt[j]:
                    continue
                v = E.iou(b, g)
                if v >= best_i:
                    best_i, best_j = v, j
            if best_j >= 0:
                matched_gt[best_j] = True
                tp += 1
        fp = len(op_dets) - tp
        fn = len(gts) - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rec_pts, prec_pts, running_tp, total_gt = [], [], 0, len(gts)
        matched_ap = [False] * total_gt
        for rank, (conf, b) in enumerate(all_dets, start=1):
            best_j, best_i = -1, args.match_iou
            for j, g in enumerate(gts):
                if matched_ap[j]:
                    continue
                v = E.iou(b, g)
                if v >= best_i:
                    best_i, best_j = v, j
            if best_j >= 0:
                matched_ap[best_j] = True
                running_tp += 1
            rec_pts.append(running_tp / total_gt if total_gt else 0.0)
            prec_pts.append(running_tp / rank if rank else 0.0)
        ap50 = E.ap_from_pr(rec_pts, prec_pts) if total_gt else 0.0
        ogt += len(gts); otp += tp; ofp += fp; ofn += fn
        summary['classes'][names[ci]] = {
            'gt': len(gts), 'tp': tp, 'fp': fp, 'fn': fn,
            'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4), 'mAP50': round(ap50, 4),
            'shortfall': fn}
        print(f"  {names[ci]:<10}{len(gts):>6}{tp:>6}{fp:>6}{fn:>6}{p:>7.3f}{r:>7.3f}"
              f"{f1:>7.3f}{ap50:>7.3f}")

    P = otp / (otp + ofp) if otp else 0.0
    R = otp / (otp + ofn) if otp else 0.0
    summary['total'] = {'gt': ogt, 'tp': otp, 'fp': ofp, 'fn': ofn,
                        'p': round(P, 4), 'r': round(R, 4)}
    print(f"\n  {'整体':<10}{ogt:>6}{otp:>6}{ofp:>6}{ofn:>6}{P:>7.3f}{R:>7.3f}"
          f"  <- 召回短差={ofn/(ofn+otp):.3f}({ofn} 漏检/{ogt} GT)")

    if args.sweep:
        confs = [float(x) for x in args.sweep.split(',') if x.strip()]
        rows = E.conf_sweep(det_records, gt_boxes, nc, names, args.match_iou, confs)
        summary['conf_sweep'] = rows
        print('\n===== conf 扫描: P/R/F1 (IoU≥%.2f) =====' % args.match_iou)
        print(f"  {'conf':>5} | {'GT':>5}{'TP':>6}{'FP':>6}{'FN':>6} | {'P':>6}{'R':>6}{'F1':>6} | 短差")
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
