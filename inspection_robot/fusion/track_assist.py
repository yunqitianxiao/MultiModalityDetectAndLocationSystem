"""
时序跟踪 (recall 杠杆) —— 把"某帧漏检但前后帧都检到"的缺陷补回来
================================================================
单帧检测在高漏检段(低置信阈值)会丢目标: 缺陷在 t-1/t+1 帧高置信, 偏偏 t 帧
置信 < 阈、被设成漏检。时序跟踪用 IoU 关联把跨帧同一缺陷串成 track, 对漏检帧
用恒速外推**补回一个框**(置信继承 track 峰值的自信度) → 该缺陷被找回,
recall 上升。这是"整条 PR 曲线往上抬"的第二个后处理杠杆(配合 evaluate_pr 工作点)。

与 evaluate_pr 的区别: 那是对随机划分的帧逐帧独立评估, 帧间无时序价值; 本模块
对**连续帧序列**评估 raw-det vs track-added, 才能体现跨帧补漏。

做法 (轻量 ByteTrack-lite):
    - 逐帧模型推理 -> dets (xyxy, conf, cls)
    - 相邻帧 IoU 关联(同类别), 贪心匹配 IoU>=0.4, 未配对新开 track
    - 未匹配的 track: miss_age++, 用 (cx,cy) 恒速外推预测本帧框, 保留其峰值 conf;
      age 超限(默认 5)则删除
    - 逐帧输出框(raw 或 raw+propagated), 用 evaluate_pr 的逐类 PR 量化 raw vs +track

用法:
    .venv/bin/python fusion/track_assist.py --ckpt runs/v2_point80/last.pt \
        --start 0 --end 60 --out runs/track_eval
产物: {out}/track_vs_raw_pr.json (+ 可选 png)
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

from fusion.config import OUTPUT_V7  # noqa: E402
from fusion.evaluate_pr import (  # noqa: E402
    infer, load_model, match_one, per_class_pr, curve_for_class, ap_from_curve,
)
from fusion.pointcloud_assist_detector import load_sample  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='时序跟踪补漏: 量化 raw vs +track 的 PR')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', default=OUTPUT_V7)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--end', type=int, default=60)
    p.add_argument('--iou-assoc', type=float, default=0.4, help='相邻帧关联 IoU')
    p.add_argument('--max-age', type=int, default=5, help='漏检多少帧删 track')
    p.add_argument('--conf-min', type=float, default=0.001)
    p.add_argument('--iou-thres', type=float, default=0.5, help='PR 匹配 IoU')
    p.add_argument('--nms-iou', type=float, default=0.45)
    p.add_argument('--plot', action='store_true')
    p.add_argument('--out', default='runs/track_eval')
    return p.parse_args()


# ================================================================
# 轻量跟踪器
# ================================================================
class _Track:
    __slots__ = ('id', 'cls', 'conf', 'box', 'miss', 'peak_conf', 'cx', 'cy', 'w', 'h', 'vcx', 'vcy', 'active')

    def __init__(self, tid, det, conf):
        self.id = tid; self.cls = det[5]; self.conf = conf; self.miss = 0; self.peak_conf = conf
        self.box = det[:4]; self.active = True
        self.cx, self.cy = float((det[0] + det[2]) / 2), float((det[1] + det[3]) / 2)
        self.w, self.h = float(det[2] - det[0]), float(det[3] - det[1])
        self.vcx = self.vcy = 0.0

    def predict(self):
        """恒速外推 -> 本帧预测框 (xyxy, conf=峰值自信)。"""
        cx, cy = self.cx + self.vcx, self.cy + self.vcy
        x1, y1 = cx - self.w / 2, cy - self.h / 2
        x2, y2 = cx + self.w / 2, cy + self.h / 2
        return np.array([x1, y1, x2, y2]), self.peak_conf

    def update(self, det, conf):
        """匹配到新检测 -> 更新状态与速度。"""
        x1, y1, x2, y2 = det[:4]
        ncx, ncy = (x1 + x2) / 2, (y1 + y2) / 2
        self.vcx = 0.8 * self.vcx + 0.2 * (ncx - self.cx)     # 平滑速度
        self.vcy = 0.8 * self.vcy + 0.2 * (ncy - self.cy)
        self.cx, self.cy = ncx, ncy
        self.w, self.h = x2 - x1, y2 - y1
        self.box = det[:4]; self.conf = conf; self.peak_conf = max(self.peak_conf, conf)
        self.miss = 0


def _iou_pairs(boxt, boxd):
    """两个 (Nt,4) (Nd,4) 的逐对 IoU, 用 torchvision。"""
    from torchvision.ops import box_iou
    return box_iou(torch.from_numpy(boxt), torch.from_numpy(boxd)).numpy()


def track_sequence(dets, iou_assoc=0.4, max_age=5):
    """dets: list[(Mi,6)] xyxy,conf,cls。返回 list[(fi, list[(xyxy,conf,cls,tid,tracked)])],
    tracked=True 表示该框是外推补回(非本帧检测)。"""
    tracks = []; next_id = 0
    frames_out = []
    for fi, di in enumerate(dets):
        if len(di) == 0:
            d = np.zeros((0, 6), np.float32)
        else:
            d = di.numpy() if hasattr(di, 'numpy') else np.asarray(di)
        pred = d[:, :4].astype(np.float32); score = d[:, 4].astype(np.float32); cls = d[:, 5].astype(int)
        # ---- 关联: 同类别 IoU 贪心 ----
        matched_det = np.zeros(len(d), bool)
        used_trk = np.zeros(len(tracks), bool)
        # 预计算每 track 预测框
        det_frames = []
        for k, trk in enumerate(tracks):
            tb, tc = trk.predict(); det_frames.append(tb)
        if len(tracks) and len(d):
            det_arr = np.stack(det_frames)
            for tcl in np.unique(np.concatenate([cls, [t.cls for t in tracks]])):
                ti = np.where([t.cls == tcl for t in tracks])[0]
                djj = np.where(cls == tcl)[0]
                if len(ti) == 0 or len(djj) == 0:
                    continue
                iou = _iou_pairs(det_arr[ti], pred[djj])
                order = np.argsort(-iou.ravel())
                for o in order:
                    a, b = divmod(int(o), iou.shape[1])
                    if used_trk[ti[a]] or matched_det[djj[b]] or iou[a, b] < iou_assoc:
                        continue
                    used_trk[ti[a]] = True; matched_det[djj[b]] = True
                    tracks[ti[a]].update(d[djj[b]], score[djj[b]])
        # ---- 未匹配 track -> 外推补回; 超龄删 ----
        frame_boxes = []
        for k, trk in enumerate(tracks):
            if used_trk[k]:
                frame_boxes.append((trk.box, trk.conf, trk.cls, trk.id, False))
                continue
            trk.miss += 1
            if trk.miss >= max_age:
                trk.active = False
                continue
            tb, tc = trk.predict()                       # 外推补回(带峰值自信)
            frame_boxes.append((tb, tc, trk.cls, trk.id, True))
        # ---- 未匹配检测 -> 新 track ----
        for j in np.where(~matched_det)[0]:
            tracks.append(_Track(next_id, d[j], score[j])); next_id += 1
            frame_boxes.append((d[j][:4], score[j], d[j][5], tracks[-1].id, False))
        # 清除非活跃
        tracks = [t for t in tracks if t.active]
        frames_out.append(frame_boxes)
    return frames_out


def assemble(frame_boxes):
    """分帧组装成 (P_i,(M,6)) 传给 evaluate_pr 的 per_class_pr。tracked 补回框也计入。"""
    preds = []
    for fb in frame_boxes:
        rows = np.array([[x1, y1, x2, y2, c, cl] for (x1, y1, x2, y2), c, cl, _, _ in fb],
                        dtype=np.float32) if fb else np.zeros((0, 6), np.float32)
        preds.append(rows)
    return preds


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ck = load_model(args.ckpt, device)
    frames = list(range(args.start, args.end))
    print(f"[track] {os.path.basename(args.ckpt)} | 帧 {args.start}..{args.end} ({len(frames)})")

    dets = infer(model, frames, args.data, device, args.conf_min, args.nms_iou)
    frames_out = track_sequence(dets, args.iou_assoc, args.max_age)
    preds_raw = assemble([det.numpy() for det in dets])        # 原始检测
    preds_track = assemble(frames_out)                          # +track 补漏

    # GT 逐帧
    frames_gt = []
    for i in frames:
        _, _, _, cls, xyxy = load_sample(i, root=args.data)
        frames_gt.append((xyxy.astype(np.float32), cls.astype(np.int64)))

    n_ins = 0
    for p in (preds_raw, preds_track):
        n_ins += sum(len(x) for x in p)
    print(f"[track] 检测框数: raw={sum(len(p) for p in preds_raw)}, +track={sum(len(p) for p in preds_track)}")

    nc = model.nc
    res = {}
    for tag, preds in (('raw', preds_raw), ('tracked', preds_track)):
        det = per_class_pr(preds, frames_gt, nc, args.iou_thres)
        aps = []
        for c in range(nc):
            cur = curve_for_class(det[c])
            if cur is None:
                continue
            aps.append(ap_from_curve(*cur[1:]))
        res[tag] = {'mAP@0.5': round(float(np.mean(aps)), 4) if aps else None}
    print(f"\n=== {os.path.basename(args.ckpt)} 帧 {args.start}..{args.end} ===")
    print(f"  raw     mAP@0.5 = {res['raw']['mAP@0.5']}")
    print(f"  +track  mAP@0.5 = {res['tracked']['mAP@0.5']}   (recall 杠杆)")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'track_vs_raw_pr.json'), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    print(f"[save] {os.path.join(args.out, 'track_vs_raw_pr.json')}")


if __name__ == '__main__':
    main()
