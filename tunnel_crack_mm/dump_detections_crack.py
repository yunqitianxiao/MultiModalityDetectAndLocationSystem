#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单类裂缝三模态融合检测 -> 检测框 + 车体系距离/方位 dump
================================================================
对一份配对数据(output_crack*)逐帧跑 fusion/train_assist.py 训出的
PointCloudAssistedYOLO, 解码检测框 -> 从 attrmap 的 depth 通道(相机系米)取框内
中值深度 -> pixel_to_camera -> camera_to_base -> 车体系 距离/方位, 写

    detections_<name>.json = {stem: [{cls, class_name, conf, distance,
                                      azimuth_deg, bbox, grid_pts}]}

供 run_closed_loop_crack.py(DecisionDecoder)离线闭环回放使用。纯本地, 不需要
服务器/ROS。几何与 pro/3/fusion_decision_dump.py 一致, 只是换成 inspection_robot
的单支融合模型 + 单类。

用法(用 inspection_robot 的 venv, 有 torch cu130):
  .venv/bin/python dump_detections_crack.py \
      --data ../inspection_robot/data_gen/output_crack_big \
      --ckpt ../inspection_robot/runs/assist_crack/last.pt \
      --out detections_crack.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_IR = os.environ.get(
    'IR_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inspection_robot')),
)
if _IR not in sys.path:
    sys.path.insert(0, _IR)

from fusion.config import YOLO_SIZE  # noqa: E402
from fusion.pointcloud_assist_detector import (  # noqa: E402
    PointCloudAssistedYOLO, load_sample,
)
from localization.img_px_to_world import camera_to_base, pixel_to_camera  # noqa: E402

CLASS_NAMES = ['crack']
DEFAULT_IMG_SIZE = (1280, 720)      # 仅作 calibration.json 缺 img_size 时的兜底


def letterbox_geo(h, w, new_size=YOLO_SIZE):
    """复刻 ultralytics LetterBox(new_shape=(640,640), auto=False) 的几何。
    返回 (r, dw, dh); 反变换 x_native = (x_lb - dw) / r。"""
    r = min(new_size / h, new_size / w)
    nw, nh = round(w * r), round(h * r)
    return r, (new_size - nw) / 2.0, (new_size - nh) / 2.0


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
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ai = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        ar = (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        iou = inter / (ai + ar - inter + 1e-9)
        order = rest[~(same & (iou > iou_thr))]
    return keep


def _box_loc(dmap, bbox, K, T_cam_base):
    """原生像素框 -> 框内深度(attrmap depth, 相机系米)中值 -> 车体系 距离/方位。
    取框内有效深度中值(抗杂点), 框中心像素作投影点; 无有效深度时退回框中心单像素。"""
    x1, y1, x2, y2 = bbox
    H, W = dmap.shape
    xa = int(max(0.0, min(W - 1, x1))); xb = int(max(xa + 1, min(W, x2)))
    ya = int(max(0.0, min(H - 1, y1))); yb = int(max(ya + 1, min(H, y2)))
    reg = dmap[ya:yb, xa:xb]
    valid = reg[reg > 0.01]
    if valid.size:
        depth = float(np.median(valid))
    else:
        cx = int(max(0.0, min(W - 1, x1 + (x2 - x1) / 2)))
        cy = int(max(0.0, min(H - 1, y1 + (y2 - y1) / 2)))
        depth = float(dmap[cy, cx])
    cx = x1 + (x2 - x1) / 2.0
    cy = y1 + (y2 - y1) / 2.0
    Xb = camera_to_base(pixel_to_camera(cx, cy, depth, K), T_cam_base)
    dist = float(np.linalg.norm(Xb))
    az = float(np.degrees(np.arctan2(Xb[1], Xb[0])))
    return dist, az, int(valid.size)


def _to_t(x):
    """load_sample 返回 numpy; 模型要 torch (B,...) —— 统一转 float32 张量。"""
    return torch.from_numpy(np.asarray(x, np.float32))


def list_stems(data_root):
    """五路齐全(images/labels/attrmaps/poses, clouds 可选)的帧 stem, 升序。"""
    ad = os.path.join(data_root, 'attrmaps')
    stems = []
    for f in sorted(os.listdir(ad)):
        if not f.endswith('.npz'):
            continue
        s = f[:-4]
        if all(os.path.exists(os.path.join(data_root, d, s + e)) for d, e in
               (('images', '.jpg'), ('poses', '.json'))):
            stems.append(s)
    return stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_IR, 'data_gen', 'output_crack'))
    ap.add_argument('--ckpt', required=True, help='fusion/train_assist.py 的 last.pt')
    ap.add_argument('--conf-thr', type=float, default=0.25)
    ap.add_argument('--iou-thr', type=float, default=0.45)
    ap.add_argument('--no-point', action='store_true', help='载入 RGB-only 消融权重')
    ap.add_argument('--limit', type=int, default=0, help='>0 只跑前 N 帧(探路)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'detections_crack.json'))
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    nc = int(ck.get('nc', 1))
    use_point = bool(ck.get('use_point', True)) and not args.no_point
    # 微调过的图像主干(裂缝 best.pt, nc=1)的 cv3 分类层与 COCO(nc=80)通道不同,
    # 必须用同一主干重建, 否则 image_encoder.*.cv3.* 形状不匹配(ckpt.args 里存了它)
    img_ckpt = (ck.get('args') or {}).get('img_ckpt') or None
    model = PointCloudAssistedYOLO(use_point=use_point, nc=nc, device=dev, ckpt=img_ckpt)
    model.load_state_dict(ck['model'])
    model.eval()
    torch.set_grad_enabled(False)
    print(f'[load] {args.ckpt} (nc={nc}, use_point={use_point}, device={dev})', flush=True)

    cal = json.load(open(os.path.join(args.data, 'calibration.json')))
    K = np.asarray(cal['K'], float)
    T_cam_base = np.asarray(cal['T_cam_base'], float)
    W, H = [int(v) for v in cal.get('img_size', DEFAULT_IMG_SIZE)]
    r, dw, dh = letterbox_geo(H, W)
    print(f'[geo] img {W}x{H} -> letterbox r={r} dw={dw} dh={dh}', flush=True)

    stems = list_stems(args.data)
    if args.limit > 0:
        stems = stems[:args.limit]
    print(f'[data] {len(stems)} 帧 @ {args.data}', flush=True)

    out = {}
    n_box = n_skip = n_frame_with = 0
    for k, stem in enumerate(stems):
        i = int(stem)
        tensor, attrmap, pose, _, _ = load_sample(i, root=args.data)
        preds = model(_to_t(tensor)[None].to(dev), _to_t(attrmap)[None].to(dev),
                      _to_t(pose)[None].to(dev))
        y = preds[0] if isinstance(preds, (tuple, list)) else preds
        y = y[0].float().cpu()                       # (4+nc, N)
        xywh = y[:4].t().numpy()                     # cx,cy,w,h @letterbox px
        conf_all = y[4:]
        cc, cid = conf_all.max(0)                    # (N,), (N,)
        cc = cc.numpy(); cid = cid.numpy()
        m = cc >= args.conf_thr
        if not m.any():
            out[stem] = []
            continue
        cc2, cid2, xywh2 = cc[m], cid[m], xywh[m]
        xc, yc, w, h = xywh2[:, 0], xywh2[:, 1], xywh2[:, 2], xywh2[:, 3]
        xyxy = np.stack([(xc - w / 2 - dw) / r, (yc - h / 2 - dh) / r,
                         (xc + w / 2 - dw) / r, (yc + h / 2 - dh) / r], 1)
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, W)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, H)
        keep = _nms(xyxy, cc2, cid2, args.iou_thr)

        dmap = np.load(os.path.join(args.data, 'attrmaps', f'{stem}.npz'))['depth']
        dets = []
        for j in keep:
            bbox = [float(v) for v in xyxy[j]]
            dist, az, npts = _box_loc(dmap, bbox, K, T_cam_base)
            if not math.isfinite(dist) or dist < 0.05 or dist > 200:
                n_skip += 1
                continue
            clsi = int(cid2[j])
            dets.append({'cls': clsi,
                         'class_name': CLASS_NAMES[clsi] if clsi < len(CLASS_NAMES) else str(clsi),
                         'conf': float(cc2[j]), 'distance': dist, 'azimuth_deg': az,
                         'bbox': [round(v, 1) for v in bbox], 'grid_pts': int(npts)})
            n_box += 1
        out[stem] = dets
        if dets:
            n_frame_with += 1
        if (k + 1) % 40 == 0:
            print(f'  {k+1}/{len(stems)} 帧... (框 {n_box})', flush=True)

    with open(args.out, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'[done] 有框帧 {n_frame_with}/{len(stems)}, 检测框 {n_box}, 跳过 {n_skip}\n'
          f'       写出 {args.out}', flush=True)


if __name__ == '__main__':
    main()
