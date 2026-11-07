# -*- coding: utf-8 -*-
"""
融合检测 -> 检测框 + 车体系距离/方位 dump（供本地路径决策离线验证）
================================================================
服务器端（yolo13l_env 已有 GPU + weights + pcdata）：
  对 output_big_s3（土建 test_set）逐帧跑 fusion_土建_1280_neg.pt，
  解码检测框 -> 从 attrmap 的 depth 通道(相机系米)采样框内中值深度 ->
  pixel_to_camera -> camera_to_base -> 车体系 距离/方位 -> 写 detections.json
  （每帧 {cls, class_name, conf, distance, azimuth_deg, bbox, grid_pts}）。

无需点云: 标定取自 output_rail_s1（同一隧道/同一传感器, 有 calibration.json）。

本地只拿 detections.json 喂 DecisionDecoder（不依赖 GPU/ultralytics fork）。
依赖纯 numpy 的 img_px_to_world.py（已 scp 到 /root/autodl-tmp/pro3/localization/）。

用法: .venv(服务器) python fusion_decision_dump.py --data pcdata/output_big_s3
     --calib pcdata/output_rail_s1 --ckpt runs/fusion_ddp/fusion_土建_1280_neg.pt --out detections_big_s3.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_P = os.path.dirname(os.path.abspath(__file__))
if _P not in sys.path:
    sys.path.insert(0, _P)
import train_fusion as T  # noqa: E402

_LOC = os.path.join(_P, 'localization')
if _LOC not in sys.path:
    sys.path.insert(0, _LOC)
from img_px_to_world import camera_to_base, pixel_to_camera  # noqa: E402  纯 numpy

BR = '土建'
CLASS_NAMES = ['裂纹', '渗漏', '剥落']
IMGSZ = 640


def _box_loc(dmap, bbox, K, T_cam_base):
    """原生像素框 -> 框内深度(attrmap depth, 米) -> 车体系 距离/方位。

    取框内有效深度(depth>0.01)的中值作目标深度(抗杂点), 框中心像素作投影点;
    无有效深度时退回框中心单像素深度。返回 (distance_m, azimuth_deg, npts)。
    """
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
    Xc = pixel_to_camera(cx, cy, depth, K)
    Xb = camera_to_base(Xc, T_cam_base)
    dist = float(np.linalg.norm(Xb))
    az = float(np.degrees(np.arctan2(Xb[1], Xb[0])))
    return dist, az, int(valid.size)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_P, 'pcdata', 'output_big_s3'))
    ap.add_argument('--calib', default=os.path.join(_P, 'pcdata', 'output_rail_s1'),
                    help='含 calibration.json 的目录(同传感器; output_big_s3 无标定文件)')
    ap.add_argument('--ckpt',
                    default=os.path.join(_P, 'runs', 'fusion_ddp', 'fusion_土建_1280_neg.pt'))
    ap.add_argument('--conf-thr', type=float, default=0.25)
    ap.add_argument('--iou-thr', type=float, default=0.45)
    ap.add_argument('--out', default=os.path.join(_P, 'detections_big_s3.json'))
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    torch.set_grad_enabled(False)
    wrapper, nc = T.build_model(BR, img_imgsz=IMGSZ, device='cuda')
    wrapper.eval()                                   # __init__ 里 det.train() 过, 必须切 eval
    # ckpt 是 wrapper.state_dict(), 必须载入否则 fusion/头是随机初始
    wrapper.load_state_dict(torch.load(args.ckpt, map_location='cuda', weights_only=False))
    print(f'[load] ckpt {args.ckpt}', flush=True)

    cal = json.load(open(os.path.join(args.calib, 'calibration.json')))
    K = np.asarray(cal['K'], float)
    T_cam_base = np.asarray(cal['T_cam_base'], float)
    img_size = tuple(int(v) for v in cal.get('img_size', [1280, 720]))
    W, H = img_size

    sim_cls = T._BRANCHES[BR]['sim_cls']
    stems = T.resolve_label_paths(args.data)
    print(f'[data] {len(stems)} 帧 @ {args.data}', flush=True)

    out = {}
    n_ok = n_skip = n_box = 0
    for k, stem in enumerate(stems):
        s = T.load_sim_frame(args.data, stem, IMGSZ, sim_cls, nc)
        if s is None:
            continue
        img, attr, nav = s[0], s[1], s[2]
        # attrmap 深度通道(720×1280, 相机系米) -> 原生几何, 与图像共用同一 letterbox
        dmap = np.load(os.path.join(args.data, 'attrmaps', f'{stem}.npz'))['depth']
        # 用【原生 H,W】建 letterbox 几何(不是 letterbox 后的 img.shape —— 那会是 no-op)
        g = T.build_letterbox(dmap.shape, IMGSZ)     # dmap.shape=(H,W)=(720,1280)
        r, nw, nh, dw, dh, imgsz = g
        nav = torch.from_numpy(np.asarray(nav, np.float32))
        y, _ = wrapper(img.unsqueeze(0).cuda(), attr.unsqueeze(0).cuda(),
                       nav.unsqueeze(0).cuda())
        y = y[0].cpu()                                # (4+nc, N)
        xywh = y[:4].t()                              # (N,4) cx,cy,w,h @letterbox px
        cs = y[4:]                                    # (nc,N) sigmoid
        cc, cid = cs.max(0)                           # (N,) conf, (N,) cls

        m = cc >= args.conf_thr
        if not m.any():
            out[stem] = []
            continue
        cc2, cid2 = cc[m], cid[m].long()
        xywh2 = xywh[m]
        # 解码到原生 1280x720 像素
        xc, yc, w, h = xywh2[:, 0], xywh2[:, 1], xywh2[:, 2], xywh2[:, 3]
        x1 = (xc - w / 2 - dw) / r
        y1 = (yc - h / 2 - dh) / r
        x2 = (xc + w / 2 - dw) / r
        y2 = (yc + h / 2 - dh) / r
        xyxy = np.stack([x1, y1, x2, y2], 1)
        keep = _nms(xyxy, cc2.numpy(), cid2.numpy(), args.iou_thr)

        dets = []
        for i in keep:
            bbox = [float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3])]
            conf = float(cc2[i]); clsi = int(cid2[i])
            dist, az, npts = _box_loc(dmap, bbox, K, T_cam_base)
            if not math.isfinite(dist) or dist < 0.05 or dist > 200:
                n_skip += 1
                continue
            dets.append({'cls': clsi, 'class_name': CLASS_NAMES[clsi] if clsi < len(CLASS_NAMES) else str(clsi),
                         'conf': conf, 'distance': dist, 'azimuth_deg': az,
                         'bbox': [round(v, 1) for v in bbox], 'grid_pts': int(npts)})
            n_box += 1
        out[stem] = dets
        n_ok += 1
        if (k + 1) % 40 == 0:
            print(f'  {k+1}/{len(stems)} 帧... (框 {n_box})', flush=True)

    with open(args.out, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'[done] 帧可定位 {n_ok}, 检测框 {n_box}, 跳过 {n_skip}; 写出 {args.out}', flush=True)


if __name__ == '__main__':
    main()
