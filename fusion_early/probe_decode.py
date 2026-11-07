# -*- coding: utf-8 -*-
"""一次性探针: 打印 fusion wrapper 在 eval 下 preds 的确切结构, 用于定稿 decode。"""
import json
import os
import sys

import numpy as np
import torch

_P = os.path.dirname(os.path.abspath(__file__))
if _P not in sys.path:
    sys.path.insert(0, _P)
import train_fusion as T  # noqa: E402

sys.path.insert(0, os.path.join(_P, 'localization'))
from img_px_to_world import camera_to_base, pixel_to_camera  # noqa: E402
import math

BR = '土建'
IMGSZ = 640
IMG_SIZE = (1280, 720)          # (W,H) 原生
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
torch.set_grad_enabled(False)

wrapper, nc = T.build_model(BR, img_imgsz=IMGSZ, device='cuda')
wrapper.eval()
_ROOT = os.environ.get('PROJECT_ROOT', _P)
_FUSION_CKPT = os.environ.get('FUSION_CKPT', os.path.join(_ROOT, 'weights', 'fusion_土建.pt'))
_CALIB = os.environ.get('CALIBRATION_JSON', os.path.join(_ROOT, 'data', 'calibration.json'))
_DATA_ROOT = os.environ.get('DATA_ROOT', os.path.join(_ROOT, 'data', 'output_big_s3'))

sd = torch.load(_FUSION_CKPT,
                map_location='cuda', weights_only=False)
wrapper.load_state_dict(sd)
print('[load] ckpt 已载入; nc =', nc)

cal = json.load(open(_CALIB))
K = np.asarray(cal['K'], float)
T_cam_base = np.asarray(cal['T_cam_base'], float)
print('[calib] K0,0=%.3f cx=%.2f cy=%.2f img_size=%s' % (K[0, 0], K[0, 2], K[1, 2], cal.get('img_size')))

root = _DATA_ROOT
stems = T.resolve_label_paths(root)
print('[data] 帧数', len(stems))
stem = stems[0]
s = T.load_sim_frame(root, stem, IMGSZ, T._BRANCHES[BR]['sim_cls'], nc)
img, attr, nav = s[0], s[1], s[2]
gt_cls, gt_xyxy = s[3], s[4]
print('[frame] stem=%s img=%s attr=%s nav=%s gt_n=%d' % (stem, tuple(img.shape), tuple(attr.shape),
                                                         tuple(nav.shape), gt_xyxy.shape[0]))
nav = torch.from_numpy(np.asarray(nav, np.float32))
with torch.no_grad():
    preds = wrapper(img.unsqueeze(0).cuda(), attr.unsqueeze(0).cuda(), nav.unsqueeze(0).cuda())
print('[preds] type=%s len=%s' % (type(preds).__name__, len(preds) if isinstance(preds, (list, tuple)) else '-'))
if isinstance(preds, (list, tuple)):
    for i, p in enumerate(preds):
        print('  [%d] %s shape=%s' % (i, type(p).__name__, tuple(p.shape) if hasattr(p, 'shape') else p))
        if hasattr(p, 'shape') and p.ndim == 3:
            a = np.asarray(p.cpu().float())[0]     # (7, N)
            print('      行0-3(boxxywh) min=%.2f max=%.2f' % (float(a[:4].min()), float(a[:4].max())))
            c = a[4:]
            sm = float(1.0 / (1.0 + math.exp(-float(c.max()))))
            print('      行4-6(class) min=%.2f max=%.2f  max过sigmoid=%.4f' % (float(c.min()), float(c.max()), sm))
            print('      class min行 max样本(默认):', 'n/a')
            print('      class max 所在列索引:', int(c.max(1)[1]))
elif isinstance(preds, dict):
    print('  dict keys', list(preds.keys()))
    for k, v in preds.items():
        print('   ', k, type(v).__name__, tuple(v.shape) if hasattr(v, 'shape') else v)
else:
    print('  shape', tuple(preds.shape) if hasattr(preds, 'shape') else preds)

# GT 帧里若有框, 打印一下 GT xyxy@640 作对照
print('[gt] 类=%s ' % gt_cls.tolist() if gt_cls.size else '[gt] 无框')
if gt_xyxy.size:
    print('  gt_xyxy@640 numpy:\n', np.asarray(gt_xyxy).round(1))

# dmap 与原生尺寸
dmap = np.load(os.path.join(root, 'attrmaps', f'{stem}.npz'))['depth']
print('[dmap] shape=%s min=%.3f max=%.3f' % (dmap.shape, float(dmap.min()), float(dmap.max())))
