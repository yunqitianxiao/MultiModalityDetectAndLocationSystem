"""配对样本装载：图像 / attrmap / 标签 -> 训练 batch。

letterbox 变换必须与 ultralytics LetterBox(new_shape=(640,640), auto=False) 一致，
否则 GT 框与网络输入错位。
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as C

_FIXED_ATTR = (int(os.environ['TUNNEL_FIXED_ATTR'])
               if os.environ.get('TUNNEL_FIXED_ATTR') else None)


def list_frames(root=None):
    root = root or C.DATA_ROOT
    d = os.path.join(root, 'images')
    return sorted(int(os.path.splitext(f)[0]) for f in os.listdir(d) if f.endswith('.jpg'))


def split_indices(frames, val_ratio=0.2, mode='sequential', seed=42):
    """sequential：按序列尾部留出（默认，避免相邻帧泄漏）。
    random：随机划分（与历史实验可比，但同序列相邻帧会跨集）。"""
    if mode == 'sequential':
        n_val = max(1, int(len(frames) * val_ratio))
        return frames[-n_val:], frames[:-n_val]
    if mode == 'random':
        idx = list(frames)
        random.Random(seed).shuffle(idx)
        n_val = max(1, int(len(frames) * val_ratio))
        return sorted(idx[:n_val]), sorted(idx[n_val:])
    raise ValueError(mode)


def letterbox_boxes(labels_norm, img_hw, new_size=C.YOLO_SIZE):
    """YOLO 归一化标签 (N,5) -> letterbox 后 640 画布像素 xyxy (N,4)。"""
    if labels_norm is None or len(labels_norm) == 0:
        return np.empty((0,), np.float32), np.empty((0, 4), np.float32)
    labels_norm = np.asarray(labels_norm, np.float32)
    H, W = img_hw
    r = min(new_size / H, new_size / W)
    nw, nh = round(W * r), round(H * r)
    dw, dh = (new_size - nw) / 2.0, (new_size - nh) / 2.0
    cls = labels_norm[:, 0]
    x = labels_norm[:, 1] * W * r + dw
    y = labels_norm[:, 2] * H * r + dh
    w = labels_norm[:, 3] * W * r
    h = labels_norm[:, 4] * H * r
    xyxy = np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
    return cls, np.clip(xyxy, 0.0, new_size)


def load_sample(i, root=None, with_attr=True):
    """-> (img_t(3,640,640) BGR, attr(3,720,1280)|None, cls(N,), xyxy(N,4)@640)"""
    import cv2
    from ultralytics.data.augment import LetterBox

    root = root or C.DATA_ROOT
    img_bgr = cv2.imread(os.path.join(root, 'images', f'{i:06d}.jpg'))
    assert img_bgr is not None, f'缺图像 {i}'

    lb = LetterBox(new_shape=(C.YOLO_SIZE, C.YOLO_SIZE), auto=False, stride=32)
    img640 = lb(image=img_bgr)
    img_t = np.transpose(img640.astype(np.float32) / 255.0, (2, 0, 1))

    lbl_path = os.path.join(root, 'labels', f'{i:06d}.txt')
    if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
        labels = np.loadtxt(lbl_path, dtype=np.float32).reshape(-1, 5)
    else:
        labels = np.empty((0, 5))
    cls, xyxy = letterbox_boxes(labels, img_bgr.shape[:2])

    attr = None
    if with_attr:
        # TUNNEL_FIXED_ATTR=<帧号>：所有帧共用同一张 attrmap。点云支照常工作但
        # 传递零信息，用来单独量出"融合层本身"的成本（BN 统计也保持正常）。
        src = _FIXED_ATTR if _FIXED_ATTR is not None else i
        a = np.load(os.path.join(root, 'attrmaps', f'{src:06d}.npz'))
        ref = a['reflec']
        if C.ZERO_REFLEC:
            ref = np.zeros_like(ref)
        attr = np.stack([a['depth'], ref, a['height']]).astype(np.float32)
    return img_t, attr, cls.astype(np.float32), xyxy.astype(np.float32)


class PairedDataset(Dataset):
    def __init__(self, indices, root=None, with_attr=True):
        self.indices = list(indices)
        self.root = root or C.DATA_ROOT
        self.with_attr = with_attr

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        return load_sample(self.indices[k], root=self.root, with_attr=self.with_attr)


def collate_batch(samples):
    """-> (imgs(B,3,640,640), attrmaps(B,3,720,1280)|None, batch dict)

    batch 的 bboxes 是 **640 画布上的归一化 xywh**（v8DetectionLoss.preprocess 的约定）。
    """
    imgs = np.stack([s[0] for s in samples])
    attrs = None
    if samples[0][1] is not None:
        attrs = np.stack([s[1] for s in samples])

    cls_list, xyxy_list, bidx_list = [], [], []
    for bi, s in enumerate(samples):
        cls_list.append(s[2])
        xyxy_list.append(s[3])
        bidx_list.append(np.full(s[3].shape[0], bi, dtype=np.float32))

    cls = np.concatenate(cls_list) if cls_list else np.empty((0,), np.float32)
    xyxy = np.concatenate(xyxy_list) if xyxy_list else np.empty((0, 4), np.float32)
    bidx = np.concatenate(bidx_list) if bidx_list else np.empty((0,), np.float32)

    if xyxy.shape[0]:
        xc = (xyxy[:, 0] + xyxy[:, 2]) / 2
        yc = (xyxy[:, 1] + xyxy[:, 3]) / 2
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        xywh = np.stack([xc, yc, w, h], 1) / C.YOLO_SIZE
    else:
        xywh = np.empty((0, 4), np.float32)

    batch = {'cls': torch.from_numpy(cls), 'bboxes': torch.from_numpy(xywh),
             'batch_idx': torch.from_numpy(bidx)}
    return (torch.from_numpy(imgs),
            torch.from_numpy(attrs) if attrs is not None else None,
            batch)
