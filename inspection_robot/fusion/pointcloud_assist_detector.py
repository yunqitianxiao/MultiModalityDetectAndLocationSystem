"""
PointCloudAssistedYOLO —— 点云辅助 2D 检测 (特征级融合)
================================================================
把刚建好的融合子系统接进 2D 检测: 点云 attrmap 与位姿经交叉注意力增强图像
neck 特征, 增强后的 image_scales 投影回 YOLO 通道, 喂给新建的 Detect head
(nc=7) 出框。冻结主干+neck(COCO 预训练, 将来换缺陷微调权重), 只训练
融合分支 + 检测头 —— 这正是"点云辅助2D"的特征级实现。

    imgs (B,3,640,640) ──ImageEncoder(冻结主干+neck)──▶ P3/P4/P5 → scale_proj
          │                                        → image_tokens(8400,256)
          │
attrmap ──PointCloudEncoder(可训)──▶ point_tokens(3600,256) ──CrossModalFusion(可训)──┐
pose   ──PoseEncoder(冻结)───────▶ (B,128) ──cond_proj───────────────────────▶ image_scales
                                                                                      │
                    head_proj 256→256/512/512 ──▶ [P3',P4',P5'] ──▶ Detect(nc=7,可训) ──▶ loss/框
                                                                                      │
use_point=False 时跳过点云/位姿融合, 直接用 scale_proj 的图像特征走同一路径 ——
即 RGB-only 基线, 与点云辅助版共享全部其它代码, 消融公平。

损失复用 ultralytics v8DetectionLoss (TaskAlignedAssigner + DFL), 只喂它
需要的 batch 字段: {'cls','bboxes'(xyxy@640),'batch_idx'}。Detect head 是
新建 nc=7(预训练是 nc=80, 通道不匹配), stride 复制原头 [8,16,32]。

用法:
    from fusion.pointcloud_assist_detector import PointCloudAssistedYOLO
    model = PointCloudAssistedYOLO(device='cuda')
    model.train()
    loss, loss_dict = model(imgs, attrmap, pose_state, batch)   # (loss.sum(), {box,cls,dfl})
"""

import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from fusion.config import (  # noqa: E402
    D_MODEL, DETECT_STRIDE, HYP, NC, NECK_CH, NECK_ORDER, OUTPUT_V7, POSE_DIM,
    POSE_EMB, REG_MAX, YOLO_SIZE,
)
from fusion.image_encoder import ImageEncoder        # noqa: E402
from fusion.pointcloud_encoder import PointCloudEncoder  # noqa: E402
from fusion.fusion import CrossModalFusion           # noqa: E402
from data_gen.pose_encoder import PoseEncoder        # noqa: E402


class _LossContainer(nn.Module):
    """v8DetectionLoss 需要 'model.args' + 'model.model[-1]' 的假模型壳。"""

    def __init__(self, detect, args):
        super().__init__()
        self.model = nn.ModuleList([nn.Identity(), detect])
        self.args = args
        self.class_weights = None


class PointCloudAssistedYOLO(nn.Module):
    """点云辅助 2D 检测: 冻结 YOLO 主干, 融合子系统增强图像特征, 喂新 Detect head。"""

    def __init__(self, ckpt=None, nc=NC, use_point=True,
                 freeze_backbone=True, unfreeze_neck=0, device=''):
        super().__init__()
        self.nc = nc
        self.use_point = use_point
        self.freeze_backbone = freeze_backbone
        self.unfreeze_neck = unfreeze_neck
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        from fusion.config import POSE_CKPT, YOLO11L_CKPT
        ckpt = ckpt or YOLO11L_CKPT

        # ---- 编码器 + 融合 (冻结主干由 ImageEncoder 内部处理) ----
        self.image_encoder = ImageEncoder(ckpt=ckpt, freeze_backbone=freeze_backbone,
                                          unfreeze_neck=unfreeze_neck, device=self.device)
        self.point_encoder = PointCloudEncoder(device=self.device)
        self.fusion = CrossModalFusion(device=self.device)
        self.pose_encoder = PoseEncoder(POSE_CKPT)     # 冻结, 非 nn.Module

        # ---- 增强特征投影回 YOLO neck 通道: image_scales(256) -> 256/512/512 ----
        self.head_proj = nn.ModuleDict({
            name: nn.Conv2d(D_MODEL, NECK_CH[name], 1) for name in NECK_ORDER
        })

        # ---- 新建 nc 检测头 (预训练头是 nc=80, 通道不匹配) ----
        from ultralytics.nn.modules.head import Detect
        self.detect = Detect(nc=nc, reg_max=REG_MAX, end2end=False,
                             ch=tuple(NECK_CH[n] for n in NECK_ORDER))
        self.detect.stride = torch.tensor(DETECT_STRIDE)

        self.to(self.device)

        # ---- 检测损失 (v8DetectionLoss, 复用 TAL+DFL) ----
        # 必须在 .to(device) 之后创建, 否则 criterion 内部 (bce/bbox_loss/proj)
        # 会留在创建时刻的 device, 与已搬走的 detect/前向特征不一致。
        from ultralytics.utils.loss import v8DetectionLoss
        self.criterion = v8DetectionLoss(_LossContainer(self.detect, SimpleNamespace(**HYP)))

    # ---------------- 前向 ----------------
    def forward(self, imgs, attrmap, pose_state, batch=None):
        """imgs(B,3,640,640) + attrmap(numpy/tensor (B,3,720,1280)) + pose_state(B,12)。
        给 batch={'cls','bboxes','batch_idx'} 则返回 (loss.sum(), loss_dict);
        否则返回 Detect 原始 preds dict。"""
        img_out = self.image_encoder(imgs)             # {P3..P5, tokens}

        if self.use_point:
            pc_out = self.point_encoder(attrmap)       # point_tokens(3600,256)
            # PoseEncoder 模型在 CPU, 先送回 CPU 跑; fusion 内部再搬到目标设备
            pe = self.pose_encoder(pose_state.to('cpu') if torch.is_tensor(pose_state) else pose_state)
            if pe.dim() == 1:
                pe = pe.unsqueeze(0)
            fused = self.fusion(img_out['tokens'], pc_out['tokens'], pe)
            scales = fused.image_scales                # {P3..P5: (B,256,H,W)}
        else:
            scales = {name: img_out[name] for name in NECK_ORDER}

        feats = [self.head_proj[name](scales[name]) for name in NECK_ORDER]
        preds = self.detect(feats)

        if batch is not None:
            loss, loss_dict = self.criterion(preds, batch)
            return loss.sum(), loss_dict
        return preds

    def trainable_params(self):
        """返回可训练参数迭代器 (冻结主干已被 ImageEncoder 置 requires_grad=False)。"""
        return (p for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"PointCloudAssistedYOLO(nc={self.nc}, use_point={self.use_point}, "
                f"freeze_backbone={self.freeze_backbone}, unfreeze_neck={self.unfreeze_neck}, "
                f"device={self.device})")


# ================================================================
# 数据装载辅助 (output_v7 配对样本 → 训练 batch)
# ================================================================

def letterbox_boxes(labels_norm, img_hw, new_size=YOLO_SIZE):
    """YOLO 归一化标签 (N,5) [cls,xc,yc,w,h] -> letterbox 后图像像素 xyxy (N,4)。
    变换与 ultralytics LetterBox(new_shape=(640,640), auto=False) 一致:
    r=min(640/H,640/W), 新图 (round(W*r), round(H*r)), 居中补 (dw,dh)。"""
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
    xyxy = np.clip(xyxy, 0.0, new_size)
    return cls, xyxy


def pose_json_to_state(p):
    """poses/{i}.json (odom) -> 12 维 /pose_state, 字段序与 pose_state_tunnel.csv 一致
    (x,y,z,sin_roll,cos_roll,sin_pitch,cos_pitch,sin_yaw,cos_yaw,vx,vy,wz)。JSON 无速度 -> 0。"""
    x, y, z = p['x'], p['y'], p['z']
    qx, qy, qz, qw = p['qx'], p['qy'], p['qz'], p['qw']
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.array(
        [x, y, z, math.sin(roll), math.cos(roll), math.sin(pitch),
         math.cos(pitch), math.sin(yaw), math.cos(yaw), 0.0, 0.0, 0.0],
        dtype=np.float32)


def load_sample(i, root=OUTPUT_V7, target_class=None):
    """读第 i 个配对样本 -> (tensor640 BGR float[0,1], attrmap(3,720,1280) raw,
    pose_state(12,), cls(N,), xyxy(N,4) @640)。标签空则 cls/xyxy 为空数组。

    target_class: 非空时把输出收窄成"只检测该类"——只保留 cls==target_class 的框并重映射
    为 0(nc=1 渗漏水检测), 其它类(扣件/裂缝/剥落/设备)框全部丢弃成背景负样本。这样
    检测头是 nc=1, 训练/评估里非渗漏水物体不产生正样本, 只当 FP 来抑制。"""
    import cv2
    img_bgr = cv2.imread(os.path.join(root, 'images', f'{i:06d}.jpg'))
    assert img_bgr is not None, f'缺图像 {i}'
    from ultralytics.data.augment import LetterBox
    lb = LetterBox(new_shape=(YOLO_SIZE, YOLO_SIZE), auto=False, stride=32)
    img640 = lb(image=img_bgr)
    tensor = np.transpose(img640.astype(np.float32) / 255.0, (2, 0, 1))   # BGR (3,640,640)

    lbl_path = os.path.join(root, 'labels', f'{i:06d}.txt')
    if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
        labels = np.loadtxt(lbl_path, dtype=np.float32).reshape(-1, 5)
    else:
        labels = np.empty((0, 5))                 # 负样本(无 GT 框)合法
    cls, xyxy = letterbox_boxes(labels, img_bgr.shape[:2])
    if target_class is not None:
        keep = cls == target_class               # 只留目标类
        cls = np.zeros(int(keep.sum()), np.float32)   # 重映射为类 0 (nc=1)
        xyxy = xyxy[keep]

    a = np.load(os.path.join(root, 'attrmaps', f'{i:06d}.npz'))
    attrmap = np.stack([a['depth'], a['reflec'], a['height']]).astype(np.float32)

    with open(os.path.join(root, 'poses', f'{i:06d}.json')) as f:
        pose_state = pose_json_to_state(json.load(f))
    return tensor, attrmap, pose_state, cls, xyxy


def collate_batch(samples):
    """list[load_sample 返回] -> (imgs(B,3,640,640), attrmaps(B,3,720,1280),
    poses(B,12), batch dict)。

    batch dict 字段约定 (v8DetectionLoss.preprocess):
      cls (N,), bboxes (N,4) —— **640 画布上的 normalized xywh**, 不是像素 xyxy!
      (loss 内部会 bboxes*imgsz 变像素再 xywh2xyxy)
      batch_idx (N,) 每框所属样本下标。
    无数框的样本为空数组, loss 按全背景处理。"""
    imgs = np.stack([s[0] for s in samples])
    attrmaps = np.stack([s[1] for s in samples])
    poses = np.stack([s[2] for s in samples])
    cls_list, xyxy_list, bidx_list = [], [], []
    for bi, s in enumerate(samples):
        cls_list.append(s[3]); xyxy_list.append(s[4])
        bidx_list.append(np.full(s[4].shape[0], bi, dtype=np.float32))
    cls = np.concatenate(cls_list) if cls_list and len(cls_list[0]) else np.empty((0,), np.float32)
    xyxy = np.concatenate(xyxy_list) if xyxy_list and len(xyxy_list[0]) else np.empty((0, 4), np.float32)
    bidx = np.concatenate(bidx_list) if bidx_list and len(bidx_list[0]) else np.empty((0,), np.float32)
    if xyxy.shape[0]:
        xc = (xyxy[:, 0] + xyxy[:, 2]) / 2
        yc = (xyxy[:, 1] + xyxy[:, 3]) / 2
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        xywh = np.stack([xc, yc, w, h], 1) / YOLO_SIZE
    else:
        xywh = np.empty((0, 4), np.float32)
    batch = {'cls': torch.from_numpy(cls), 'bboxes': torch.from_numpy(xywh),
             'batch_idx': torch.from_numpy(bidx)}
    return (torch.from_numpy(imgs), torch.from_numpy(attrmaps),
            torch.from_numpy(poses), batch)


if __name__ == '__main__':
    # 链路冒烟: 一个真实样本前向 + 反向梯度有限
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PointCloudAssistedYOLO(device=dev)
    model.train()
    print(model)
    imgs, attrmaps, poses, batch = collate_batch([load_sample(0)])
    imgs = imgs.to(dev); attrmaps = attrmaps.to(dev); poses = poses.to(dev)
    for k, v in batch.items():
        batch[k] = v.to(dev)
    loss, ld = model(imgs, attrmaps, poses, batch)
    print('loss:', float(loss.detach()), {k: float(v) for k, v in ld.items()})
    loss.backward()
    bad = sum(1 for p in model.trainable_params()
              if p.grad is None or not bool(torch.isfinite(p.grad).all()))
    n = sum(1 for _ in model.trainable_params())
    print(f'可训练 {n} 个, 梯度有限 check: {bad} 个异常')
    # 消融路径 (use_point=False) 同样本也应能前向
    model2 = PointCloudAssistedYOLO(device=dev, use_point=False)
    model2.train()
    loss2, _ = model2(imgs, attrmaps, poses, batch)
    print('RGB-only loss:', float(loss2.detach()))
    assert bad == 0 and float(loss.detach()) == float(loss.detach()), '链路异常'
