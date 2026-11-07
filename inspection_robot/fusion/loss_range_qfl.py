# -*- coding: utf-8 -*-
"""
Range-QFL —— 距离分层质量焦点损失 (裂缝巡检定制)
================================================================================
不是新数学, 是把两个已有机制按本项目的**实测失败模式**重新组合:

  ① Quality Focal Loss (Li et al., GFL 2020)
     q = TAL 的对齐度量, p = sigmoid(score)
     QFL = |q - p|^gamma * BCE(p, q)
     动机: 实测该模型全部框分数落在 0.010~0.022, 排序有效但绝对值不可用
           (工作点只能靠 opt_thresh 实测到 0.0057)。原版 BCE 的目标 q 本来就是
           对齐度软标签, 所以"分数-质量耦合"不是它的缺陷; 缺陷在于它对**所有**
           锚点一视同仁地上奏, 难样本被海量易负样本淹没。QFL 用 |q-p|^gamma
           把梯度集中到"还没学对的锚点"上。分数尺度能否因此回到可用区间,
           是**待实测**的问题, 不是本文件声称的结论。

  ② 尺度/距离分层加权 (小目标检测的常规做法, 如 NWD 一类的动机)
     w_i = (area_i / median_area)^(-beta), clip 到 [w_min, w_max]
     动机: 检测数随里程单调衰减 13->0, 闭环只能查到 8~10/16。
           透视投影下框面积 ~ 1/d^2, 所以"小框"就是"远处目标"。
           标准 TAL 按 IoU 分配, 远处小框 IoU 天然低 -> 正样本少 -> 梯度少。
           这里直接把远处目标的 cls 梯度上奏。

只改 cls 分支。box/dfl 与 ultralytics 原版逐字一致 —— 保证消融能归因:
QFL 改了分数语义, 尺度权重改了正样本配比, 两者分别可开关。
================================================================================
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss, make_anchors


class RangeQFLDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss 的 cls 分支替换为 QFL + 尺度加权, box/dfl 原样。

    参数:
        gamma   QFL 难易样本聚焦指数 (论文默认 2.0)
        beta    尺度权重指数 (0 = 关闭加权, 与 v8DetectionLoss 完全等价)
        w_min/w_max  权重截断, 防止极小目标把梯度全吃掉
    """

    def __init__(self, model, gamma: float = 2.0, beta: float = 0.5,
                 w_min: float = 0.5, w_max: float = 3.0, tal_topk: int = 10):
        super().__init__(model, tal_topk=tal_topk)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.last_stats: dict = {}

    def get_assigned_targets_and_loss(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:],
                             device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size,
                                  scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )

        # ---------- 尺度权重 ----------
        # gt_bboxes 是 @640 画布像素 xyxy; 面积 ~ 1/d^2, 用面积代理距离
        # 注意: TaskAlignedAssigner 在整批无 GT 时返回 float 全零 fg_mask, 必须 .bool() 归一
        fg_bool = fg_mask.bool()
        n_gt = gt_bboxes.shape[1]
        if n_gt > 0 and self.beta > 0:
            gt_area = ((gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=0)
                       * (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=0))
            valid = mask_gt.squeeze(-1).bool() & (gt_area > 0)
            if valid.any():
                med = gt_area[valid].median()
            else:
                med = torch.ones((), device=self.device, dtype=gt_area.dtype)
            anchor_area = torch.gather(gt_area, 1, target_gt_idx.clamp(min=0, max=n_gt - 1))
            scale = (anchor_area / med.clamp(min=1e-6)).clamp(min=1e-6).sqrt()
            w = scale.pow(-self.beta).clamp(self.w_min, self.w_max)
            # 背景锚点保持 1: 只重加权正样本, 否则全背景 batch 的 cls loss 会被归零,
            # 抹掉原版"整帧无目标也要压低背景分数"的约束
            w = torch.where(fg_bool, w, torch.ones_like(w))
        else:
            w = torch.ones_like(fg_mask, dtype=dtype)

        # ---------- QFL 取代 BCE ----------
        p = pred_scores.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction="none")
        qfl = (target_scores - p).abs().pow(self.gamma) * ce
        if self.class_weights is not None:
            qfl = qfl * self.class_weights

        # beta=0 走原版归一化分支, 保证 (gamma=0, beta=0) 与 v8DetectionLoss 逐位相等;
        # beta>0 时用"加权目标分数和", 与原版 target_scores.sum() 同量级, cls 增益无需重调
        target_scores_sum = max(target_scores.sum(), 1)
        if self.beta > 0:
            w_cls = w.unsqueeze(-1)
            loss[1] = (qfl * w_cls).sum() / (w_cls * target_scores).sum().clamp(min=1.0)
        else:
            loss[1] = qfl.sum() / target_scores_sum

        if self.beta > 0:
            w_pos = w[fg_bool]
            self.last_stats = {
                "n_gt": n_gt,
                "n_pos": int(fg_bool.sum()),
                "w_mean": float(w_pos.mean()) if w_pos.numel() else 0.0,
                "w_min_obs": float(w_pos.min()) if w_pos.numel() else 0.0,
                "w_max_obs": float(w_pos.max()) if w_pos.numel() else 0.0,
            }

        # ---------- box / dfl: 与 v8DetectionLoss 逐字一致 ----------
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores,
                max(target_scores.sum(), 1),
                fg_mask, imgsz, stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )
