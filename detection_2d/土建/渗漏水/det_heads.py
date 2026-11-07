#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⑤ 检测解码器 —— 融合特征 F2–F5 → 通道适配 → YOLO 检测头(复用结构, 微调)
================================================================================
这里是把「跨分支自注意力融合」接到检测的关键一环。原 YOLO 检测头(Detect)吃的是
各分支自己的 P3/P4/P5(C=256/512/512), 而融合层输出统一 256。所以要用一个
【通道适配器】把融合输出映射回检测头要的通道, 再接原 Detect 结构。

【结构】
    F2–F5 (B,256,H,W)          ← CrossModalFusion 输出(融合特征金字塔)
      ↓ 通道适配: P3:256→256, P4:256→512, P5:256→512   ← 1x1 conv
    (B, 各尺度 C, H, W)        ← 对齐到原 Detect 头输入通道
      ↓ YOLO Detect(复用 structure, 微调)
    {boxes, scores, cls}

【设计选择】
  - 复用原 YOLO Detect 头结构(含 anchor/cls/reg 分支), 但通道适配层+检测头需微调:
    因为输入通道变了、且是融合(跨分支)特征。训练时编码器+融合层冻结, 只训适配+头。
  - 各尺度通道=原头 in_channels: neck 下 P3=256, P4=512, P5=512。
  - 检测头用原 weight 的 model.model[-1] 结构(复制) + 新适配层, 加 1x1 投影到各尺度通道。

【为什么不用原头直接吃融合输出】
  原头 Conv 权重是按 (256/512/512, 类别) 训的, 融合输出 256 通道对不上。
  ★直接喂会 shape error★。所以必须中间垫一个适配卷积。

【注意】本模块不替代原来的每个分支头 —— 三个独立检测器仍在各自 stage 下可用;
        这里多的是"融合后加一个统一头"的可选件, 供"看跨分支增益"验证用。

用法:
    import det_heads as H
    det = H.FusionDetectHead(num_classes=3, per_scale_channels=(256,512,512), embed_dim=256)
    fused = fusion_out   # dict{'P3':(B,256,H,W), 'P4':..., 'P5':...}
    boxes = det(fused, feature_map_sizes=...)   # 见 test()
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Detect  # 复用 ultralytics 的 Detect(含 anchor 处理)


class FusionDetectHead(nn.Module):
    """融合特征 → 通道适配 → YOLO Detect 头(微调)。"""

    def __init__(self, num_classes: int = 3,
                 per_scale_channels: tuple[int, ...] = (256, 512, 512),
                 embed_dim: int = 256) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.per_scale_channels = per_scale_channels
        self.embed_dim = embed_dim

        # 通道适配: 每个尺度 1x1 conv 把 embed_dim 映射到检测头要的通道
        self.adapters = nn.ModuleList(
            nn.Conv2d(embed_dim, c, 1, bias=True) for c in per_scale_channels
        )

        # 复用 ultralytics Detect 结构(未经训练的骨架, 需要微调)
        # 注意: Detect 内部回归头通道数由 nc 决定, 这里给个占位 stride=8 即可,
        #       实际 stride 由训练时 imgsz 决定, 此处仅保证结构可用。
        self.detect = Detect(nc=num_classes, ch=list(per_scale_channels))
        # Detect 期望 stride 存在; 先给默认, 训练时用 model.val 覆盖
        if not hasattr(self.detect, "stride") or self.detect.stride is None:
            self.detect.stride = torch.tensor([8., 16., 32.], device="cpu")

    def forward(self, fused: dict, imgsz: int = 1024) -> dict | list:
        """fused: {'P3':(B,256,H,W), 'P4':..., 'P5':...} 顺序按名字。
        返回 YOLO Detect 的原始输出(预测)/训练时 loss 由 model 调度。"""
        # 按固定尺度顺序取出并适配
        names = ("P3", "P4", "P5")
        x = []
        for i, nname in enumerate(names):
            if nname not in fused:
                continue
            t = self.adapters[i](fused[nname])
            x.append(t)
        # x: list[(B, C_i, H_i, W_i)]  → 送 YOLO Detect
        return self.detect(x)


def load_detect_from_pretrained(ckpt_path: str, num_classes: int,
                                per_scale_channels: tuple[int, ...] = (256, 512, 512),
                                embed_dim: int = 256) -> FusionDetectHead:
    """从训练好的 best.pt 里拷贝原 Detect 头权重(回归/分类层), 适配层随机初始化。
    这样检测头尽量继承原分支检测能力, 只重训适配层+微调头。"""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = ck["model"]
    head = FusionDetectHead(num_classes, per_scale_channels, embed_dim)
    # 从原头的 Detect(在 m.model[-1])拷贝能对上的权重(分类/回归); 通道对不上的跳过
    # 这是一个"尽力继承"示例; 实际训练建议直接整头微调。
    orig = m.model[-1]
    for name, param in orig.state_dict().items():
        if name in head.detect.state_dict() and param.shape == head.detect.state_dict()[name].shape:
            head.detect.state_dict()[name].copy_(param)
    return head


def build_fusion_model(embed_dim: int = 256, head_channels: tuple[int, ...] = (256, 512, 512),
                       num_classes: int = 3) -> nn.Module:
    """快速构建 编码器(冻结)+融合+检测头 的整体, 前向 shape 校验用。"""
    return FusionDetectHead(num_classes, head_channels, embed_dim)


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    print("=== det_heads 前向+反向 shape 校验 ===")
    # 模拟 fusion 输出的融合特征(跨分支 self-attn 已生成, 这里喂 256 通道)
    H_, W_ = 64, 64
    fused = {"P3": torch.rand(1, 256, H_, W_, requires_grad=True),
             "P4": torch.rand(1, 256, H_ // 2, W_ // 2, requires_grad=True),
             "P5": torch.rand(1, 256, H_ // 4, W_ // 4, requires_grad=True)}
    for c in fused.values():
        # 让这两个值连通图, 用于反向验证
        fused["P3"] = fused["P3"] * 1.0
        break

    det = FusionDetectHead(num_classes=3, per_scale_channels=(256, 512, 512), embed_dim=256)
    out = det(fused)
    # YOLO Detect 返回可能是 tuple / list / dict, 打印结构并取一个可微张量做反向
    def collect(t, acc):
        if isinstance(t, torch.Tensor):
            acc.append(t)
        elif isinstance(t, (list, tuple)):
            for x in t:
                collect(x, acc)
        elif isinstance(t, dict):
            for v in t.values():
                collect(v, acc)
    acc = []
    collect(out, acc)
    print(f"Detect 输出类型: {type(out).__name__}, 收集到 {len(acc)} 个张量")
    for i, t in enumerate(acc):
        print(f"  张量[{i}] shape={tuple(t.shape)}")

    # 反向验证: 造一个假的 loss 反向
    loss = sum(t.float().abs().mean() for t in acc)
    print("假 loss =", float(loss))
    loss.backward()
    print("反向传播 OK(检测头+适配器可训练)")
    print("DET_HEADS_OK")
