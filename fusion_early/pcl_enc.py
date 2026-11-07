#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
③ 点云编码器 PointCloudEncoder —— pro/3 主项目版（自 pro/2/土建/渗漏水/pcl_enc.py 拷贝）
================================================================================
在 pro/2 原版基础上新增 `forward_maps()` —— 供点云检测式预训（train_pcl_detect.py）用：
返回每尺度【投影后的 2D 特征图】(B,embed_dim,H',W') 而不是 token 序列，直接喂 Detect head。
`forward()` 保持原版 token 契约（给 fusion 层 cross-attn K/V），两者完全独立、共享主干。

【输出契约】
  - forward(attr_maps)      -> list[ (B,Np,embed_dim) ]  x 3 尺度  —— 融合层 pcl_feats
  - forward_maps(attr_maps) -> list[ (B,embed_dim,H',W') ] x 3 尺度 stride 8/16/32 —— 检测头

【为什么 forward_maps 不做 max_tokens 截断】检测头需要全空间分辨率（640→80×80/40×40/20×20）；
forward() 的 max_tokens=512 自适应池化只适合融合 cross-attn 的定长 token，两用途不同。

用法:
    import pcl_enc
    enc = pcl_enc.PointCloudEncoder()                 # in_channels=4(深度/反射/高度/遮罩)
    attr = torch.rand(1, 4, 640, 640)                 # 4通道属性图, 归一化到 [0,1]
    tokens = enc(attr)        # list[(1,Np,256)] x3    —— 融合用
    maps   = enc.forward_maps(attr)  # list[(1,256,80/40/20,80/40/20)] —— 检测用
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ORDER = ("P3", "P4", "P5")   # 与融合层尺度序对应


class PointCloudEncoder(nn.Module):
    """轻量 2D-CNN: 属性图 → 逐尺度点云 token(供 cross-attn K/V) / 2D 特征图(供检测头)。"""

    def __init__(self, in_channels: int = 4, embed_dim: int = 256,
                 out_stages: int = 3, num_tokens_per_stage: int | None = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_stages = out_stages

        # 轻量 2D-CNN 主干: 下采样卷积 + 残差瓶颈 —— 每级降分辨率, 通道增
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
        )
        # stage: 分辨率逐级 /2, 通道逐级增, 供输出 3 档尺度
        self.stages = nn.ModuleList()
        in_ch = 128
        for i in range(out_stages):
            out_ch = embed_dim * (2 ** min(i, 2))   # 256, 512, 512
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch), nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch), nn.GELU(),
            ))
            in_ch = out_ch

        # 每个尺度: 投影到 embed_dim
        self.proj = nn.ModuleList(
            nn.Conv2d(embed_dim * (2 ** min(i, 2)), embed_dim, 1) for i in range(out_stages)
        )
        # 位置嵌入按尺度上限数预留
        self.max_tokens = num_tokens_per_stage or 512
        self.pos = nn.Parameter(
            torch.randn(out_stages, self.max_tokens, embed_dim) * 0.02
        )

    # ---------------- 融合用: token 序列 ----------------
    def forward(self, attr_maps: torch.Tensor) -> list[torch.Tensor]:
        """attr_maps: (B, inCh, H, W) 属性图([0,1])。
        返回: list 长度=out_stages, 每项 (B, Np, embed_dim) —— 每尺度点云 token。"""
        x = self.stem(attr_maps)         # (B, 128, H/4, W/4)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)                  # (B, C_i, H/4/2^i, W/4/2^i)
            f = self.proj[i](x)           # (B, embed_dim, H', W')
            n = f.shape[-2] * f.shape[-1]
            if n > self.max_tokens:       # 超上限则空间平均池化压到 max_tokens 以内
                ratio = (self.max_tokens ** 0.5) / max(f.shape[-2], f.shape[-1])
                ratio = max(0.05, min(1.0, ratio))
                f = F.interpolate(f, scale_factor=ratio, mode="bilinear", align_corners=False)
            tok = f.flatten(2).transpose(1, 2)           # (B, Np, embed_dim)
            Np = tok.size(1)
            p = self.pos[i, :Np, :]                       # 截到实际 Np
            outs.append(tok + p)
        return outs

    # ---------------- 检测用: 全分辨率 2D 特征图 ----------------
    def forward_maps(self, attr_maps: torch.Tensor) -> list[torch.Tensor]:
        """attr_maps: (B, inCh, H, W) 属性图([0,1])。返回 list[ (B,embed_dim,H',W') ]，
        stride 8/16/32 —— 对 640×640 输入即 80×80/40×40/20×20, 与 YOLO P3/P4/P5 对齐。
        不拉平、不做位置嵌入、不 max_tokens 截断(检测要全分辨率)。"""
        x = self.stem(attr_maps)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            outs.append(self.proj[i](x))
        return outs

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (f"PointCloudEncoder(轻量2D-CNN, params={n/1e6:.2f}M, "
                f"in通4(深度/反射/高度/遮罩), 出{self.out_stages}档, embed_dim={self.embed_dim})")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    enc = PointCloudEncoder()
    print(enc.describe())
    attr = torch.rand(1, 4, 640, 640)
    print(f"属性图: {tuple(attr.shape)}")
    tokens = enc(attr)
    print("点云 token(每尺度):")
    for i, t in enumerate(tokens):
        print(f"  {_ORDER[i]}: {tuple(t.shape)}")
    maps = enc.forward_maps(attr)
    print("检测特征图(每尺度):")
    for i, m in enumerate(maps):
        print(f"  {_ORDER[i]}: {tuple(m.shape)}")
    print("PCL_ENC_OK")
