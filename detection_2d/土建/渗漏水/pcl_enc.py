#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
③ 点云编码器 PointCloudEncoder —— 轻量 2D-CNN（属性图→点云特征）骨架
================================================================================
对应 整体架构/模型架构图.dot 的 ③ pcl_enc:
    ② 数据对齐层(空间对齐投影 + 点云属性图) → pcl_enc(轻量ViT / 2D-CNN) → 点云特征 → ④ 融合层

【设计 —— 轻量 2D-CNN】
  架构写的是「轻量 ViT / 2D-CNN（属性图→点云特征）」。这里用【轻量 2D-CNN】,
  输入是【点云属性图】(由空间对齐投影按像素网格生成, 每像素带 深度/反射率/高度/遮罩),
  输出每尺度的【点云 token 序列】, 直接喂 fusion.py 的 cross-attn K/V。

【属性图通道(占位)】深度X / 反射率 / 高度图 / 有效遮罩 —— 与架构图"弱纹理·暗光增强通道"一致。
  真实点云时由 数据对齐层(空间对齐投影) 生成; 占位阶段用随机/归一化填充, 后面真数据直接换。

【输出契约 —— 与 fusion.py 的 pcl_feats 对齐】
  forward(attr_maps) -> list[每尺度token]  长度 = 尺度数(默认3, 对应 P3/P4/P5)。
  每个 token: (B, Np, embed_dim), Np = 该尺度投影网格的 token 数(与图像 hw 解耦)。
  fusion.CrossModalFusion 的 pcl_feats 直接吃这个 list 即可。

【多尺度】为融合层逐尺度对齐, 输出 前向网络可产出 3 档 token(不同 stride)。
  实现: 属性图经过逐级下采样卷积, 在 3 个 stage 各自拉平成 token(带可学习位置嵌入)。

用法:
    import pcl_enc
    enc = pcl_enc.PointCloudEncoder()
    attr = torch.rand(1, 4, 512, 512)      # (B, 4通道, H, W) 占位属性图
    tokens = enc(attr)                      # list[ (1, Np, 256) ]  x 3尺度
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ORDER = ("P3", "P4", "P5")   # 与融合层尺度序对应(输出3档token)


class PointCloudEncoder(nn.Module):
    """轻量 2D-CNN: 属性图 → 逐尺度点云 token(供 cross-attn K/V)。"""

    def __init__(self, in_channels: int = 4, embed_dim: int = 256,
                 out_stages: int = 3, num_tokens_per_stage: int | None = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_stages = out_stages

        # 轻量 2D-CNN 主干: 下采样卷积 + 残差瓶颈 —— 每级降分辨率, 通道增
        # 4通道(深度/反射/高度/遮罩) → 256 特征
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
        )
        # stage: 分辨率逐级 /2, 通道逐级增, 供输出 3 档尺度 token
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

        # 每个尺度: 投影到 embed_dim 并加可学习位置嵌入
        self.proj = nn.ModuleList(
            nn.Conv2d(embed_dim * (2 ** min(i, 2)), embed_dim, 1) for i in range(out_stages)
        )
        # 位置嵌入按尺度上限数预留, 用 nn.Parameter 而非 ParameterList 才能保存 state_dict
        self.max_tokens = num_tokens_per_stage or 512
        self.pos = nn.Parameter(
            torch.randn(out_stages, self.max_tokens, embed_dim) * 0.02
        )
        # 自适应下采样: 让每尺度 token 数不超过 max_tokens
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, attr_maps: torch.Tensor) -> list[torch.Tensor]:
        """attr_maps: (B, inCh, H, W) 属性图(占位可 rand)。
        返回: list 长度=out_stages, 每项 (B, Np, embed_dim) —— 每尺度点云 token。"""
        x = self.stem(attr_maps)         # (B, 128, H/4, W/4)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)                  # (B, C_i, H/4/2^i, W/4/2^i)
            f = self.proj[i](x)           # (B, embed_dim, H', W')
            # 超过 max_tokens 则空间平均池化压到 max_tokens 以内(先估 token 数)
            n = f.shape[-2] * f.shape[-1]
            if n > self.max_tokens:
                ratio = (self.max_tokens ** 0.5) / max(f.shape[-2], f.shape[-1])
                ratio = max(0.05, min(1.0, ratio))
                f = F.interpolate(f, scale_factor=ratio, mode="bilinear", align_corners=False)
            tok = f.flatten(2).transpose(1, 2)           # (B, Np, embed_dim)
            Np = tok.size(1)
            p = self.pos[i, :Np, :]                       # 截到实际 Np
            outs.append(tok + p)
        return outs

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (f"PointCloudEncoder(轻量2D-CNN, params={n/1e6:.2f}M, "
                f"in通4(深度/反射/高度/遮罩), 出{self.out_stages}档token, embed_dim={self.embed_dim})")


def make_attr_maps(batch: int = 1, h: int = 512, w: int = 512, device: str = "cpu") -> torch.Tensor:
    """占位属性图: (B, 4, H, W) —— 深度/反射率/高度/遮罩, [0,1]。
    真实点云时替换成 数据对齐层投影结果。"""
    x = torch.rand(batch, 4, h, w, device=device)
    mask = (torch.rand(batch, 1, h, w, device=device) > 0.2).float()  # 模拟遮挡/无效区
    x[:, 3:4] = mask
    x = x * mask                          # 无效区归零
    return x


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    enc = PointCloudEncoder()
    print(enc.describe())
    attr = make_attr_maps(1, 512, 512)
    print(f"属性图: {tuple(attr.shape)}")
    tokens = enc(attr)
    print("点云 token(每尺度):")
    for i, t in enumerate(tokens):
        print(f"  {_ORDER[i]}: {tuple(t.shape)}")
    print("PCL_ENC_OK")
