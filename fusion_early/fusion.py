#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
④ 特征融合层 · 跨模态融合(独立 Self-Attn + 独立 Cross-Attn + 门控) —— 骨架版
================================================================================
对应 整体架构/模型架构图.dot 的 ④:
    det_civil / det_equip / det_track (图像特征) + pcl_enc(点云特征) + nav_enc(导航条件)
    → 融合 → 输出 特征金字塔 F2–F5 → 送 ⑤ 检测/路径决策双头

【设计 —— 两个独立注意力, 不是一整棵 Transformer 树】
  架构里写的是「交叉注意力(图像↔点云, 按外参空间对齐)+门控加权逐尺度融合」。
  正确拆法(每尺度一个块):
      ① Self-Attn    Q=K=V=图像token → 模态内空间/跨分支上下文
      ② Cross-Attn   Q=图像, K/V=点云 → 跨模态对齐(按外参空间)
      ③ 门控加权(SE) + FFN + LayerNorm
  这是【两个独立子层】, 而非复用同一块。轻量, 适合巡检实时。

【点云侧——先占位再插】
  pcl_feats=None  → 跳过 ②(Cross-Attn), 只跑 Self-Attn+门控, 图像三支内部先能真跑。
  pcl_feats=提供 → ②启用, 自动变 图像↔点云 交叉注意力。接口留好, 摘出来直接插。

【输入约定(与 det_encoders_runtime 一致)】
  branch_feats: {分支名:{mode:[P3,P4,P5]}} 或 {分支名:[P3,P4,P5]}, 各(B,C,h/stride,w/stride)。
  pcl_feats:   每尺度 token 序列 {尺度: (B,Np,D)} 或 list, 或 None。
  nav_embed:   (B,D) 导航状态嵌入(条件查询), 或 None。

【降采样防止 OOM】token 数超 max_tokens 时, 注意力前对空间做下采样, 完成后再上采样回原分辨率。

【输出】{尺度名: (B, embed_dim, h, w)} —— 即融合特征金字塔, 尺寸与对应输入尺度一致。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_BRANCH_ORDER = ("轨道", "土建", "设备")   # 统一分支顺序, 影响 concat


class ScaledFusionBlock(nn.Module):
    """一个尺度: ① Self-Attn → ② Cross-Attn(可选) → ③ 门控 + FFN。
    pcl_tokens 为 None 时跳过 ②(cross), 只做 ①③ —— 图像三支内部融合占位版。"""

    def __init__(self, in_channels: int, embed_dim: int = 256, heads: int = 4,
                 max_tokens: int = 4096, use_gate: bool = True, ffn_ratio: int = 4) -> None:
        super().__init__()
        if embed_dim % heads != 0:
            raise ValueError(f"embed_dim({embed_dim}) 需能被 heads({heads}) 整除")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.heads = heads
        self.max_tokens = max_tokens

        self.proj = nn.Conv2d(in_channels, embed_dim, 1, bias=True)

        # ① Self-Attention(模态内, Q=K=V=图像)
        self.self_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        # ② Cross-Attention(跨模态, Q=图像, K/V=点云) —— 点云未就绪则跳过
        self.cross_attn = nn.MultiheadAttention(embed_dim, heads, batch_first=True)

        self.norm1 = nn.LayerNorm(embed_dim)   # self-attn 后
        self.norm2 = nn.LayerNorm(embed_dim)   # cross-attn 后
        self.norm3 = nn.LayerNorm(embed_dim)   # FFN 后

        # ③ 门控加权(SE-like) —— "逐尺度加权融合"
        self.gate = (nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(embed_dim, embed_dim // 4, 1),
            nn.ReLU(inplace=True), nn.Conv2d(embed_dim // 4, embed_dim, 1),
            nn.Sigmoid(),
        ) if use_gate else None)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.GELU(), nn.Linear(embed_dim * ffn_ratio, embed_dim),
        )

    # ---- token 与空间互换 ----
    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)          # (B, emb, h, w) -> (B, hw, emb)

    def _pool_for_attn(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """若 token 数超 max_tokens, 下采样空间网格。返回 (池化后张量, 池化后的 hw)。
        down 因子由 max_tokens 决定; 上采回原分辨率用 _unpool(目标 = 传入 img_feat 的 hw)。"""
        hw = x.shape[-2:]
        n = hw[0] * hw[1]
        k = 1
        while n // (k * k) > self.max_tokens:
            k *= 2
        if k == 1:
            return x, hw
        xp = F.avg_pool2d(x, kernel_size=k, stride=k)
        return xp, (xp.shape[-2], xp.shape[-1])

    @staticmethod
    def _unpool(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """把池化后的特征图插值回原 hw。"""
        if x.shape[-2:] == hw:
            return x
        return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

    def forward(self, img_feat: torch.Tensor, pcl_tokens: torch.Tensor | None = None,
                nav_embed: torch.Tensor | None = None) -> torch.Tensor:
        # img_feat: (B, inC, h, w) —— 跨分支 concat 后的该尺度特征
        x = self.proj(img_feat)                       # (B, embed, h, w)
        hw = x.shape[-2:]

        # 下采样(防 OOM) → token 序列
        xp, hw = self._pool_for_attn(x)
        tokens = self._flatten(xp)                    # (B, hw', embed)

        # 导航条件注入(可选): 投影后加到 query 上
        if nav_embed is not None:
            tokens = tokens + nav_embed.unsqueeze(1)   # (B,1,D) 广播到所有 token

        # ① Self-Attention(模态内 / 跨分支上下文)
        a, _ = self.self_attn(query=tokens, key=tokens, value=tokens)
        tokens = self.norm1(tokens + a)

        # ② Cross-Attention(图像↔点云, 按外参空间对齐) —— 点云未提供则跳过
        if pcl_tokens is not None:
            a, _ = self.cross_attn(query=tokens, key=pcl_tokens, value=pcl_tokens)
            tokens = self.norm2(tokens + a)

        # FFN
        tokens = self.norm3(tokens + self.ffn(tokens))

        # 回到 2D 图并上采样回原分辨率
        z = tokens.transpose(1, 2).reshape(x.size(0), self.embed_dim, *hw)
        z = self._unpool(z, img_feat.shape[-2:])

        # ③ 门控加权(逐通道)
        if self.gate is not None:
            z = z * self.gate(z)
        return z


class CrossModalFusion(nn.Module):
    """跨模态融合层骨架: 图像三支(+可选点云) → 融合特征金字塔 {P3,P4,P5}。"""

    def __init__(self, per_scale_channels: dict, embed_dim: int = 256, heads: int = 4,
                 branches: tuple = _BRANCH_ORDER, max_tokens: int = 4096) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.branches = branches
        self.keep_scale_order = list(per_scale_channels.keys())
        self.scales = nn.ModuleDict({
            name: ScaledFusionBlock(ch, embed_dim, heads, max_tokens=max_tokens)
            for name, ch in per_scale_channels.items()
        })
        self.nav_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, branch_feats: dict, pcl_feats: dict | list | None = None,
                nav_embed: torch.Tensor | None = None) -> dict:
        # 规整每个分支为 [P3,P4,P5] list
        norm = {}
        for b, md in branch_feats.items():
            if isinstance(md, dict):
                md = next(iter(md.values()))
            if not isinstance(md, (list, tuple)) or len(md) != 3:
                raise ValueError(f"{b}: 每分支需 [P3,P4,P5] 三尺度, 收到 {type(md)}")
            norm[b] = list(md)

        nav = self.nav_proj(nav_embed) if nav_embed is not None else None

        out = {}
        for i, name in enumerate(self.keep_scale_order):
            feats = [norm[b][i] for b in self.branches if b in norm]
            if not feats:
                continue
            img_feat = torch.cat(feats, dim=1)         # (B, sumC, h, w)

            # 该尺度点云 token(可选)
            pcl_tok = None
            if pcl_feats is not None:
                pcl_tok = (pcl_feats[i] if isinstance(pcl_feats, (list, tuple))
                           else pcl_feats.get(name))
            out[name] = self.scales[name](img_feat, pcl_tok, nav)
        return out


# ------------------------------------------------------------------ 便捷封装
def total_channels_per_scale(network: str = "neck") -> dict:
    """每尺度(跨 3 分支 concat 后)通道数。neck: P3=256*3, P4=512*3, P5=512*3。"""
    if network == "neck":
        return {"P3": 256 * 3, "P4": 512 * 3, "P5": 512 * 3}
    if network == "backbone":
        return {"P3": 512 * 3, "P4": 512 * 3, "P5": 512 * 3}
    raise ValueError(f"network 只能 neck/backbone, 收到 {network!r}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    import det_encoders_runtime as R

    encs = R.load_all_encoders()
    img = os.environ.get("DEMO_IMAGE", os.path.join(os.path.dirname(__file__), "..", "data", "demo.png"))
    bf = R.encode_all(img, encs, imgsz=1024, modes=("neck",))

    fusion = CrossModalFusion(total_channels_per_scale("neck"), embed_dim=256, heads=4)
    fusion.eval()
    out = fusion(bf, pcl_feats=None, nav_embed=None)   # 点云/导航未就绪 → 占位(Self-Attn only)

    print("输入分支特征(neck):")
    for b, md in bf.items():
        print(f"  {b}: " + ", ".join(f"{p}{tuple(f.shape)}"
              for p, f in zip(("P3", "P4", "P5"), md["neck"])))
    print("\n融合输出(点云/导航 None → 只跑 Self-Attn+门控, 图像三支内部融合):")
    for k, t in out.items():
        print(f"  {k}: {tuple(t.shape)}")
    print("\nFUSION_OK")
