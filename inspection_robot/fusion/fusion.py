"""
CrossModalFusion —— 交叉注意力·位姿条件融合 (核心模块)
================================================================
多模态特征融合: 图像 tokens (YOLO11l neck 三尺度, 8400) + 点云 tokens
(attrmap CNN, 3600) 作 **query**, 位姿嵌入 (PoseEncoder, 128 维) 经 MLP
映射成 **N_COND 个条件 token 作 key/value**, 做 Pre-LN 交叉注意力 + FFN。

拓扑 (用户拍板: 交叉注意力·位姿条件):
    Q = concat(image_tokens, point_tokens)   各加模态 embedding(2)
        图像另按尺度加 scale embedding(3: P3/P4/P5)
    K/V = pose_emb(B,128) -> Linear(128,d)+GELU+Linear(d,N_COND*d)
          -> (B,N_COND,d), N_COND=2
    注意力矩阵 (B,8,12000,2): 与 token 数近似线性, 远好于全自注意力
    (12000×12000≈4.6GB 的 O(N²))

返回 FusionOutput:
    tokens       (B,12000,d)  融合后全量 tokens
    image_tokens (B,8400,d)   融合后图像 tokens (供检测 head 或再融合)
    point_tokens (B,3600,d)   融合后点云 tokens
    image_scales {'P3'..}     (B,d,H,W) 按 N_IMG_SPLIT 切回三尺度空间图
    pose_embed   (B,128)      位姿条件原始嵌入(透传, 供 FiLM/条件解码用)

接口先行: 将来换真模型(缺陷微调 YOLO / 训练好的 attrmap CNN)只动
ImageEncoder / PointCloudEncoder 权重, 本模块契约 shape 不变。

实现要点:
- Pre-LN 交叉注意力: norm -> q_proj/k_proj/v_proj -> 多头 reshape ->
  F.scaled_dot_product_attention(is_causal=False) -> out_proj -> 残差。
- FFN(4d, Pre-LN, 残差), dropout 只在训练生效(SDPA 的 dropout_p 同理)。
- K/V 用同一份条件 token(位姿), N_COND 小所以每头 k/v 长度就是 N_COND。
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from fusion.config import (  # noqa: E402
    D_MODEL, DROPOUT, FFN_HIDDEN, N_COND, N_HEADS, N_IMG, N_IMG_SPLIT,
    N_PC, NECK_HW, NECK_ORDER, NUM_MODALITIES, NUM_SCALES, POSE_EMB, POSE_CKPT,
)


@dataclass
class FusionOutput:
    """融合结果, 全部为 torch.Tensor(B, ...)。

    - tokens:       (B,N_TOKENS,d) 图像在前、点云在后, 与 embed 索引一致
    - image_tokens: (B,N_IMG,d)
    - point_tokens: (B,N_PC,d)
    - image_scales: {'P3':(B,d,80,80), 'P4':(B,d,40,40), 'P5':(B,d,20,20)}
    - pose_embed:   (B,128)
    """
    tokens: torch.Tensor
    image_tokens: torch.Tensor
    point_tokens: torch.Tensor
    image_scales: Dict[str, torch.Tensor] = field(default_factory=dict)
    pose_embed: torch.Tensor = None  # type: ignore[assignment]


class CrossModalFusion(nn.Module):
    """交叉注意力·位姿条件融合。Q=图像/点云 tokens, K/V=位姿条件 tokens。"""

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_cond=N_COND,
                 ffn_hidden=FFN_HIDDEN, dropout=DROPOUT, device=''):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} 必须能被 n_heads={n_heads} 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_cond = n_cond
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # ---- 条件投影: pose_emb(128) -> (B,N_COND,d) ----
        self.cond_proj = nn.Sequential(
            nn.Linear(POSE_EMB, d_model), nn.GELU(),
            nn.Linear(d_model, n_cond * d_model),
        )

        # ---- token 索引 embedding (模态 2 + 尺度 3) ----
        self.mod_emb = nn.Embedding(NUM_MODALITIES, d_model)   # 0=image 1=point
        self.scale_emb = nn.Embedding(NUM_SCALES, d_model)     # P3/P4/P5

        # ---- Pre-LN 交叉注意力 ----
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

        # ---- FFN (Pre-LN, 残差) ----
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model), nn.Dropout(dropout),
        )

        # ---- 预构建的索引 buffer: 每 token 的尺度/模态标签 ----
        self._build_index_buffers()

        self.to(self.device)

    def _build_index_buffers(self):
        """把 8400/3600 个 token 的 scale / modality 标签预构造成索引 buffer。
        随 .to() 搬到目标 device, forward 里一次 Embedding 查表即可。"""
        scale_idx = torch.cat([
            torch.full((N_IMG_SPLIT[i + 1] - N_IMG_SPLIT[i],), i, dtype=torch.long)
            for i in range(len(NECK_ORDER))
        ])                                     # (N_IMG,) 每图像 token 的尺度标签 0/1/2
        self.register_buffer('scale_idx', scale_idx)
        self.register_buffer('img_mod_idx', torch.zeros(N_IMG, dtype=torch.long))   # 0
        self.register_buffer('pc_mod_idx', torch.ones(N_PC, dtype=torch.long))      # 1

    # ---------------- 前向 ----------------
    def forward(self, image_tokens, point_tokens, pose_embed):
        """image_tokens(B,N_IMG,d) + point_tokens(B,N_PC,d) + pose_embed(B,128)
        -> FusionOutput。"""
        dev = next(self.parameters()).device
        image_tokens = image_tokens.to(dev)
        point_tokens = point_tokens.to(dev)
        pose_embed = pose_embed.to(dev)

        # ---- 1. 拼接 + 位置/模态 embedding ----
        img_emb = self.scale_emb(self.scale_idx) + self.mod_emb(self.img_mod_idx)  # (N_IMG,d)
        pc_emb = self.mod_emb(self.pc_mod_idx)                                     # (N_PC,d)
        q_in = torch.cat([image_tokens + img_emb, point_tokens + pc_emb], dim=1)   # (B,12000,d)

        # ---- 2. 条件 K/V: (B,128) -> (B,N_COND,d) ----
        cond = self.cond_proj(pose_embed).view(-1, self.n_cond, self.d_model)      # (B,2,d)

        # ---- 3. Pre-LN 多头交叉注意力 (K/V 来自位姿条件) ----
        B, N_q, _ = q_in.shape
        q = self.q_proj(self.norm_q(q_in)).view(B, N_q, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.norm_kv(cond)
        k = self.k_proj(kv).view(B, self.n_cond, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, self.n_cond, self.n_heads, self.head_dim).transpose(1, 2)
        # 注意力矩阵 (B,8,12000,2) —— 线性于 token 数
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        x = q_in + self.out_proj(attn)                                            # 残差

        # ---- 4. FFN (Pre-LN) ----
        x = x + self.ffn(self.norm2(x))                                           # (B,12000,d)

        # ---- 5. 拆回图像 / 点云 ----
        image_tokens_out, point_tokens_out = x.split([N_IMG, N_PC], dim=1)
        image_scales = {}
        for name, (a, b), (h, w) in zip(NECK_ORDER,
                                        zip(N_IMG_SPLIT[:-1], N_IMG_SPLIT[1:]),
                                        NECK_HW.values()):
            image_scales[name] = image_tokens_out[:, a:b].permute(0, 2, 1).view(
                B, self.d_model, h, w)

        return FusionOutput(
            tokens=x, image_tokens=image_tokens_out, point_tokens=point_tokens_out,
            image_scales=image_scales, pose_embed=pose_embed,
        )

    def __repr__(self):
        return (f"CrossModalFusion(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_cond={self.n_cond}, dropout={self.dropout}, device={self.device})")


class FusionPipeline(nn.Module):
    """接口先行的组合器: 图像 list + attrmap + pose_state -> FusionOutput。

    三段编码器在内部组合, 将来换真模型权重只改构造参数(config 路径), 本类不动。
    forward(image_bgr_list, attrmap, pose_state):
      - image_bgr_list: list[np.ndarray] BGR uint8 (1280×720)
      - attrmap:        np.ndarray (3,720,1280) float32 原始 attrmap
      - pose_state:     (12,) 或 (B,12) 位姿状态 (/pose_state 字段序)
    """

    def __init__(self, image_encoder=None, point_encoder=None, pose_encoder=None,
                 fusion=None, device=''):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 延迟 import, 避免 config 里路径不存在时本文件都 import 失败
        from fusion.image_encoder import ImageEncoder
        from fusion.pointcloud_encoder import PointCloudEncoder
        from data_gen.pose_encoder import PoseEncoder

        self.image_encoder = image_encoder or ImageEncoder(device=self.device)
        self.point_encoder = point_encoder or PointCloudEncoder(device=self.device)
        self.pose_encoder = pose_encoder or PoseEncoder(POSE_CKPT)
        self.fusion = fusion or CrossModalFusion(device=self.device)

    def forward(self, image_bgr_list, attrmap, pose_state):
        imgs = self.image_encoder.preprocess(image_bgr_list, device=self.device)
        img_out = self.image_encoder(imgs)
        pc_out = self.point_encoder(attrmap)
        pe = self.pose_encoder(pose_state)          # (B,128) 或 (128,)
        if pe.dim() == 1:
            pe = pe.unsqueeze(0)
        return self.fusion(img_out['tokens'], pc_out['tokens'], pe)

    def __repr__(self):
        return (f"FusionPipeline(image={type(self.image_encoder).__name__}, "
                f"point={type(self.point_encoder).__name__}, "
                f"pose={type(self.pose_encoder).__name__}, "
                f"fusion={type(self.fusion).__name__}, device={self.device})")


if __name__ == '__main__':
    fusion = CrossModalFusion()
    print(fusion)
    B = 2
    img_tokens = torch.randn(B, N_IMG, D_MODEL)
    pc_tokens = torch.randn(B, N_PC, D_MODEL)
    pose = torch.randn(B, POSE_EMB)
    out = fusion(img_tokens, pc_tokens, pose)
    print({k: (tuple(v.shape) if torch.is_tensor(v)
               else {kk: tuple(vv.shape) for kk, vv in v.items()})
           for k, v in out.__dict__.items()})
