"""CrossModalFusion：图像/点云 token 拼接后，用点云 token 的池化结果做 K/V 的
Pre-LN 交叉注意力 + FFN。

     Q   = concat[image_tokens(8400), point_tokens(3600)] + 模态/尺度 embedding
     K/V = [avgpool, maxpool](point_tokens -> 8×16) => 256 个 token

K/V 必须由 point_tokens 算出，否则图像 token 的输出与点云无关、点云支等于死支。
avg 保留整体分布、max 保留局部尖峰（扣件立牌这类稀疏几何差异在 avg 下会被稀释）。

注意力矩阵是 (B,8,12000,256)，代价对 token 数近似线性；换成 12000² 自注意力
需要约 5.8 GB，这里刻意避开。

token 数 12000 × FFN 1024 在 batch 8 下光 FFN 激活就 1.3 GB，训练时按 token
分块并做梯度检查点（反向重算）：FFN 逐 token 独立，所以分块在数学上等价，
只是换内存。评测（no_grad）走整批。
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import config as C


@dataclass
class FusionOutput:
    tokens: torch.Tensor
    image_tokens: torch.Tensor
    point_tokens: torch.Tensor
    image_scales: dict = field(default_factory=dict)


class CrossModalFusion(nn.Module):
    def __init__(self, d_model=C.D_MODEL, n_heads=C.N_HEADS,
                 ffn_hidden=C.FFN_HIDDEN, dropout=C.DROPOUT, chunk=2048, device=''):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.chunk = chunk
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # K/V 源：点云 token 的空间池化（无参数，形状固定）
        self.pc_kv_avg = nn.AdaptiveAvgPool2d(C.PC_KV_HW)
        self.pc_kv_max = nn.AdaptiveMaxPool2d(C.PC_KV_HW)

        self.mod_emb = nn.Embedding(C.NUM_MODALITIES, d_model)
        self.scale_emb = nn.Embedding(C.NUM_SCALES, d_model)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model), nn.Dropout(dropout))

        scale_idx = torch.cat([
            torch.full((C.N_IMG_SPLIT[i + 1] - C.N_IMG_SPLIT[i],), i, dtype=torch.long)
            for i in range(len(C.NECK_ORDER))])
        self.register_buffer('scale_idx', scale_idx)
        self.register_buffer('img_mod_idx', torch.zeros(C.N_IMG, dtype=torch.long))
        self.register_buffer('pc_mod_idx', torch.ones(C.N_PC, dtype=torch.long))

        self.to(self.device)

    def point_kv(self, point_tokens):
        """(B,3600,d) -> (B,256,d)：avg/max 双池化在 token 维拼接。"""
        B = point_tokens.shape[0]
        m = point_tokens.transpose(1, 2).reshape(B, self.d_model, C.PC_H, C.PC_W)
        a = self.pc_kv_avg(m).flatten(2)
        b = self.pc_kv_max(m).flatten(2)
        return torch.cat([a, b], dim=2).transpose(1, 2)

    def forward(self, image_tokens, point_tokens):
        dev = next(self.parameters()).device
        image_tokens = image_tokens.to(dev)
        point_tokens = point_tokens.to(dev)

        img_emb = self.scale_emb(self.scale_idx) + self.mod_emb(self.img_mod_idx)
        pc_emb = self.mod_emb(self.pc_mod_idx)
        q_in = torch.cat([image_tokens + img_emb, point_tokens + pc_emb], dim=1)

        B, N_q, _ = q_in.shape
        kv = self.norm_kv(self.point_kv(point_tokens))
        n_kv = kv.shape[1]
        k = self.k_proj(kv).view(B, n_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, n_kv, self.n_heads, self.head_dim).transpose(1, 2)

        step = self.chunk if (self.training and N_q > self.chunk) else N_q
        use_ckpt = self.training and step < N_q
        parts = []
        for s in range(0, N_q, step):
            blk = q_in[:, s:s + step]
            if use_ckpt:
                parts.append(checkpoint(self._block, blk, k, v, use_reentrant=False))
            else:
                parts.append(self._block(blk, k, v))
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

        img_out, pc_out = x.split([C.N_IMG, C.N_PC], dim=1)
        return FusionOutput(tokens=x, image_tokens=img_out, point_tokens=pc_out,
                            image_scales=split_scales(img_out))

    def _block(self, blk, k, v):
        B, N, _ = blk.shape
        q = self.q_proj(self.norm_q(blk)).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(B, N, self.d_model)
        x = blk + self.out_proj(attn)
        return x + self.ffn(self.norm2(x))

    def __repr__(self):
        return (f'CrossModalFusion(d_model={self.d_model}, n_heads={self.n_heads}, '
                f'kv={2 * C.PC_KV_HW[0] * C.PC_KV_HW[1]}@point-tokens, '
                f'dropout={self.dropout})')


def split_scales(image_tokens):
    """(B,8400,d) -> {'P3':(B,d,80,80),'P4':(B,d,40,40),'P5':(B,d,20,20)}"""
    B, _, d = image_tokens.shape
    out = {}
    for name, (a, b), (h, w) in zip(C.NECK_ORDER, zip(C.N_IMG_SPLIT[:-1], C.N_IMG_SPLIT[1:]),
                                    C.NECK_HW.values()):
        out[name] = image_tokens[:, a:b].permute(0, 2, 1).view(B, d, h, w)
    return out
