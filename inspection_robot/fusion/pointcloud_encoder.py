"""
PointCloudEncoder —— attrmap CNN 编码器 (接口先行)
================================================================
点云经定位链投影成 depth/reflec/height 三通道 attrmap (720×1280), 本项目
用它做点云编码器: 输入 (B,3,720,1280) 原始值 -> 归一化(模块内) -> CNN ->
特征图 (B,256,45,80) + 拉平 tokens (B,3600,256)。

将来 attrmap CNN 训练完后, 换权重即可(契约 shape 不变, 融合模块不动)。

归一化常量 (depth/30, height/3, clamp) 覆盖当前 v7 数据范围, 注册为 buffer
(随 state_dict 存, 不可训练); 真训练前按全集统计重新校准 config 常量。
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from fusion.config import (  # noqa: E402
    ATTR_CH, ATTR_H, ATTR_W, D_MODEL, DEPTH_SCALE, HEIGHT_OFFSET, HEIGHT_SCALE, PC_H, PC_W,
)


class PointCloudEncoder(nn.Module):
    """attrmap CNN 编码器: (B,3,720,1280) 原始 attrmap -> feat(B,256,45,80) + tokens(B,3600,256)。"""

    def __init__(self, in_ch=ATTR_CH, out_ch=D_MODEL, normalize=True, device=''):
        super().__init__()
        self.out_ch = out_ch
        self.normalize = normalize
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # ---- 归一化常量(随 state_dict 存) ----
        self.register_buffer('depth_scale', torch.tensor(DEPTH_SCALE, dtype=torch.float32))
        self.register_buffer('height_offset', torch.tensor(HEIGHT_OFFSET, dtype=torch.float32))
        self.register_buffer('height_scale', torch.tensor(HEIGHT_SCALE, dtype=torch.float32))

        # ---- 4 个 stride-2 卷积: 3 -> 32 -> 64 -> 128 -> out_ch, 720×1280 -> 45×80 ----
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.BatchNorm2d(c_out), nn.GELU(),
            )
        self.net = nn.Sequential(
            block(in_ch, 32),
            block(32, 64),
            block(64, 128),
            block(128, out_ch),
        )
        self.to(self.device)

    def _normalize(self, x):
        """depth/30, reflec 原样, (height-0)/3; 各 clamp 到 [0,1]。"""
        x = x.to(self.depth_scale.device)     # 防御: 与归一化 buffer 同 device
        d = torch.clamp(x[:, 0:1] / self.depth_scale, 0.0, 1.0)
        r = torch.clamp(x[:, 1:2], 0.0, 1.0)
        h = torch.clamp((x[:, 2:3] - self.height_offset) / self.height_scale, 0.0, 1.0)
        return torch.cat([d, r, h], dim=1)

    def forward(self, attr):
        """attr: (B,3,720,1280) float32 原始 attrmap (numpy 或 tensor)。
        返回 dict: {'feat': (B,out_ch,45,80), 'tokens': (B,3600,out_ch)}。"""
        if not isinstance(attr, torch.Tensor):
            attr = torch.from_numpy(np.asarray(attr, dtype=np.float32))
        if attr.dim() == 3:                       # (3,720,1280) 单样本 -> (1,3,720,1280)
            attr = attr.unsqueeze(0)
        x = attr.to(self.device)
        if self.normalize:
            x = self._normalize(x)
        feat = self.net(x)                                # (B,out_ch,45,80)
        tokens = feat.flatten(2).transpose(1, 2)          # (B,3600,out_ch)
        return {'feat': feat, 'tokens': tokens}

    def __repr__(self):
        return f"PointCloudEncoder(out_ch={self.out_ch}, normalize={self.normalize}, device={self.device})"


if __name__ == '__main__':
    enc = PointCloudEncoder()
    print(enc)
    dummy = torch.zeros(1, ATTR_CH, ATTR_H, ATTR_W)
    out = enc(dummy)
    print({k: tuple(v.shape) for k, v in out.items()})
    # 归一化生效检查
    x = torch.zeros(1, 3, ATTR_H, ATTR_W)
    x[:, 0] = 60.0   # depth 超范围
    x[:, 2] = -5.0   # height 超范围
    print('depth 归一化后 max:', float(enc._normalize(x)[:, 0].max()), '| height min:', float(enc._normalize(x)[:, 2].min()))
