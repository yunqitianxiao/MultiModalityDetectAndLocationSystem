"""
位姿编码器 —— 从训练脚手架中摘出的独立推理接口
================================================================
把 train_pose_encoder.py 训练好的 PoseMLP **编码器**封装成融合模块可直接
import 的独立接口：/pose_state (12 维) -> 128 维嵌入。权重冻结、eval 模式。

职责分离:
    - 训练:   scripts/train_pose_encoder.py  (自监督代理任务, 训练完即弃预测头)
    - 部署:   本模块 + data_gen/pose_mlp.py   (只加载编码器权重, 无预测头)
    - 融合:   后续 FiLM 训练 import PoseEncoder, 前向即得全局时序上下文, 无需碰训练脚本

用法:
    from data_gen.pose_encoder import PoseEncoder
    enc = PoseEncoder('/home/ylf/pose_mlp_tunnel.pt')   # 不给 ckpt 则随机初始化(仅打通链路)
    emb = enc.encode(state)      # state: (12,) -> (128,); 或 (B,12) -> (B,128)
    emb = enc(state)             # 同 encode, 可当 nn.Module 直接调用
"""
import os

import numpy as np
import torch

# 让脚本在源码树 / 安装目录两种方式下都能 import data_gen
_DIR = os.path.dirname(os.path.abspath(__file__))
for _P in (_DIR, os.path.dirname(_DIR)):
    if _P not in __import__('sys').path and os.path.isdir(os.path.join(_P, 'data_gen')):
        __import__('sys').path.insert(0, _P)
        break

from data_gen.pose_mlp import PoseMLP, load_ckpt  # noqa: E402


class PoseEncoder:
    """冻结的位姿编码器: 12 维状态 -> 128 维嵌入(供多模态融合全局上下文)。"""

    def __init__(self, ckpt: str = ''):
        if ckpt and os.path.exists(ckpt):
            self.model = load_ckpt(ckpt)
        else:
            self.model = PoseMLP()
            self.model.eval()
        self.dim_in = self.model.dim_in
        self.dim_emb = self.model.dim_emb

    @torch.no_grad()
    def encode(self, state):
        """state: 序列化数组 (12,) 或 (B,12) -> 嵌入 (128,) 或 (B,128)。"""
        x = state if isinstance(state, torch.Tensor) else torch.from_numpy(
            np.asarray(state, dtype=np.float32))
        if x.dtype != torch.float32:
            x = x.float()
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        emb = self.model(x)
        return emb.squeeze(0) if single else emb

    def __call__(self, state):
        return self.encode(state)
