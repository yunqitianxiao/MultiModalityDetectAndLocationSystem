"""
位姿状态 MLP 编码器 (共享定义)
================================================================
/pose_state (12 维) -> 3 层 MLP -> 128 维嵌入。

被两个脚本共用:
    scripts/pose_encoder_node.py   在线推理 (订阅 /pose_state -> 发 /pose_embedding)
    scripts/train_pose_encoder.py  训练 (CSV / 合成数据 -> 存 ckpt)

输入 12 维约定 (与 pose_state_node.py 输出一致):
    [x, y, z, sin(roll), cos(roll), sin(pitch), cos(pitch), sin(yaw), cos(yaw), vx, vy, wz]
位置为原始单位(米), 角度已做 sin/cos 编码。各通道量纲不同(米 vs 0~1 vs m/s),
不能用 LayerNorm(逐样本跨通道归一化, 会被位置量纲主导), 改用 BatchNorm1d
逐通道按训练集统计归一化; running stats 随 ckpt 保存, 推理喂原始值即可。
"""
import torch
import torch.nn as nn

DIM_IN = 12     # /pose_state 维度
DIM_HID = 64
DIM_EMB = 128   # 嵌入维度; 以后和图像特征对齐时按需调整


class PoseMLP(nn.Module):
    def __init__(self, dim_in=DIM_IN, dim_hid=DIM_HID, dim_emb=DIM_EMB):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_emb = dim_emb
        self.net = nn.Sequential(
            # 逐通道归一化: 各通道量纲差异大(米/0~1/速度), 训练集统计随 ckpt 保存
            nn.BatchNorm1d(dim_in),
            nn.Linear(dim_in, dim_hid), nn.GELU(),
            nn.Linear(dim_hid, dim_hid), nn.GELU(),
            nn.Linear(dim_hid, dim_emb),
        )

    def forward(self, x):
        return self.net(x)


def save_ckpt(model, path):
    """保存权重 + 结构参数, 供 load_ckpt 恢复。"""
    torch.save({
        'dim_in': model.dim_in,
        'dim_hid': model.dim_hid,
        'dim_emb': model.dim_emb,
        'state_dict': model.state_dict(),
    }, path)


def load_ckpt(path, map_location='cpu'):
    """从 save_ckpt 产出的文件恢复模型(推理用)。"""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = PoseMLP(ckpt.get('dim_in', DIM_IN),
                    ckpt.get('dim_hid', DIM_HID),
                    ckpt.get('dim_emb', DIM_EMB))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model
