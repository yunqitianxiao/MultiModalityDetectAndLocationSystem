"""
融合子系统 —— 冻结契约常量
================================================================
接口先行策略: 以下 shape 一旦冻结, 将来用真模型(缺陷微调 YOLO / 训练好的
attrmap CNN)替换 ImageEncoder / PointCloudEncoder 的权重或实现时, 融合
模块 (CrossModalFusion) 只认这些 shape, 代码不用改。

数字来源 (已实测确认):
- yolov11l neck P3/P4/P5 = 256 / 512 / 512 通道 (P5 被 max_channels=512 截断)
- 640×640 输入下特征图 80² / 40² / 20² -> N_img = 8400 tokens
- attrmap: .npz 三 key (depth/reflec/height), 各 720×1280 float32
- 位姿: 12 维 /pose_state -> PoseEncoder -> 128 维嵌入
- 融合: d_model=256, heads=8, 位姿条件 token 数 N_COND=2, 点云 stride 16 -> 3600 tokens
"""

import os

import numpy as np

# ---------------- 路径 ----------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO11L_CKPT = os.environ.get(
    "YOLO11L_CKPT",
    os.path.abspath(os.path.join(REPO_ROOT, "..", "weights", "yolo11l.pt")),
)                                                           # 官方 COCO 预训练/缺陷微调权重
POSE_CKPT = os.environ.get(
    "POSE_CKPT",
    os.path.join(REPO_ROOT, "weights", "pose_mlp_tunnel.pt"),
)                                                           # 训练好的位姿编码器
OUTPUT_V7    = os.path.join(REPO_ROOT, "data_gen", "output_v7")  # 配对数据(测试样本)

# ---------------- 图像 / YOLO ----------------
IMG_W, IMG_H = 1280, 720
IMG_CH = 3
YOLO_SIZE = 640                    # ultralytics letterbox 目标尺寸
NECK_ORDER = ("P3", "P4", "P5")
NECK_CH = {"P3": 256, "P4": 512, "P5": 512}      # P5 被 max_channels=512 截断, 非 1024
NECK_HW = {"P3": (80, 80), "P4": (40, 40), "P5": (20, 20)}
N_IMG_PER_SCALE = {"P3": 80 * 80, "P4": 40 * 40, "P5": 20 * 20}
N_IMG = 6400 + 1600 + 400          # 8400, 全拉平 token 数
N_IMG_SPLIT = (0, 6400, 8000, 8400)   # tokens 切回三尺度图的分界 (P3,P4,P5)

# ---------------- attrmap 点云 ----------------
ATTR_W, ATTR_H = 1280, 720
ATTR_CH = 3
ATTR_ORDER = ("depth", "reflec", "height")
# 归一化常量 (当前 v7 数值范围校准; 真训练前按全集统计重新校准)
DEPTH_SCALE   = 30.0                # depth∈[1.17,28.2]m -> /30 夹到 [0,1]
HEIGHT_OFFSET = 0.0                 # height∈[-0.2,2.57]m -> (h-0)/3 夹到 [0,1]
HEIGHT_SCALE  = 3.0
PC_STRIDE = 16                      # attrmap CNN 下采样倍数
PC_H, PC_W = ATTR_H // PC_STRIDE, ATTR_W // PC_STRIDE   # (45, 80)
N_PC = PC_H * PC_W                  # 3600

# ---------------- 位姿 ----------------
POSE_DIM = 12                       # /pose_state 维度
POSE_EMB = 128                      # PoseEncoder 嵌入维度
POS_SCALE = 10.0                    # /pose_state 的位置归一化因子

# ---------------- 融合 ----------------
D_MODEL = 256
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS        # 32
FFN_HIDDEN = D_MODEL * 4             # 1024
N_COND = 2                           # 位姿条件 token 数 (K/V 源)
NUM_MODALITIES = 2                   # image / point
NUM_SCALES = 3                       # P3/P4/P5
N_TOKENS = N_IMG + N_PC              # 12000
DROPOUT = 0.1

# ---------------- 点云辅助 2D 检测 (PointCloudAssistedYOLO) ----------------
NC = 7                               # output_v7 类别数 (data.yaml)
DETECT_STRIDE = (8.0, 16.0, 32.0)    # P3/P4/P5 对应检测头 stride
REG_MAX = 16                         # yolo11l DFL 通道数
HYP = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}   # 检测损失增益 (ultralytics 默认)


def normalize_attrmap(m: np.ndarray) -> np.ndarray:
    """attrmap (3,720,1280) float32 原始值 -> 归一化 [0,1] 三通道。
    与 PointCloudEncoder._normalize 一致(纯 numpy 版, 供数据装载用)。
    """
    depth, reflec, height = m[0], m[1], m[2]
    out = np.stack([
        np.clip(depth / DEPTH_SCALE, 0.0, 1.0),
        np.clip(reflec, 0.0, 1.0),
        np.clip((height - HEIGHT_OFFSET) / HEIGHT_SCALE, 0.0, 1.0),
    ]).astype(np.float32)
    return out
