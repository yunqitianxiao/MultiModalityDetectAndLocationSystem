"""冻结契约常量。形状一旦定下，换编码器权重/实现时融合模块不用改。

数字来源（实测确认）：
- YOLO11l neck P3/P4/P5 = 256 / 512 / 512 通道（P5 被 max_channels=512 截断）
- 640×640 输入下特征图 80²/40²/20² → N_IMG = 8400 tokens
- attrmap .npz 三通道 depth/reflec/height，各 720×1280
- 点云 CNN 4 次 stride-2 → 45×80 → 3600 tokens
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(env_key, default):
    return os.path.abspath(os.path.expanduser(os.environ.get(env_key) or default))


LOCAL_DATA = os.path.expanduser(
    '~/workspace/inspection_robot/data_gen/output_crack_big')
LOCAL_YOLO = os.path.expanduser('~/workspace/pro/2/yolo11l.pt')

# 项目内 data/weights 优先（服务器上的摆放），否则回落到本地原始路径
_PROJ_DATA = os.path.join(ROOT, 'data', 'output_crack_big')
_PROJ_YOLO = os.path.join(ROOT, 'weights', 'yolo11l.pt')
DATA_ROOT = _path('TUNNEL_DATA', _PROJ_DATA if os.path.isdir(_PROJ_DATA) else LOCAL_DATA)
YOLO_CKPT = _path('TUNNEL_YOLO', _PROJ_YOLO if os.path.exists(_PROJ_YOLO) else LOCAL_YOLO)

# ---------------- 图像 / YOLO ----------------
IMG_W, IMG_H = 1280, 720
YOLO_SIZE = 640
NECK_ORDER = ('P3', 'P4', 'P5')
NECK_CH = {'P3': 256, 'P4': 512, 'P5': 512}
NECK_HW = {'P3': (80, 80), 'P4': (40, 40), 'P5': (20, 20)}
N_IMG = 6400 + 1600 + 400            # 8400
N_IMG_SPLIT = (0, 6400, 8000, 8400)
DETECT_STRIDE = (8.0, 16.0, 32.0)
REG_MAX = 16
NC = 1                               # 单类：crack
HYP = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}

# ---------------- attrmap 点云 ----------------
ATTR_W, ATTR_H = 1280, 720
ATTR_CH = 3
ATTR_ORDER = ('depth', 'reflec', 'height')
# reflec 通道 = gpu_lidar intensity = 世界生成时按类写的 laser_retro，未设
# DEFECT_RETRO_ZERO=1 的数据集里它是缺陷掩膜（框内饱和、墙=0）= 标签泄漏。
# TUNNEL_ZERO_REFLEC=1 时清零，只留 depth/height 纯几何。
ZERO_REFLEC = os.environ.get('TUNNEL_ZERO_REFLEC') == '1'
DEPTH_SCALE = 30.0                   # depth ∈ [1.17, 19.6] m
HEIGHT_OFFSET = 0.0
HEIGHT_SCALE = 3.0                   # height ∈ [-0.21, 2.57] m
PC_STRIDE = 16
PC_H, PC_W = ATTR_H // PC_STRIDE, ATTR_W // PC_STRIDE   # (45, 80)
N_PC = PC_H * PC_W                   # 3600

# ---------------- 融合 ----------------
D_MODEL = 256
N_HEADS = 8
FFN_HIDDEN = D_MODEL * 4
PC_KV_HW = (8, 16)                   # 点云 token 池化到的网格，2 份(avg/max) => 256 个 K/V
NUM_MODALITIES = 2
NUM_SCALES = 3
DROPOUT = 0.1
N_TOKENS = N_IMG + N_PC              # 12000


def normalize_attrmap(m):
    """(3,720,1280) 原始 attrmap -> [0,1] 归一化（numpy 版，与 encoder 内一致）。"""
    import numpy as np
    return np.stack([
        np.clip(m[0] / DEPTH_SCALE, 0.0, 1.0),
        np.clip(m[1], 0.0, 1.0),
        np.clip((m[2] - HEIGHT_OFFSET) / HEIGHT_SCALE, 0.0, 1.0),
    ]).astype(np.float32)
