"""
多模态特征融合子系统 —— 包入口
================================================================
图像 (YOLO11l) + 点云 (attrmap CNN) + 位姿 (PoseEncoder) 的交叉注意力融合。
接口先行: 现在用真实 backbone / 冻结编码器打通链路, 将来换真模型(缺陷微调
YOLO / 训练好的 attrmap CNN)权重, 契约 shape 不变, 融合代码不动。

用法:
    from fusion import ImageEncoder, PointCloudEncoder, CrossModalFusion, FusionPipeline
    pipe = FusionPipeline()
    out  = pipe(image_bgr_list, attrmap, pose_state)   # -> FusionOutput
    out.tokens.shape      # (B,12000,256)
    out.image_scales['P3']  # (B,256,80,80)
"""

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DIR)
for _P in (_REPO_ROOT, _DIR):
    if _P not in sys.path:
        sys.path.insert(0, _P)

from fusion.config import *                # noqa: F401,F403  常量 (D_MODEL, N_TOKENS, ...)
from fusion import config as CONFIG        # noqa: F401  模块本体(含 normalize_attrmap)
from fusion.image_encoder import ImageEncoder            # noqa: F401
from fusion.pointcloud_encoder import PointCloudEncoder  # noqa: F401
from fusion.fusion import (CrossModalFusion, FusionOutput, FusionPipeline)  # noqa: F401
