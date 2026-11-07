"""
YOLO11 编码器 —— 从训练好的 YOLO11 检测模型里摘出特征提取器(去掉检测头)。

两种模式:
  backbone  纯骨干网络 (第 0~10 层, 到 SPPF+C2PSA), 输出 P3/P4/P5 三尺度特征
            strides 8/16/32, yolo11l 下通道数 512/512/512 (max_channels=512 上限)
  neck      骨干+颈 (第 0~22 层, 检测头之前), 输出 PAN 融合后的 P3'/P4'/P5'
            通道数 256/512/512 (yolo11l)

输入约定: RGB float 张量, 范围 [0,1] (★权重在 [0,1] 归一化下训练★), 形状 (1,3,H,W),
H/W 为 32 的整数倍。本模块 forward 不做归一化, 请由调用方先 letterbox 并 /255 到 [0,1]。
(直接喂 [0,255] 会使特征值爆炸 ~25x。) 输出为三尺度特征张量列表,
可用于下游 2D 特征提取/融合。

构建:  build_encoder("best.pt", mode="backbone")
加载:  import yolo11_encoder; enc = torch.load("yolo11l_encoder.pt", weights_only=False)
"""
from __future__ import annotations

import torch
import torch.nn as nn

# YOLO11 (8.4.x) 实际层布局(24层): 0 Conv,1 Conv,2 C3k2,3 Conv,4 C3k2,5 Conv,6 C3k2,
# 7 Conv,8 C3k2,9 SPPF,10 C2PSA,11-22 颈,23 Detect。检测头消费的 P3/P4/P5:
# 骨干三尺度输出: P3(stride8)=第4层, P4(stride16)=第6层, P5(stride32)=第10层(C2PSA, 非SPPF)
_BACKBONE_OUT = [4, 6, 10]
# 骨干+颈三尺度输出(检测头 [16,19,22] 之前的融合特征): P3'=16, P4'=19, P5'=22
_NECK_OUT = [16, 19, 22]
# 切片终点(不含): backbone 到第11层(含C2PSA), neck 到第23层(不含Detect)
_SLICE_END = {"backbone": 11, "neck": 23}


class YOLO11Encoder(nn.Module):
    """YOLO11 特征提取器(无检测头)。前向返回三尺度特征张量列表。"""

    def __init__(self, layers: nn.Sequential, output_idxs: list[int], imgsz: int):
        super().__init__()
        self.layers = layers
        self.output_idxs = output_idxs
        self.imgsz = imgsz
        # 每尺度输出的 stride = imgsz / 特征图边长, 由 forward 后按需填充
        self.out_strides: list[int] = []

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B,3,H,W) RGB [0,1]; 返回 [P3,P4,P5] (与 strides 对应)。"""
        y: list = []
        for m in self.layers:
            f = m.f  # ultralytics 每个模块带 .f (from 索引, 默认 -1)
            if f != -1:
                x = (y[f] if isinstance(f, int)
                     else [x if j == -1 else y[j] for j in f])
            x = m(x)
            y.append(x)
        outs = [y[i] for i in self.output_idxs]
        if not self.out_strides:
            self.out_strides = [self.imgsz // o.shape[-1] for o in outs]
        return outs


def build_encoder(weights_path: str, mode: str = "backbone",
                  imgsz: int = 512, device: str = "cpu") -> YOLO11Encoder:
    """从 YOLO11 检测权重里摘出编码器(去掉检测头)。weights_path 可为预训练或训练好的 best.pt。"""
    if mode not in _SLICE_END:
        raise ValueError(f"mode 只能是 backbone/neck, 收到 {mode!r}")
    from ultralytics import YOLO
    model = YOLO(weights_path)
    m = model.model  # DetectionModel
    # m.model 是 nn.Sequential; 切片保留原层索引, 保证 head 里 Concat 的 .f 引用仍有效
    layers = m.model[:_SLICE_END[mode]]
    output_idxs = list(_BACKBONE_OUT if mode == "backbone" else _NECK_OUT)
    enc = YOLO11Encoder(layers, output_idxs, imgsz)
    enc.eval().to(device)
    return enc


def describe_encoder(enc: YOLO11Encoder) -> str:
    """打印/返回编码器各输出层的类型、通道、stride。"""
    lines = [f"编码器: {len(enc.layers)} 层, {enc.output_idxs} 为输出层"]
    lines.append(f"输入: (1,3,{enc.imgsz},{enc.imgsz}) RGB[0,255], H/W 需为 32 倍数")
    for i, idx in enumerate(enc.output_idxs):
        lay = enc.layers[idx]
        lines.append(f"  输出[{i}] 层{idx} ({type(lay).__name__}) "
                     f"stride={enc.out_strides[i]} (imgsz {enc.imgsz} 时)")
    return "\n".join(lines)


def load_encoder(path: str, map_location="cpu") -> YOLO11Encoder:
    """加载之前保存的编码器 .pt。需要本模块已 import (pickle 反序列化类引用)。"""
    return torch.load(path, map_location=map_location, weights_only=False)
