#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D 检测编码器运行模块 —— 预处理器(letterbox/对齐) + 三个分支编码器统一调用
================================================================================
把三个分支(土建/设备/轨道)的检测编码器(已从 YOLO11l 摘出, 去掉检测头)统一起来,
供 ④ 跨模态融合层使用。输入一张原图 -> 输出三个分支各自的 P3/P4/P5 特征金字塔。

【核心规范 —— 对齐与输入】
  1) 输入图像统一 letterbox 到 1024×1024 (32 倍数), 不缩放变形, 灰度填充 114。
  2) 归一化到 [0,1]: RGB float tensor。★权重是在 [0,1] 下训练的★
     - yolo11_encoder.forward 不做归一化, 所以必须在【这里】统一 /255。
     - 直接喂 [0,255] 会让特征值爆炸 ~25x (conv 权重由 [0,1] 输入训练而来)。
  3) 默认 neck 编码器(骨干+颈, P3'/P4'/P5', 检测头前融合特征), 最适合跨模态融合;
     也可切 backbone(纯骨干 P3/P4/P5)。

【输出】
  encode_all(...) -> {
      '轨道': {'neck': [P3,P4,P5], 'backbone': [...]},
      '土建': {...}, '设备': {...},
  }
  各尺度张量 (B, C, H/stride, W/stride), 三分支在相同输入下输出同尺寸特征图。

【注意】加载编码器 .pt 前必须 import yolo11_encoder(反序列化需类定义);
        分支来源/类别信息在 enc.branch / enc.names(生成时已写入)。

用法:
    import det_encoders_runtime as R
    encs = R.load_all_encoders()
    out = R.encode_all(image, encs, imgsz=1024)   # image 可为 BGR ndarray / PIL / 路径str
    feats = out['轨道']['neck']
"""
from __future__ import annotations

import os
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image

# 让 .pt 反序列化能找到类定义
try:
    import yolo11_encoder  # noqa: F401
except ImportError:
    _ENC_DIR = os.path.dirname(os.path.abspath(__file__))
    if _ENC_DIR not in os.sys.path:
        os.sys.path.insert(0, _ENC_DIR)
    import yolo11_encoder  # noqa: F401

_P = os.path.dirname(os.path.abspath(__file__))


def _find_repo():
    """向上找仓库根: 第一个含 data1 / guidao_best.pt / data(已知数据集) 的目录。
    本地用 data1, 服务器用 data/设备_异物支架_合并_yolo 作根标志。"""
    d = _P
    srv_marker = os.path.join("data", "设备_异物支架_合并_yolo", "data.yaml")
    for _ in range(6):
        if (os.path.isdir(os.path.join(d, "data1"))
                or os.path.exists(os.path.join(d, "guidao_best.pt"))
                or os.path.exists(os.path.join(d, srv_marker))):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return _P


_REPO = _find_repo()

# 三个分支: (中文名, neck编码器相对路径, backbone编码器相对路径)
BRANCHES = [
    ("轨道",
     "data1/轨道固件_4cls_stage/yolo11l_轨道3cls_neck_encoder.pt",
     "data1/轨道固件_4cls_stage/yolo11l_轨道3cls_backbone_encoder.pt"),
    ("土建",
     "data1/土建_3cls_stage/yolo11l_土建3cls_neck_encoder.pt",
     "data1/土建_3cls_stage/yolo11l_土建3cls_backbone_encoder.pt"),
    ("设备",
     "data1/设备_2cls_stage/yolo11l_设备2cls_neck_encoder.pt",
     "data1/设备_2cls_stage/yolo11l_设备2cls_backbone_encoder.pt"),
]

_INTER_AREA = cv2.INTER_AREA
_INTER_LINEAR = cv2.INTER_LINEAR


def letterbox_to(img: np.ndarray, size: int = 1024, fill: int = 114) -> np.ndarray:
    """HWC ndarray -> letterbox 到 size×size, 不缩放变形, 灰填充。size 上取整到 32 倍数。"""
    size = (int(size) + 31) // 32 * 32
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = round(h * s), round(w * s)
    interp = _INTER_AREA if s < 1 else _INTER_LINEAR
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=interp)
    padded = np.full((size, size, img.shape[2]), fill, dtype=img.dtype)
    padded[(size - nh) // 2:(size - nh) // 2 + nh,
           (size - nw) // 2:(size - nw) // 2 + nw] = img
    return padded


def preprocess(img: Union[str, "os.PathLike", np.ndarray, Image.Image], imgsz: int = 1024) -> torch.Tensor:
    """统一预处理: 任意输入 -> (1,3,H,W) RGB float [0,1], H=W=imgsz(32倍数)。
    默认按 RGB 输入处理; 若是 OpenCV 读的 BGR, 先 read_rgb() 转一下。"""
    if isinstance(img, (str, os.PathLike)):
        img = np.array(Image.open(os.fspath(img)).convert("RGB"))
    elif isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    else:
        img = np.asarray(img)
        if img.ndim == 2:                       # 灰度 -> RGB
            img = np.stack([img] * 3, axis=-1)
    img = letterbox_to(img, imgsz)
    x = torch.from_numpy(np.ascontiguousarray(img).astype(np.float32)) / 255.0
    return x.permute(2, 0, 1).unsqueeze(0)      # (1,3,H,W) [0,1]


def read_rgb(path: Union[str, "os.PathLike"]) -> np.ndarray:
    """用 OpenCV 读图 -> RGB HWC uint8(处理 BGR 习惯)。"""
    bgr = cv2.imread(os.fspath(path))
    return bgr[..., ::-1] if bgr is not None else None


def _resolve_enc(repo: str, rel: str) -> str | None:
    """定位编码器文件: 本地在 data1/<stage>/ 下, 服务器把编码器平铺在根目录。
    按  repo/rel → repo/<basename> → repo/data1/<basename>  依次找, 返回存在路径或 None。"""
    cands = (os.path.join(repo, rel),
             os.path.join(repo, os.path.basename(rel)),
             os.path.join(repo, "data1", os.path.basename(rel)))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def load_all_encoders(repo: str = _REPO, modes: tuple = ("neck", "backbone")) -> dict:
    """加载三个分支编码器 -> {'轨道': {'neck': enc, 'backbone': enc}, ...}。缺则跳过该模式。"""
    encs = {}
    for branch, neck_p, bb_p in BRANCHES:
        d = {}
        for mode in modes:
            rel = neck_p if mode == "neck" else bb_p
            path = _resolve_enc(repo, rel)
            if path is None:
                print(f"[跳过] {branch}/{mode} 编码器不存在: {os.path.join(repo, rel)}")
                continue
            enc = torch.load(path, map_location="cpu", weights_only=False)
            enc.eval()
            d[mode] = enc
        if d:
            encs[branch] = d
    return encs


@torch.no_grad()
def encode_all(img: Union[str, "os.PathLike", np.ndarray, Image.Image], encs: dict,
               imgsz: int = 1024, modes: tuple = ("neck",)) -> dict:
    """一张原图 -> 三分支特征 {branch: {mode: [P3,P4,P5]}}。
    modes 默认 ('neck',) —— 跨模态融合用 neck; 其它模式需预先 add/backbone 分支存在。"""
    x = preprocess(img, imgsz)
    out = {}
    for branch, md in encs.items():
        out[branch] = {}
        for mode in modes:
            enc = md.get(mode)
            if enc is None:
                continue
            out[branch][mode] = enc(x)
    return out


def describe(out: dict) -> str:
    lines = []
    for branch, md in out.items():
        for mode, feats in md.items():
            strs = "  ".join(f"P{i+3}{tuple(f.shape)}" for i, f in enumerate(feats))
            lines.append(f"  [{branch}/{mode}] {strs}")
    return "\n".join(lines)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    print("加载三分支编码器 ...")
    encs = load_all_encoders()
    print(f"  已加载分支: {list(encs.keys())}")
    for b, md in encs.items():
        print(f"    {b}: {list(md.keys())}")

    img = None
    for pat in ("data1/轨道_3cls_均衡/images/*/*.png", "data1/轨道_3cls/images/*/*.jpg",
                "data1/土建_病害_yolo_3cls/images/*/*.jpg",
                "data1/设备_异物支架_合并_yolo/images/*/*.jpg"):
        found = sorted(__import__("glob").glob(os.path.join(_REPO, pat)))
        if found:
            img = found[0]
            break
    if not img:
        img = sorted(__import__("glob").glob(os.path.join(_REPO, "data1/**/*.jpg"), recursive=True))[0]
    print(f"\n测试图: {img}")
    out = encode_all(img, encs, imgsz=1024, modes=("neck", "backbone"))
    print("特征(输入统一 letterbox 到 1024×1024 [0,1]):")
    print(describe(out))

    print("\n对齐校验(neck 三分支同尺度):")
    nm = {b: out[b]["neck"] for b in out}
    for i in range(3):
        shp = {b: tuple(nm[b][i].shape) for b in nm}
        same = len(set(shp.values())) == 1
        print(f"  P{i+3}: {shp}  {'✓ 一致' if same else '✗ 不一致(融合前需 resize)'}")
