"""
ImageEncoder —— YOLO11l 主干特征提取 (接口先行)
================================================================
把 ultralytics YOLO11l 的 neck 输出 P3/P4/P5 摘成融合可用的特征:
  输入 (B,3,640,640) RGB float[0,1]
  -> 主干(冻结) -> neck 特征 P3(256,80²) P4(512,40²) P5(512,20²)
  -> per-scale 1×1 投影到 d_model -> 空间图 + 拉平 tokens (B,8400,d)

将来缺陷微调 YOLO 训练完后, 换 `ckpt` 路径即可(权重替换, 契约 shape 不变,
融合模块代码不动)。官方 COCO 预训练权重现在就能给出真实 neck 特征。

实现要点:
- 权重是 fp16 存储, 加载后必须 .float().eval()。
- 用 Detect 层(模型最后一层)的 forward_pre_hook 捕获其输入 = [P3,P4,P5],
  不硬编码层索引(isinstance 定位, 抵御 ultralytics 版本变化)。
- 勿用 model.predict(embed=...) —— 它会把特征图 adaptive_avg_pool 成向量。
- freeze_backbone=True(默认): 主干 no_grad + requires_grad_(False),
  梯度只走 per-scale 投影层(no_grad 块之外)。解锁主干时注意 eval 下
  Detect 的 decode 路径可能有非可导算子, 需把 Detect 临时换 Identity。
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
    D_MODEL, IMG_H, IMG_W, NECK_CH, NECK_HW, NECK_ORDER, N_IMG_SPLIT, YOLO11L_CKPT, YOLO_SIZE,
)


class ImageEncoder(nn.Module):
    """YOLO11l 主干特征提取: (B,3,640,640) BGR float[0,1] -> {P3,P4,P5 空间图} + tokens(B,8400,d)。"""

    def __init__(self, ckpt=YOLO11L_CKPT, d_model=D_MODEL,
                 freeze_backbone=True, unfreeze_neck=0, device=''):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.unfreeze_neck = unfreeze_neck
        self.d_model = d_model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # ---- 加载 YOLO11l (fp16 -> fp32, eval) ----
        ckpt_data = torch.load(ckpt, map_location='cpu', weights_only=False)
        # 官方 COCO 权重把网络存在 'model'(此时 'ema' 为 None); ultralytics 训练产出的
        # best.pt 反过来: 'model' 置 None, 真正的权重在 'ema'(EMA 滑动平均)。两者都要兼容。
        model = ckpt_data.get('model')
        if model is None:
            model = ckpt_data.get('ema')
        assert model is not None, f"{ckpt} 里既无 'model' 也无 'ema' 权重"
        model = model.float().eval()
        self.model = model.to(self.device)

        # ---- 定位 Detect 层并挂 pre-hook 捕获 neck 输入 [P3,P4,P5] ----
        from ultralytics.nn.modules.head import Detect
        self._detect = next(m for m in model.model if isinstance(m, Detect))
        self._hook = self._detect.register_forward_pre_hook(self._capture)
        self._feats = {}
        self._identity_installed = False
        # 注: 我们把 Detect 的 pre-hook 输入(P3/P4/P5)当作前馈特征, Detect 自身输出不消费;
        #     即便解锁主干带梯度前向, Detect.forward 只是 conv+sigmoid 解码(可导), NMS 在
        #     postprocess 不进 Detect.forward, 故无需真换 Identity —— 与原生 yolo 训练一致。

        # ---- per-scale 投影: 256/512/512 -> d_model ----
        self.scale_proj = nn.ModuleDict({
            name: nn.Conv2d(NECK_CH[name], d_model, 1) for name in NECK_ORDER
        })

        # ---- 冻结/解锁: 全骨干冻结, 或解锁末端 neck 反传 ----
        if freeze_backbone and unfreeze_neck <= 0:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self._unfrozen = []
        else:
            # 解锁主干/neck 反传: 全置 requires_grad=False 后, 把末段 unfreeze_neck 个模块
            # (或全部)置为可训。Detect 保持原样, 只是它的输出我们不用, 沿 hook 输入的 neck
            # 特征反传即可(见上方注释)。
            for p in self.model.parameters():
                p.requires_grad_(False)
            self._unfrozen = []
            n_unfreeze = len(self.model.model) if unfreeze_neck <= 0 else unfreeze_neck
            for mi in range(len(self.model.model) - 1, max(len(self.model.model) - 1 - n_unfreeze, 0), -1):
                for p in self.model.model[mi].parameters():
                    p.requires_grad_(True)
                    self._unfrozen.append(p)

        self.to(self.device)          # 投影层也搬到目标 device

    def _capture(self, module, args):
        """pre-hook: args[0] 是 Detect 的输入 list [P3,P4,P5] (stride 8/16/32)。
        Detect 可能已被换成 Identity (解锁主干时), 这时 hook 不触发, 靠 _capture_identity。"""
        feats = args[0]
        self._feats = {name: f for name, f in zip(NECK_ORDER, feats)}

    def _capture_identity(self, module, args):
        """Identity 的 pre-hook: args[0] 即 Detect 原本的输入 list [P3,P4,P5]。"""
        feats = args[0]
        self._feats = {name: f for name, f in zip(NECK_ORDER, feats)}

    def _install_identity(self):
        """把模型末层 Detect 换成 Identity, 让前向直接返回 neck 特征 [P3,P4,P5]。

        原因(见模块 docstring): 解锁主干时需对 neck 反传, 但 Detect 的 decode 路径
        含非可导/就地算子, 且我们不消费它的输出(只取 pre-hook 的输入)。换成 Identity
        后模型输出的就是 [P3,P4,P5], hook 挂到 Identity 上同样能捕获。"""
        from ultralytics.nn.modules.head import Detect
        det = self.model.model[-1]
        if isinstance(det, Detect):
            f = getattr(det, 'f', -1)          # 记录 Detect 的 f (从哪些层组装输入 [P3,P4,P5])
            idx = len(self.model.model) - 1
            ident = nn.Identity().to(self.device)
            ident.f = f                        # 必须设 f, 否则 yolo Model.forward 访问 m.f 报错
            self.model.model[idx] = ident
            self._detect = ident
            ident.register_forward_pre_hook(self._capture)
            self._identity_installed = True

    def forward(self, imgs):
        """imgs: (B,3,640,640) RGB float[0,1]。
        返回 dict: {'P3','P4','P5': (B,d,H,W) 空间图, 'tokens': (B,8400,d)}。"""
        imgs = imgs.to(self.device)
        self._feats = {}
        if not self._unfrozen:                      # 全冻结 -> 无梯度
            with torch.no_grad():
                self.model(imgs)
        else:                                       # 至少 neck 可训 -> 带梯度前向
            self.model(imgs)
        assert len(self._feats) == 3, "Detect pre-hook 未捕获到 P3/P4/P5"

        parts, scales = [], {}
        for name in NECK_ORDER:
            f = self.scale_proj[name](self._feats[name])      # (B,d,H,W)
            scales[name] = f
            parts.append(f.flatten(2).transpose(1, 2))        # (B,h*w,d)
        return {**scales, 'tokens': torch.cat(parts, dim=1)}

    # ---------------- 数据预处理(仅装载用, 非训练接口) ----------------
    @classmethod
    def preprocess(cls, imgs_bgr, device=''):
        """list[np.ndarray] BGR 1280×720 uint8 -> (B,3,640,640) BGR float[0,1]。
        保持 BGR(不翻 RGB), 与 ultralytics 原生输入一致 —— 冻结的 COCO 主干
        是按 BGR 训的, 喂 RGB 会通道错位掉迁移质量。letterbox 必须与
        ultralytics 一致(LetterBox, padding=114)。"""
        from ultralytics.data.augment import LetterBox
        lb = LetterBox(new_shape=(YOLO_SIZE, YOLO_SIZE), auto=False, stride=32)
        batch = []
        for bgr in imgs_bgr:
            img = lb(image=bgr)                             # (640,640,3) BGR
            x = img.astype(np.float32) / 255.0
            batch.append(np.transpose(x, (2, 0, 1)))
        x = torch.from_numpy(np.stack(batch)).to(
            device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        return x

    # ---------------- tokens 切回三尺度空间图 ----------------
    @staticmethod
    def split_image_tokens(tokens):
        """(B,8400,d) -> {'P3':(B,d,80,80),'P4':(B,d,40,40),'P5':(B,d,20,20)}。
        与 N_IMG_SPLIT 分界一致; 融合输出的 image_tokens 切回检测 head 用。"""
        B, _, d = tokens.shape
        out = {}
        for name, (a, b), (h, w) in zip(NECK_ORDER, zip(N_IMG_SPLIT[:-1], N_IMG_SPLIT[1:]), NECK_HW.values()):
            out[name] = tokens[:, a:b].permute(0, 2, 1).view(B, d, h, w)
        return out

    def __repr__(self):
        return (f"ImageEncoder(freeze_backbone={self.freeze_backbone}, "
                f"unfreeze_neck={self.unfreeze_neck}({len(self._unfrozen)} mods), "
                f"d_model={self.d_model}, device={self.device})")


if __name__ == '__main__':
    enc = ImageEncoder()
    print(enc)
    dummy = torch.zeros(1, 3, YOLO_SIZE, YOLO_SIZE).to(enc.device)
    out = enc(dummy)
    print({k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in out.items()})
