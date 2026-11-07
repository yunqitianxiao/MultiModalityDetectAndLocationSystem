"""图像（YOLO11l 冻结主干）与点云（attrmap CNN）两条编码器。"""

import numpy as np
import torch
import torch.nn as nn

from . import config as C


class ImageEncoder(nn.Module):
    """(B,3,640,640) BGR float[0,1] -> {P3,P4,P5 空间图} + tokens(B,8400,d)。

    用 Detect 层的 forward_pre_hook 捕获其输入（即 neck 输出），不硬编码层索引。
    主干默认冻结；unfreeze_neck=N 时解锁末 N 个模块参与反传。
    """

    def __init__(self, ckpt=None, d_model=C.D_MODEL, freeze_backbone=True,
                 unfreeze_neck=0, device=''):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.d_model = d_model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        ckpt = ckpt or C.YOLO_CKPT
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        model = ck.get('model')
        if model is None:
            model = ck.get('ema')
        assert model is not None, f'{ckpt} 里既无 model 也无 ema'
        self.model = model.float().eval().to(self.device)

        from ultralytics.nn.modules.head import Detect
        self._detect = next(m for m in self.model.model if isinstance(m, Detect))
        self._feats = {}
        self._detect.register_forward_pre_hook(self._capture)

        self.scale_proj = nn.ModuleDict({
            name: nn.Conv2d(C.NECK_CH[name], d_model, 1) for name in C.NECK_ORDER
        })

        for p in self.model.parameters():
            p.requires_grad_(False)
        self._unfrozen = []
        if not freeze_backbone and unfreeze_neck > 0:
            n = len(self.model.model)
            for mi in range(n - 1, max(n - 1 - unfreeze_neck, 0), -1):
                for p in self.model.model[mi].parameters():
                    p.requires_grad_(True)
                    self._unfrozen.append(p)

        self.to(self.device)

    def _capture(self, module, args):
        self._feats = {name: f for name, f in zip(C.NECK_ORDER, args[0])}

    def forward(self, imgs):
        imgs = imgs.to(self.device)
        self._feats = {}
        if self._unfrozen:
            self.model(imgs)
        else:
            with torch.no_grad():
                self.model(imgs)
        assert len(self._feats) == 3, 'Detect pre-hook 未捕获到 P3/P4/P5'

        scales, parts = {}, []
        for name in C.NECK_ORDER:
            f = self.scale_proj[name](self._feats[name])
            scales[name] = f
            parts.append(f.flatten(2).transpose(1, 2))
        return {**scales, 'tokens': torch.cat(parts, dim=1)}

    def __repr__(self):
        return (f'ImageEncoder(freeze_backbone={self.freeze_backbone}, '
                f'unfreeze_neck={len(self._unfrozen)} mods, '
                f'd_model={self.d_model}, device={self.device})')


class PointCloudEncoder(nn.Module):
    """(B,3,720,1280) 原始 attrmap -> feat(B,d,45,80) + tokens(B,3600,d)。"""

    def __init__(self, in_ch=C.ATTR_CH, out_ch=C.D_MODEL, device=''):
        super().__init__()
        self.out_ch = out_ch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.register_buffer('depth_scale', torch.tensor(C.DEPTH_SCALE))
        self.register_buffer('height_offset', torch.tensor(C.HEIGHT_OFFSET))
        self.register_buffer('height_scale', torch.tensor(C.HEIGHT_SCALE))

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
                nn.BatchNorm2d(c_out), nn.GELU())

        self.net = nn.Sequential(
            block(in_ch, 32), block(32, 64), block(64, 128), block(128, out_ch))
        self.to(self.device)

    def _normalize(self, x):
        d = torch.clamp(x[:, 0:1] / self.depth_scale, 0.0, 1.0)
        r = torch.clamp(x[:, 1:2], 0.0, 1.0)
        h = torch.clamp((x[:, 2:3] - self.height_offset) / self.height_scale, 0.0, 1.0)
        return torch.cat([d, r, h], dim=1)

    def forward(self, attr):
        if not isinstance(attr, torch.Tensor):
            attr = torch.from_numpy(np.asarray(attr, dtype=np.float32))
        if attr.dim() == 3:
            attr = attr.unsqueeze(0)
        x = self._normalize(attr.to(self.device))
        feat = self.net(x)
        return {'feat': feat, 'tokens': feat.flatten(2).transpose(1, 2)}

    def __repr__(self):
        return f'PointCloudEncoder(out_ch={self.out_ch}, device={self.device})'
