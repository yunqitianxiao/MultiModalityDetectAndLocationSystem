"""DualModalYOLO：两档消融（rgb / rgb+pc）共用同一套代码。"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from . import config as C
from .encoders import ImageEncoder, PointCloudEncoder
from .fusion import CrossModalFusion


class _LossContainer(nn.Module):
    """v8DetectionLoss 需要 model.args + model.model[-1] 的壳。"""

    def __init__(self, detect, args):
        super().__init__()
        self.model = nn.ModuleList([nn.Identity(), detect])
        self.args = args
        self.class_weights = None


class DualModalYOLO(nn.Module):
    def __init__(self, ckpt=None, nc=C.NC, modalities='rgb+pc',
                 freeze_backbone=True, unfreeze_neck=0, device=''):
        super().__init__()
        assert modalities in ('rgb', 'rgb+pc'), modalities
        self.modalities = modalities
        self.use_point = modalities == 'rgb+pc'
        self.nc = nc
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_encoder = ImageEncoder(
            ckpt=ckpt, freeze_backbone=freeze_backbone,
            unfreeze_neck=unfreeze_neck, device=self.device)
        if self.use_point:
            self.point_encoder = PointCloudEncoder(device=self.device)
            self.fusion = CrossModalFusion(device=self.device)

        self.head_proj = nn.ModuleDict({
            name: nn.Conv2d(C.D_MODEL, C.NECK_CH[name], 1) for name in C.NECK_ORDER})

        from ultralytics.nn.modules.head import Detect
        self.detect = Detect(nc=nc, reg_max=C.REG_MAX, end2end=False,
                             ch=tuple(C.NECK_CH[n] for n in C.NECK_ORDER))
        self.detect.stride = torch.tensor(C.DETECT_STRIDE)

        self.to(self.device)

        # 必须在 .to(device) 之后创建：criterion 内部的 proj/bce 会固定在创建时的 device
        from ultralytics.utils.loss import v8DetectionLoss
        self.criterion = v8DetectionLoss(
            _LossContainer(self.detect, SimpleNamespace(**C.HYP)))

    def forward(self, imgs, attrmap=None, batch=None):
        img_out = self.image_encoder(imgs)

        if self.use_point:
            assert attrmap is not None, 'rgb+pc 档必须给 attrmap'
            pc_out = self.point_encoder(attrmap)
            scales = self.fusion(img_out['tokens'], pc_out['tokens']).image_scales
        else:
            scales = {name: img_out[name] for name in C.NECK_ORDER}

        feats = [self.head_proj[name](scales[name]) for name in C.NECK_ORDER]
        preds = self.detect(feats)

        if batch is not None:
            loss, loss_dict = self.criterion(preds, batch)
            return loss.sum(), loss_dict
        return preds

    def trainable_params(self):
        return (p for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        n_train = sum(p.numel() for p in self.trainable_params())
        return (f'DualModalYOLO(modalities={self.modalities}, nc={self.nc}, '
                f'd_model={C.D_MODEL}, trainable={n_train/1e6:.2f}M, device={self.device})')
