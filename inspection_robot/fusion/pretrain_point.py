"""
点云编码器自监督预训练 —— 掩码自编码 (MAE 式)
================================================================
把 PointCloudEncoder (attrmap CNN) 从随机初始化变成能理解 attrmap 结构的
编码器, 再进点云辅助 2D 检测的联合训练 —— 接口先行策略的补强: 之前实测
随机点云编码器让点云辅助反而更差(注入噪声), 预训练就是先教会编码器几何结构。

任务 (无需任何标签, 用全部 output_v7 attrmaps):
    按 45×80 网格随机遮 75% 的 16×16 输入块(置零)
    -> 编码器只看到可见块, 输出 feat(256,45,80)
    -> CNN 解码器上采样重建整张归一化 attrmap
    -> 损失 = 被遮区域的重建 MSE (avgpool16 到网格粒度再比较)

编码器学到"用局部几何(深度连续性/反射率/高度)补全全局" —— 这正是交叉注意力
里点云 tokens 该提供的信息。预训练只动编码器, 解码器训练完即弃。

用法:
    .venv/bin/python fusion/pretrain_point.py                 # 默认 20 epoch
    .venv/bin/python fusion/pretrain_point.py --epochs 30 --quick
产物: {out}/point_encoder.pt  = PointCloudEncoder.state_dict(含 buffer), 直接
       load_state_dict 进 PointCloudAssistedYOLO.point_encoder。
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from fusion.config import ATTR_CH, ATTR_H, ATTR_W, OUTPUT_V7, PC_H, PC_W  # noqa: E402
from fusion.pointcloud_encoder import PointCloudEncoder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='点云编码器 MAE 自监督预训练')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--mask-ratio', type=float, default=0.75)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true', help='3 epoch 冒烟')
    p.add_argument('--out', default='runs/attrmap_ae')
    p.add_argument('--data', default=OUTPUT_V7)
    return p.parse_args()


class MaskedAttrmapAE(nn.Module):
    """掩码自编码: PointCloudEncoder + CNN 解码器, 重建被遮区域。"""

    def __init__(self, encoder=None, mask_ratio=0.75, device=''):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask_ratio = mask_ratio
        self.encoder = encoder or PointCloudEncoder(device=self.device)

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.GELU(),
            )
        # 45×80 -> 720×1280 (×16): 4 次 ×2 上采样 + 末层输出 3 通道。
        # 高分辨率中间张量是显存大头, 最后一层(720×1280 处)只留 8 通道。
        self.decoder = nn.Sequential(
            block(256, 128), block(128, 64), block(64, 32), block(32, 8),
            nn.Conv2d(8, ATTR_CH, 3, padding=1), nn.Sigmoid(),
        )
        self.to(self.device)

    # ---------------- 掩码 ----------------
    def make_mask(self, B):
        """(B,45,80) float, 1=被遮, ~mask_ratio 比例; 保证至少留一个可见块。"""
        m = (torch.rand(B, PC_H, PC_W, device=self.device) < self.mask_ratio)
        if bool(m.all()):
            m[0, 0, 0] = False
        return m.float()

    # ---------------- 前向 + 损失 ----------------
    def forward(self, attr):
        """attr (B,3,720,1280) 原始 attrmap -> (loss, {mse})。"""
        if not isinstance(attr, torch.Tensor):
            attr = torch.from_numpy(np.asarray(attr, dtype=np.float32))
        attr = attr.to(self.device)
        x = self.encoder._normalize(attr)                 # (B,3,720,1280) [0,1]
        B = x.shape[0]

        mask_grid = self.make_mask(B)                     # (B,45,80) 1=被遮
        mask_full = F.interpolate(mask_grid.unsqueeze(1), scale_factor=16,
                                  mode='nearest')          # (B,1,720,1280)
        x_masked = x * (1 - mask_full)                    # 被遮块置零

        feat = self.encoder.net(x_masked)                 # (B,256,45,80)
        pred = self.decoder(feat)                         # (B,3,720,1280) sigmoid

        # 池化到网格粒度比较, 只算被遮区域
        pred_pool = F.avg_pool2d(pred, 16)
        targ_pool = F.avg_pool2d(x, 16)
        mse = (pred_pool - targ_pool).pow(2) * mask_grid.unsqueeze(1)
        loss = mse.sum() / mask_grid.sum().clamp(min=1)
        return loss, {'mse': float(loss.detach())}

    def encoder_state_dict(self):
        """预训练产物: 只取 PointCloudEncoder 的权重(含归一化 buffer)。"""
        return self.encoder.state_dict()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 3
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 装载全部 attrmap (自监督不需要标签) ----
    attr_dir = os.path.join(args.data, 'attrmaps')
    paths = sorted(f for f in os.listdir(attr_dir) if f.endswith('.npz'))
    attrs = []
    for f in paths:
        a = np.load(os.path.join(attr_dir, f))
        # float16 常驻: 624 帧 float32 全量 np.stack 约 6.9GB, 与其他训练并发会触发 OOM;
        # attrmap 已归一化到 [0,1], 半精度足够, 训练时按 batch 转回 float32。
        attrs.append(np.stack([a['depth'], a['reflec'], a['height']]).astype(np.float16))
    attrs = np.stack(attrs)
    print(f"[data] {len(attrs)} 帧 attrmap {tuple(attrs.shape)}, device {device}, "
          f"mask_ratio={args.mask_ratio}")

    model = MaskedAttrmapAE(mask_ratio=args.mask_ratio, device=device)
    model.train()
    opt = torch.optim.AdamW(model.encoder.parameters(), lr=args.lr, weight_decay=0.05)
    n_batch = int(np.ceil(len(attrs) / args.batch))

    os.makedirs(args.out, exist_ok=True)
    log = os.path.join(args.out, 'pretrain.log')
    with open(log, 'w') as f:
        f.write('epoch,mse,seconds\n')

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        order = np.random.permutation(len(attrs))
        ep_loss = []
        for bi in range(n_batch):
            idx = order[bi * args.batch:(bi + 1) * args.batch]
            x = torch.from_numpy(attrs[idx]).float().to(device)
            loss, ld = model(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss.append(ld['mse'])
        m = float(np.mean(ep_loss))
        with open(log, 'a') as f:
            f.write(f'{ep},{m:.6f},{time.time()-t0:.1f}\n')
        print(f"[ep{ep:02d}] masked-MSE={m:.6f} ({time.time()-t0:.0f}s)")
        # 周期性落盘: 进程被内存回收/中断时也留有可用编码器, 不必从头重跑
        if ep % 4 == 0 or ep == args.epochs:
            torch.save({'encoder': model.encoder_state_dict(),
                        'mask_ratio': args.mask_ratio, 'epochs': ep, 'mse': m},
                       os.path.join(args.out, 'point_encoder.pt'))

    # ---- 只存编码器(解码器训练完即弃) ----
    torch.save({'encoder': model.encoder_state_dict(),
                'mask_ratio': args.mask_ratio, 'epochs': args.epochs},
               os.path.join(args.out, 'point_encoder.pt'))
    print(f"[save] {os.path.join(args.out, 'point_encoder.pt')} "
          f"(PointCloudEncoder state_dict, {len(model.encoder_state_dict())} 项)")


if __name__ == '__main__':
    main()
