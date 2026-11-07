"""
图像编码器深度预训练 —— 单目深度估计/补全 (自监督, 无需任何标注)
================================================================
目的 (接口先行策略的补强, 与 pretrain_point.py 对应):
    让"图像编码器"在标注到位之前就具备几何感知表示 —— 用 LiDAR 稀疏深度
    当免费监督, 逼图像分支学会 3D 结构 + 抗隧道光照(深度是光度无关量)。

任务 (零标注, 用全部 output_v7 配对帧):
    输入 RGB 图像 (可选叠加稀疏深度+有效掩码做补全 / 纯单目估计)
      -> U-Net 编码器-解码器 -> 预测稠密深度 (720×1280)
    损失: 只对 LiDAR 有效像素算 masked SILog + L1 (标准深度损失, 尺度不变)

产物: {out}/depth_encoder.pt      —— DepthCompletionNet.encoder state_dict
      (stride-16 特征图, 45×80, 256通道; 与点云编码器同粒度, 可作融合模块
      图像分支的新编码器, 或作深度先验; 换进 ImageEncoder / 点云辅助检测即可)
      {out}/depth_completion.pt   —— 整网 state_dict (含 decoder, 用于做稠密深度)
      {out}/depth_pretrain.log    —— 训练曲线

两种无标注监督 (--target):
    sparse: 只对原始 LiDAR 点投影的有效像素取损失 (标准补全, 无补密偏差, 真实)
    dense : 对全像素取损失, 目标用 attrmap 的补密深度 (监督更密, 156帧也能训;
            带补密偏差, 训练量大时优先 sparse)

用法:
    .venv/bin/python fusion/pretrain_depth.py                 # 默认 completion, 20 epoch
    .venv/bin/python fusion/pretrain_depth.py --mono          # 纯单目(图像->深度), 不给稀疏输入
    .venv/bin/python fusion/pretrain_depth.py --target dense  # 补密深度作稠密监督
    .venv/bin/python fusion/pretrain_depth.py --quick         # 3 epoch 冒烟
"""

import argparse
import math
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

from fusion.config import DEPTH_SCALE, OUTPUT_V7  # noqa: E402
from data_gen.common import IMG_SIZE, K, T_CAM_BASE, lidar_to_image_attributes  # noqa: E402

RGB_CH = 3


def parse_args():
    p = argparse.ArgumentParser(description='图像编码器 单目深度估计/补全 自监督预训练')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mono', action='store_true',
                   help='纯单目估计(不给稀疏深度输入, 最强化图像编码器的几何含量)')
    p.add_argument('--target', choices=['sparse', 'dense'], default='sparse',
                   help='sparse=只对LiDAR有效像素取损失(标准补全); dense=全像素用补密深度')
    p.add_argument('--quick', action='store_true', help='3 epoch 冒烟')
    p.add_argument('--out', default='runs/depth_encoder')
    p.add_argument('--data', default=OUTPUT_V7)
    return p.parse_args()


# ================================================================
# 1. 稀疏深度读取: clouds/{i}.npy (车体系 [x,y,z,intensity]) -> 相机系稀疏深度+有效掩码
# ================================================================
def load_sample(i, root, mono, target='sparse'):
    """读第 i 帧 -> (rgb(BGR,3,720,1280), sparse_depth(1,720,1280) 无效=0, valid(1,720,1280),
    gt_depth(1,720,1280))。sparse/valid 由原点云投影(无效=NaN->0 / 0), gt 取稀疏或补密深度。"""
    import cv2
    rgb = cv2.imread(os.path.join(root, 'images', f'{i:06d}.jpg'))          # (720,1280,3) BGR
    assert rgb is not None, f'缺图像 {i}'
    rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0  # (3,720,1280)

    cloud = np.load(os.path.join(root, 'clouds', f'{i:06d}.npy'))          # (N,4) 车体系
    depth_map, _, _ = lidar_to_image_attributes(cloud[:, :3], T_CAM_BASE, K, IMG_SIZE)
    valid = np.isfinite(depth_map).astype(np.float32)[None]                 # (1,720,1280)
    sparse = np.where(np.isfinite(depth_map), depth_map, 0.0).astype(np.float32)[None]

    if target == 'dense':                                                  # 补密深度当稠密目标
        a = np.load(os.path.join(root, 'attrmaps', f'{i:06d}.npz'))
        gt = a['depth'].astype(np.float32)[None]                           # (1,720,1280) 稠密
        gt_valid = np.ones_like(gt, np.float32)
    else:
        gt, gt_valid = sparse, valid                                       # 稀疏监督(标准补全)

    return (torch.from_numpy(rgb), torch.from_numpy(sparse), torch.from_numpy(valid),
            torch.from_numpy(gt), torch.from_numpy(gt_valid))


def collate(samples):
    """list[load_sample] -> 堆叠成 batch tensor (全部 (B,*,720,1280))。"""
    return tuple(torch.stack(x) for x in zip(*samples))


# ================================================================
# 2. U-Net 深度估计/补全网络 (encoder 即产物: stride-16, 45×80, 256通道)
# ================================================================
class DepthCompletionNet(nn.Module):
    """单目深度估计/补全 U-Net。输入 (B,in_ch,720,1280) -> 对数深度 (B,1,720,1280)。

    in_ch: 补全=RGB(3)+稀疏深度(1)+有效掩码(1)=5; 纯单目=RGB(3)=3。
    encoder: 4 个 stride-2 卷积块 -> (B,256,45,80) 特征图 (与点云编码器同粒度)。
    """

    def __init__(self, in_ch=RGB_CH + 2, max_depth=DEPTH_SCALE, device=''):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_depth = float(max_depth)

        def down(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1),
                                 nn.BatchNorm2d(co), nn.GELU())

        def up(ci, co):  # ×2 上采样后接 3×3 卷积(不跨步)
            return nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                 nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU())

        # 编码器: 4 个 stride-2 块 (产物)
        self.encoder = nn.Sequential(down(in_ch, 32), down(32, 64), down(64, 128), down(128, 256))

        # 解码器: 每级 ×2 上采样 -> cat skip -> 1×1 卷积压回
        self.u1 = up(256, 128); self.c1 = nn.Conv2d(128 + 128, 128, 1)
        self.u2 = up(128, 64);  self.c2 = nn.Conv2d(64 + 64, 64, 1)
        self.u3 = up(64, 32);   self.c3 = nn.Conv2d(32 + 32, 32, 1)
        self.u4 = up(32, 8);    self.c4 = nn.Sequential(nn.Conv2d(8 + RGB_CH, 16, 3, padding=1), nn.GELU())
        self.head = nn.Conv2d(16, 1, 3, padding=1)                        # 输出对数深度

        self.to(self.device)

    def encode(self, x):
        """x (B,in_ch,720,1280) -> (feat(B,256,45,80), skips=[f3,f2,f1])。"""
        f1 = self.encoder[0](x)
        f2 = self.encoder[1](f1)
        f3 = self.encoder[2](f2)
        f4 = self.encoder[3](f3)
        return f4, [f3, f2, f1]

    def forward(self, x):
        """x (B,in_ch,720,1280) -> log_depth(B,1,720,1280)。"""
        f4, (f3, f2, f1) = self.encode(x)
        h = self.c1(torch.cat([self.u1(f4), f3], 1))     # 90×160
        h = self.c2(torch.cat([self.u2(h), f2], 1))      # 180×320
        h = self.c3(torch.cat([self.u3(h), f1], 1))      # 360×640
        h = self.u4(h)                                   # 720×1280 ×8
        h = self.c4(torch.cat([h, x[:, :RGB_CH]], 1))    # 拼接原图 RGB 做 skip
        return self.head(h)                              # 对数深度

    @staticmethod
    def depth(log_depth, max_depth):
        """对数深度 -> 深度 (m), clamp 到 [0,max_depth]。"""
        return torch.exp(log_depth.clamp(min=-6.0, max=float(np.log(max_depth))))


# ================================================================
# 3. 深度损失: masked SILog (尺度不变) + masked L1
# ================================================================
def depth_loss(log_pred, gt_depth, valid, max_depth, silog_w=0.85, l1_w=0.15):
    """log_pred(B,1,H,W), gt_depth(B,1,H,W), valid(B,1,H,W) 0/1。只对 valid 取损失。"""
    eps = 1e-6
    log_gt = torch.log(gt_depth.clamp_min(eps))
    d = (log_pred - log_gt) * valid
    n = valid.sum().clamp(min=1)
    silog = torch.sqrt(((d ** 2).sum() / n - (d.sum() / n) ** 2).clamp(min=0.0) + eps)
    pred_depth = DepthCompletionNet.depth(log_pred, max_depth)
    l1 = ((pred_depth - gt_depth).abs() * valid).sum() / n
    return silog_w * silog + l1_w * l1, {'silog': float(silog.detach()), 'l1': float(l1.detach())}


@torch.no_grad()
def rel_err_and_density(pred_depth, gt_depth, valid):
    """(平均相对误差, 有效像素占比%)。"""
    d = ((pred_depth - gt_depth).abs() / gt_depth.clamp(min=1e-3)) * valid
    rate = float((d.sum() / valid.sum().clamp(min=1)).detach())
    return rate, float(valid.mean().detach()) * 100.0


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 3
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_dir = os.path.join(args.data, 'images')
    n_img = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    idx = list(range(n_img))
    in_ch = RGB_CH if args.mono else RGB_CH + 2
    print(f"[data] {args.data}: {n_img} 帧, device {device}, "
          f"mode={'mono' if args.mono else 'completion'}, target={args.target}, in_ch={in_ch}")

    model = DepthCompletionNet(in_ch=in_ch, device=device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    n_batch = int(np.ceil(len(idx) / args.batch))

    os.makedirs(args.out, exist_ok=True)
    log = os.path.join(args.out, 'depth_pretrain.log')
    with open(log, 'w') as f:
        f.write('epoch,silog,l1,rel_err,seconds\n')

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        order = np.random.permutation(len(idx))
        silogs, l1s, rels = [], [], []
        for bi in range(n_batch):
            bidxs = order[bi * args.batch:(bi + 1) * args.batch]
            rgb, sparse, valid, gt, gt_valid = collate(
                [load_sample(i, args.data, args.mono, args.target) for i in bidxs])
            rgb = rgb.to(device)
            x = rgb if args.mono else torch.cat([rgb, sparse.to(device), valid.to(device)], 1)
            gt = gt.to(device); gt_valid = gt_valid.to(device)

            log_pred = model(x)
            loss, ld = depth_loss(log_pred, gt, gt_valid, model.max_depth)
            opt.zero_grad(); loss.backward(); opt.step()
            silogs.append(ld['silog']); l1s.append(ld['l1'])
            rels.append(rel_err_and_density(model.depth(log_pred, model.max_depth), gt, gt_valid)[0])
        s, l1, r = float(np.mean(silogs)), float(np.mean(l1s)), float(np.mean(rels))
        with open(log, 'a') as f:
            f.write(f'{ep},{s:.6f},{l1:.4f},{r:.4f},{time.time()-t0:.1f}\n')
        print(f"[ep{ep:02d}] silog={s:.4f} l1={l1:.4f} rel_err={r:.4f} ({time.time()-t0:.0f}s)")

    # ---- 产物: encoder(可换进图像分支) + 整网 ----
    torch.save({'encoder': model.encoder.state_dict(), 'en_ch': 256, 'stride': 16,
                'in_ch': in_ch, 'epochs': args.epochs},
               os.path.join(args.out, 'depth_encoder.pt'))
    torch.save({'model': model.state_dict(), 'in_ch': in_ch, 'epochs': args.epochs},
               os.path.join(args.out, 'depth_completion.pt'))
    print(f"[save] depth_encoder.pt(可换进图像分支) + depth_completion.pt(整网) -> {args.out}")


if __name__ == '__main__':
    main()
