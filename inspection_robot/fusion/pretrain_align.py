"""
多模态自监督对齐预训练 —— 图像+点云+位姿 联合预训练 (零标注)
================================================================
目的 (接口先行策略的补强, 与 pretrain_point.py / pretrain_depth.py 互补):
    在标注到位前把"融合栈"教会: PointCloudEncoder + 融合(CrossModalFusion)
    + 位姿条件, 让图像/点云/位姿三路共享一个几何一致的表示, 而不是随机初化
    直接进联合训练(实测随机点云编码器让点云辅助反而更差, 注入噪声)。

与 pretrain_point 的区别: 那是对点云属性图做单模态 MAE; 本模块把点云遮块,
让**融合后的点云 token** 重建整张属性图 —— 重建信号穿过 CrossModalFusion,
图像与位姿条件都参与, 强制三路对齐。另一项是对齐损失 InfoNCE: 同帧的
融合后图像嵌入与点云嵌入拉近, 跨帧推开。

任务 (无需任何标签, 用全部 output_v7 配对帧):
    1. 遮块重建: 按 45×80 网格随机遮 75% 的 16×16 属性图块(置零)
       -> 图像(不漏)+遮后点云(漏)-> CrossModalFusion(位姿作 K/V)-> 融合点云 token
       -> 解码器上采样重建整张归一化属性图 -> 只算被遮区 MSE
    2. 对齐: InfoNCE(融合后 image/pc 嵌入, 同帧正样本), 权重 --nce-weight
    编码器学到"图像/位姿能补点云几何" + "三路表示对齐" —— 正是交叉注意力
    里该提供的信息。只动可训参数, 冻结 YOLO 主干由 ImageEncoder 内部处理。

用法:
    .venv/bin/python fusion/pretrain_align.py                  # 默认 20 epoch, nce=0.1
    .venv/bin/python fusion/pretrain_align.py --point-ckpt runs/attrmap_ae/point_encoder.pt
    .venv/bin/python fusion/pretrain_align.py --quick          # 3 epoch 冒烟
    .venv/bin/python fusion/pretrain_align.py --nce-weight 0   # 关掉 InfoNCE
产物: {out}/align_pretrain.pt —— 可训参数(state_dict) + 元信息
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

from fusion.config import ATTR_CH, D_MODEL, OUTPUT_V7, PC_H, PC_W, POSE_EMB  # noqa: E402
from fusion.image_encoder import ImageEncoder          # noqa: E402
from fusion.pointcloud_encoder import PointCloudEncoder  # noqa: E402
from fusion.fusion import CrossModalFusion             # noqa: E402
from fusion.pointcloud_assist_detector import collate_batch, load_sample  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='多模态(图像+点云+位姿) 自监督对齐预训练')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--mask-ratio', type=float, default=0.75)
    p.add_argument('--nce-weight', type=float, default=0.1, help='InfoNCE 对齐损失权重(0 关闭)')
    p.add_argument('--nce-temp', type=float, default=0.07)
    p.add_argument('--point-ckpt', default='', help='可选: 预训练点云编码器权重(pretrain_point.py 产物)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quick', action='store_true', help='3 epoch 冒烟')
    p.add_argument('--out', default='runs/align_pretrain')
    p.add_argument('--data', default=OUTPUT_V7)
    return p.parse_args()


# ================================================================
# 1. 解码器: 融合点云 token -> 归一化属性图 (与 pretrain_point 同结构)
# ================================================================
class AttrmapDecoder(nn.Module):
    """(B,3600,d) 融合点云 token -> (B,3,720,1280) 归一化属性图 [0,1]。"""

    def __init__(self, d_in=D_MODEL, out_ch=ATTR_CH):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(),
            )
        self.reshape_ch = 256                 # 与 PointCloudEncoder 输出 feat 通道一致
        # 45×80 -> 720×1280, 最后(高分辨率处)只留 8 通道省显存
        self.net = nn.Sequential(
            nn.Conv2d(d_in, self.reshape_ch, 1),
            block(256, 128), block(128, 64), block(64, 32),
            block(32, 8), nn.Conv2d(8, out_ch, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, tokens):
        """tokens (B,3600,d) -> (B,3,720,1280)。"""
        B, _, d = tokens.shape
        x = tokens.transpose(1, 2).view(B, d, PC_H, PC_W)   # (B,d,45,80)
        return self.net(x)


# ================================================================
# 2. 对齐预训练模型
# ================================================================
class AlignPretrain(nn.Module):
    def __init__(self, image_encoder=None, point_encoder=None, fusion=None,
                 pose_encoder=None, mask_ratio=0.75, device=''):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask_ratio = mask_ratio
        self.image_encoder = image_encoder or ImageEncoder(device=self.device)
        self.point_encoder = point_encoder or PointCloudEncoder(device=self.device)
        self.pose_encoder = pose_encoder                       # PoseEncoder(非 nn.Module, 冻结)
        self.fusion = fusion or CrossModalFusion(device=self.device)
        self.decoder = AttrmapDecoder()
        self.to(self.device)

    def make_mask(self, B):
        """(B,45,80) float 1=被遮, ~mask_ratio 比例; 保证至少留一个可见块。"""
        m = (torch.rand(B, PC_H, PC_W, device=self.device) < self.mask_ratio)
        if bool(m.all()):
            m[0, 0, 0] = False
        return m.float()

    def forward(self, imgs, attr, pose_state, nce_weight=0.0, nce_temp=0.07):
        """imgs(B,3,640,640) + attr(B,3,720,1280 原始) + pose_state(B,12)
        -> (loss, {'recon','nce'})。"""
        img_out = self.image_encoder(imgs)                    # {tokens:(B,8400,d), P3..}
        # 点云遮块(归一化空间, 与 encoder 一致)
        x = self.point_encoder._normalize(attr.to(self.device))   # (B,3,720,1280) [0,1]
        B = x.shape[0]
        mask_grid = self.make_mask(B)
        mask_full = F.interpolate(mask_grid.unsqueeze(1), scale_factor=16, mode='nearest')
        x_masked = x * (1 - mask_full)
        pc_feat = self.point_encoder.net(x_masked)            # (B,256,45,80)
        pc_tokens = pc_feat.flatten(2).transpose(1, 2)        # (B,3600,d)

        # 位姿(CPU->fusion 内部搬 target)
        pe = self.pose_encoder(pose_state.to('cpu') if torch.is_tensor(pose_state) else pose_state)
        if pe.dim() == 1:
            pe = pe.unsqueeze(0)

        fused = self.fusion(img_out['tokens'], pc_tokens, pe)   # FusionOutput

        # ---- 重建被遮点云属性图 ----
        pred = self.decoder(fused.point_tokens)               # (B,3,720,1280) sigmoid
        pred_pool = F.avg_pool2d(pred, 16)
        targ_pool = F.avg_pool2d(x, 16)
        mse = ((pred_pool - targ_pool).pow(2) * mask_grid.unsqueeze(1)).sum() / mask_grid.sum().clamp(min=1)
        recon = mse

        # ---- InfoNCE 对齐(融合后 image/pc 嵌入, 同帧正样本) ----
        nce = torch.zeros((), device=self.device)
        if nce_weight > 0:
            ie = F.normalize(fused.image_tokens.mean(dim=1), dim=-1)   # (B,d)
            pe2 = F.normalize(fused.point_tokens.mean(dim=1), dim=-1)  # (B,d)
            logits = (ie @ pe2.T) / nce_temp                            # (B,B)
            labels = torch.arange(B, device=self.device)
            nce = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return recon + nce_weight * nce, {'recon': float(recon.detach()), 'nce': float(nce.detach())}

    def trainable_params(self):
        """可训参数(冻结 YOLO 主干由 ImageEncoder 内部置 requires_grad=False)。"""
        return (p for p in self.parameters() if p.requires_grad)


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 3
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from data_gen.pose_encoder import PoseEncoder
    from fusion.config import POSE_CKPT

    n_img = len([f for f in os.listdir(os.path.join(args.data, 'images')) if f.endswith('.jpg')])
    print(f"[data] {args.data}: {n_img} 帧, device {device}, mask={args.mask_ratio}, nce={args.nce_weight}")

    point_encoder = PointCloudEncoder(device=device)
    if args.point_ckpt:
        ck = torch.load(args.point_ckpt, map_location='cpu', weights_only=False)
        enc_sd = ck['encoder'] if isinstance(ck, dict) and 'encoder' in ck else ck
        miss, unexp = point_encoder.load_state_dict(enc_sd, strict=False)
        assert not miss, f"预训练点云编码器缺键: {miss}"
        print(f"[point-ckpt] 载入 {args.point_ckpt} (missing={len(miss)}, unexpected={len(unexp)})")

    model = AlignPretrain(image_encoder=ImageEncoder(device=device),
                          point_encoder=point_encoder, fusion=CrossModalFusion(device=device),
                          pose_encoder=PoseEncoder(POSE_CKPT), mask_ratio=args.mask_ratio,
                          device=device)
    model.train()
    opt = torch.optim.AdamW(model.trainable_params(), lr=args.lr, weight_decay=0.05)

    os.makedirs(args.out, exist_ok=True)
    log = os.path.join(args.out, 'align_pretrain.log')
    with open(log, 'w') as f:
        f.write('epoch,recon,nce,seconds\n')

    from fusion.pointcloud_assist_detector import collate_batch, load_sample  # noqa: F811
    idx = list(range(n_img))
    n_batch = int(np.ceil(len(idx) / args.batch))
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        order = np.random.permutation(len(idx))
        recs, nces = [], []
        for bi in range(n_batch):
            bidxs = order[bi * args.batch:(bi + 1) * args.batch]
            imgs, attrmaps, poses, _ = collate_batch([load_sample(i, root=args.data) for i in bidxs])
            imgs = imgs.to(device); attrmaps = attrmaps.to(device); poses = poses.to(device)
            loss, ld = model(imgs, attrmaps, poses, nce_weight=args.nce_weight,
                             nce_temp=args.nce_temp)
            opt.zero_grad(); loss.backward(); opt.step()
            recs.append(ld['recon']); nces.append(ld['nce'])
        r, nc = float(np.mean(recs)), float(np.mean(nces))
        with open(log, 'a') as f:
            f.write(f'{ep},{r:.6f},{nc:.6f},{time.time()-t0:.1f}\n')
        print(f"[ep{ep:02d}] recon={r:.6f} nce={nc:.4f} ({time.time()-t0:.0f}s)")

    # ---- 只存可训参数(冻结主干不入档) ----
    trainable_keys = {k for k, p in model.named_parameters() if p.requires_grad}
    sd = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_keys}
    torch.save({'epochs': args.epochs, 'mask_ratio': args.mask_ratio,
                'nce_weight': args.nce_weight, 'model': sd},
               os.path.join(args.out, 'align_pretrain.pt'))
    print(f"[save] {os.path.join(args.out, 'align_pretrain.pt')} (可训参数 {len(sd)} 项)")

    # ---- 额外存一份裸 point_encoder(state_dict 已是子模块裸 key), 可直接喂 train_assist --point-ckpt ----
    pe_sd = {k: v.cpu() for k, v in model.point_encoder.state_dict().items()}
    torch.save({'encoder': pe_sd, 'mask_ratio': args.mask_ratio, 'epochs': args.epochs},
               os.path.join(args.out, 'point_encoder.pt'))
    print(f"[save] {os.path.join(args.out, 'point_encoder.pt')} (裸点云编码器, {len(pe_sd)} 项)")


if __name__ == '__main__':
    main()
