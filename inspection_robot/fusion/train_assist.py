"""
点云辅助 2D 检测 —— 训练脚本
================================================================
在 output_v7 配对数据上训练 PointCloudAssistedYOLO (冻结 YOLO 主干+neck,
可训融合分支+检测头)。`--no-point` 关掉点云/位姿融合 = RGB-only 基线, 与
点云辅助版共享全部其它代码, 公平消融对比。

用法:
    .venv/bin/python fusion/train_assist.py                # 默认 30 epoch, 点云辅助
    .venv/bin/python fusion/train_assist.py --no-point     # RGB-only 基线
    .venv/bin/python fusion/train_assist.py --quick        # 2 epoch 冒烟(loss 下降验证)
    .venv/bin/python fusion/train_assist.py --epochs 60 --batch 8 --out runs/assist_v1

产物:
    {out}/last.pt  —— 只存可训参数 + 元信息(冻结主干不入档, 载回时从 YOLO ckpt 重建)
    {out}/train.log —— box/cls/dfl loss 曲线
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from fusion.config import HYP, NC, OUTPUT_V7, YOLO_SIZE  # noqa: E402
from fusion.pointcloud_assist_detector import (  # noqa: E402
    PointCloudAssistedYOLO, collate_batch, load_sample,
)


def parse_args():
    p = argparse.ArgumentParser(description='点云辅助 2D 检测训练')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3, help='检测头/融合/分类器学习率(主组)')
    p.add_argument('--lr-backbone', type=float, default=1e-5,
                   help='图像主干/neck 学习率(仅解锁时生效); 预训练主干微调要低, 否则冲掉迁移特征')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--no-point', action='store_true', help='关闭点云/位姿融合, 跑 RGB-only 基线')
    p.add_argument('--point-ckpt', default='', help='预训练 PointCloudEncoder 权重'
                    '(pretrain_point.py 产物 runs/attrmap_ae/point_encoder.pt), 空则随机初始化')
    p.add_argument('--img-ckpt', default='', help='图像主干 ckpt(默认 COCO yolo11l); '
                    '可传缺陷/裂缝微调后的 YOLO best.pt, 用其主干+neck 特征')
    p.add_argument('--no-aug', action='store_true', help='关闭 hflip/光度增强(默认开)')
    p.add_argument('--no-transfer-head', action='store_true',
                   help='不把 YOLO11l 预训练检测头搬进 head(默认开, 小数据必需)')
    p.add_argument('--unfreeze-backbone', action='store_true',
                   help='解锁整个图像主干+neck 端到端微调 (修复冻结下检测头塌缩, 需调低 lr/batch)')
    p.add_argument('--unfreeze-neck', type=int, default=0,
                   metavar='N', help='只解锁图像模型末段 N 个 neck 模块(含 Detect 前), 0=不解')
    p.add_argument('--quick', action='store_true', help='2 epoch 冒烟, 验证 loss 下降')
    p.add_argument('--out', default='runs/assist_v1', help='输出目录')
    p.add_argument('--data', default=OUTPUT_V7)
    p.add_argument('--nc', type=int, default=None, metavar='N',
                   help='检测类别数; 缺省读 data.yaml 的 nc(如 data1=6), 无则回退 config.NC')
    p.add_argument('--target-class', type=int, default=None, metavar='ID',
                   help='只检测该类别(nc 收窄为 1): 只保留 class ID 的框并重映射为 0, '
                        '其它类框归背景负样本。例: --target-class 1 即渗漏水检测。')
    p.add_argument('--class-weights', action='store_true',
                   help='按训练集类别逆频率加权 cls loss, 抑制少数类塌缩(4/7 类 0 检测的根因)')
    p.add_argument('--cw-power', type=float, default=1.5,
                   help='逆频率上奏幂 (>=1 加强少数类; 1=线性逆频, 越大多数类越被推)')
    p.add_argument('--loss', default='v8', choices=('v8', 'range-qfl'),
                   help='v8=ultralytics 原版(默认, 行为不变); range-qfl=RangeQFLDetectionLoss')
    p.add_argument('--qfl-gamma', type=float, default=2.0, help='QFL 聚焦指数')
    p.add_argument('--qfl-beta', type=float, default=0.5, help='尺度加权指数 (0=关闭)')
    p.add_argument('--qfl-wmin', type=float, default=0.5, help='尺度权重下限')
    p.add_argument('--qfl-wmax', type=float, default=3.0, help='尺度权重上限')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================================================================
# 数据增强 —— 156 帧从零训检测头必需要, 否则头塌缩(大框+某类独大)
# ================================================================
def augment_sample(s, rng, flip=0.5, jitter=0.5):
    """对 load_sample 的 5 元组做轻度几何+光度增强, 返回新 5 元组。
    hflip 同时翻图像/属性图/框, 跨模态镜像一致; pose 不变(翻转只是镜反射视野,
    不改变机器人位姿, 故位姿条件保持正确)。仅影响图像的光度抖动不破坏任何对齐。"""
    tensor, attrmap, pose, cls, xyxy = s
    tensor = tensor.copy(); attrmap = attrmap.copy()
    if rng.random() < flip:
        tensor = tensor[:, :, ::-1]                    # (3,640,640) 宽轴
        attrmap = attrmap[:, :, ::-1]                  # (3,720,1280) 宽轴
        if xyxy.shape[0]:
            xyxy = np.stack([YOLO_SIZE - xyxy[:, 2], xyxy[:, 1],
                             YOLO_SIZE - xyxy[:, 0], xyxy[:, 3]], 1)
    if rng.random() < jitter:                          # 亮度/对比度, 不碰几何
        b = rng.uniform(0.7, 1.3); c = rng.uniform(0.8, 1.2)
        m = float(tensor.mean())
        tensor = np.clip((tensor - m) * c + m * b + rng.uniform(-0.05, 0.05), 0, 1)
    return tensor, attrmap, pose, cls, xyxy


def transfer_pretrained_head(model, device=''):
    """把 YOLO11l 预训练 Detect 头的 box-DFL(几何先验)与 cls 特征抽取层搬进
    nc=7 检测头, 只重初始化 cls 最后一层 1×1。box_head 通道(256/512/512)与
    本工程一致(主干被 max_channels=512 截断), 故可整体搬家; 骨架从 COCO 学到的
    目标几何/特征口径直接复用, 小数据下检测头才不会塌缩。"""
    from fusion.config import YOLO11L_CKPT
    src = torch.load(YOLO11L_CKPT, map_location='cpu', weights_only=False)
    src_detect = next(m for m in src['model'].float().model if m.__class__.__name__ == 'Detect')
    det = model.detect
    for i in range(len(det.one2many['box_head'])):
        det.one2many['box_head'][i].load_state_dict(src_detect.one2many['box_head'][i].state_dict())
    for i in range(len(det.one2many['cls_head'])):
        dst, s = det.one2many['cls_head'][i], src_detect.one2many['cls_head'][i]
        dst[0].load_state_dict(s[0].state_dict())      # DWConv+1×1 特征抽取(类无关)可迁移
        dst[1].load_state_dict(s[1].state_dict())
    det.bias_init()                                    # 重设 7 类先验
    return model


def neutralize_cls_head(model):
    """清掉 cls 末层 1×1 的饱和负偏置, 换零初始化(weight=0, bias=0)。

    背景: transfer_pretrained_head 调用 det.bias_init() 后, cls 末层 bias 被设为
    log(5/nc/(640/stride)^2), nc=7 时 str=8/16/32 分别 ≈ -9.1/-7.7/-6.3 (强负)。
    这在 ultralytics 完整管线里靠"COCO 已知类先验 + 主干全可训 + warmup"才能逃出;
    本工程冻结主干 + 156 帧小数据 + 无 warmup, 保守初始让所有 sigmoid≈0.0001,
    conf_min=0.001 下的 NMS 直接丢弃全部框 -> 检测头塌缩(7 类全 0 检测)。

    改为零初始化: 初始 logits=0、sigmoid=0.5, 正/负样本都拿到对称梯度压力,
    头真正能学分离; 对点云/no-point 两臂完全一致, 保证公平消融。
    """
    with torch.no_grad():
        for i in range(len(model.detect.one2many['cls_head'])):
            cls_conv = model.detect.one2many['cls_head'][i][-1]   # 末层 1×1 分类 conv
            if hasattr(cls_conv, 'weight'):
                cls_conv.weight.data.zero_()
            if hasattr(cls_conv, 'bias'):
                cls_conv.bias.data.zero_()
    return model


def build_param_groups(model, lr, lr_backbone):
    """按模块来源分组学习率: 图像主干/neck(可迁移预训练)用低 lr, 其余(检测头/融合/
    点云编码器/分类器末层)用主 lr。随机初始化的分类器末层需要高 lr 才能从零学起来,
    否则 156 帧+类别不均衡下必然饿死塌缩。"""
    backbone, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if n.startswith('image_encoder.model.') else head).append(p)
    n_b, n_h = len(backbone), len(head)
    print(f"[param-groups] backbone={n_b} @ lr={lr_backbone:.1e}, head={n_h} @ lr={lr:.1e}")
    return [{'params': backbone, 'lr': lr_backbone}, {'params': head, 'lr': lr}]


def compute_class_weights(data_root, idx, nc, power=1.5):
    """按训练集类别逆频率算 cls 权重, 抑制少数类塌缩。

    7 类里 crack/spalling/rail_fastener_missing/rail_fastener_loose 在 156 帧小数据+类别
    不均衡下被高频类别独吞, 检测 head 对它们输出全 0。v8DetectionLoss 支持 `class_weights`
    (multiply BCE per-class), 用逆频上奏少数类的正样本梯度, 让这些类也能拿到分离信号。

    返回 shape (nc,) 的权重: w = (total/(count+0.5))^power, 再归一化让最常见类=1(只上奏,
    不下压多数类, 保护 broken/equip 这些已能检出的类)。power>1 放大少数类被压制的程度。
    """
    counts = np.zeros(nc, np.float64)
    for i in idx:
        p = os.path.join(data_root, 'labels', f'{i:06d}.txt')
        if os.path.exists(p) and os.path.getsize(p) > 0:
            lab = np.loadtxt(p, dtype=np.float32).reshape(-1, 5)
            for c in lab[:, 0].astype(int):
                if c < nc:
                    counts[c] += 1
    total = counts.sum()
    inv = (total / (counts + 0.5))
    w = (inv / inv.min()) ** power        # 最常见类=1, 其余上奏
    return w


def _resolve_nc(data_root, target_class, args_nc):
    """类别数: --nc 显式 > target_class(收窄 1) > data.yaml 声明 > config.NC。
    data1 源 6 类但 config.NC=7(以 output_v7 计), 不从 data.yaml 读 nc 会把头建成
    7 类、第 7 类永无正样本, cls loss 被空白类抬偏。"""
    if target_class is not None:
        return 1
    if args_nc:
        return args_nc
    yaml_p = os.path.join(data_root, 'data.yaml')
    if os.path.exists(yaml_p):
        import re
        m = re.search(r'^nc:\s*(\d+)', open(yaml_p, encoding='utf-8').read(), re.M)
        if m:
            return int(m.group(1))
    return NC


def build_lr(opt, lr, epochs):
    """无 warmup 的简单余弦衰减, 让头缓缓落到收敛。"""
    import math
    def fn(ep):
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ep / epochs)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=fn)


def main():
    args = parse_args()
    if args.quick:
        args.epochs, args.batch = 2, 4
        args.out = os.path.join(args.out, '_quick')
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- 样本划分 ----
    n_img = len([f for f in os.listdir(os.path.join(args.data, 'images')) if f.endswith('.jpg')])
    idx = list(range(n_img))
    random.shuffle(idx)
    n_val = max(1, int(n_img * args.val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"[data] {args.data}: {n_img} 帧, train {len(train_idx)} / val {len(val_idx)}, "
          f"device {device}, use_point={not args.no_point}")

    # ---- 模型 ----
    nc = _resolve_nc(args.data, args.target_class, args.nc)
    if nc != NC:
        print(f"[nc] {args.data} 声明 {nc} 类 (config.NC={NC}), 检测头按 {nc} 类建")
    model = PointCloudAssistedYOLO(use_point=not args.no_point, nc=nc, device=device,
                                   ckpt=(args.img_ckpt or None),
                                   freeze_backbone=not args.unfreeze_backbone,
                                   unfreeze_neck=args.unfreeze_neck)
    if args.img_ckpt:
        print(f"[img-ckpt] 图像主干使用 {args.img_ckpt} (非 COCO yolo11l)")
    if args.point_ckpt and not args.no_point:
        ck = torch.load(args.point_ckpt, map_location='cpu', weights_only=False)
        enc_sd = ck['encoder'] if isinstance(ck, dict) and 'encoder' in ck else ck
        miss, unexp = model.point_encoder.load_state_dict(enc_sd, strict=False)
        assert not miss, f"预训练点云编码器缺键: {miss}"
        print(f"[point-ckpt] 载入预训练编码器 {args.point_ckpt} "
              f"(missing={len(miss)}, unexpected={len(unexp)})")
    # 小数据从零训练检测头会塌缩: 优先把预训练检测头搬进来给几何先验
    if not args.no_transfer_head:
        transfer_pretrained_head(model)
        print("[transfer-head] 已载入 YOLO11l 预训练 box-DFL + cls 特征抽取层")
    # 迁移头会 bias_init 设置强负 cls 偏置(见 neutralize_cls_head docstring),
    # 冻结主干+小数据下必塌缩成"0 检测"; 零初始化 cls 末层, 让头能学。
    neutralize_cls_head(model)
    print("[neutralize-cls] cls 末层已零初始化(weight=0,bias=0), 逃出 -9 饱和")
    if args.loss != 'v8':
        from fusion.loss_range_qfl import RangeQFLDetectionLoss
        from fusion.pointcloud_assist_detector import _LossContainer
        model.criterion = RangeQFLDetectionLoss(
            _LossContainer(model.detect, model.criterion.hyp),
            gamma=args.qfl_gamma, beta=args.qfl_beta,
            w_min=args.qfl_wmin, w_max=args.qfl_wmax)
        print(f"[loss] RangeQFL gamma={args.qfl_gamma} beta={args.qfl_beta} "
              f"w=[{args.qfl_wmin},{args.qfl_wmax}] (box/dfl 与原版一致)")
    if args.class_weights:
        cw = compute_class_weights(args.data, train_idx, model.nc, args.cw_power)
        model.criterion.class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
        # 只改变类间相对权重, 用 1/mean 补偿 cls 增益全局尺度, 避免大类背景 BCE 把 cls loss 抬到
        # 数百(见 smoke 的 675 尖峰)而淹没 box/dfl。相对上奏(少数类 vs 多数类)不受影响。
        model.criterion.hyp.cls = HYP['cls'] / float(cw.mean())
        print(f"[class-weights] 上奏少数类 (power={args.cw_power}): {np.round(cw, 2).tolist()} | "
              f"cls 增益 {HYP['cls']}->{model.criterion.hyp.cls:.3f} (均值补偿)")
    model.train()
    opt = torch.optim.AdamW(build_param_groups(model, args.lr, args.lr_backbone),
                            lr=args.lr, weight_decay=0.05)
    sched = build_lr(opt, args.lr, args.epochs)
    print(model)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, 'train.log')
    with open(log_path, 'w') as flog:
        flog.write('epoch,iter,box,cls,dfl,val_box,val_cls,val_dfl,seconds\n')

    def run_batch(idxs, grad, aug=False):
        """装一批, 前向+反向(或 no_grad 验证), 返回 loss 分量 dict。训练批做增强。"""
        samples = [load_sample(i, root=args.data, target_class=args.target_class) for i in idxs]
        if aug:
            samples = [augment_sample(s, aug_rng) for s in samples]
        imgs, attrmaps, poses, batch = collate_batch(samples)
        imgs = imgs.to(device); attrmaps = attrmaps.to(device); poses = poses.to(device)
        for k, v in batch.items():
            batch[k] = v.to(device)
        if grad:
            loss, ld = model(imgs, attrmaps, poses, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            return {k: float(v) for k, v in ld.items()}, float(loss.detach())
        with torch.no_grad():
            _, ld = model(imgs, attrmaps, poses, batch)   # model.train() 下返回 loss
            return {k: float(v) for k, v in ld.items()}, None

    # ---- 训练 ----
    aug_rng = random.Random(args.seed + 1)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        random.shuffle(train_idx)
        for bi in range(0, len(train_idx), args.batch):
            ld, _ = run_batch(train_idx[bi:bi + args.batch], grad=True, aug=not args.no_aug)
            if bi % (args.batch * 4) == 0 or args.quick:
                print(f"  ep{ep:02d} it{bi//args.batch:03d} "
                      f"box={ld['box_loss']:.3f} cls={ld['cls_loss']:.1f} dfl={ld['dfl_loss']:.3f}")
        sched.step()
        # 验证
        vld = {'box_loss': [], 'cls_loss': [], 'dfl_loss': []}
        for bi in range(0, len(val_idx), args.batch):
            ldv, _ = run_batch(val_idx[bi:bi + args.batch], grad=False)
            for k in vld:
                vld[k].append(ldv[k])
        vavg = {k: float(np.mean(v)) for k, v in vld.items()}
        line = (f"{ep},{-1},{ld['box_loss']:.3f},{ld['cls_loss']:.1f},{ld['dfl_loss']:.3f},"
                f"{vavg['box_loss']:.3f},{vavg['cls_loss']:.1f},{vavg['dfl_loss']:.3f},"
                f"{time.time()-t0:.1f}")
        with open(log_path, 'a') as flog:
            flog.write(line + '\n')
        print(f"[ep{ep:02d}] val box={vavg['box_loss']:.3f} cls={vavg['cls_loss']:.1f} "
              f"dfl={vavg['dfl_loss']:.3f} lr={sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)")
        # 周期落盘: 长训练被中断(如系统内存回收)时也留有可用权重, 不必从头重跑。
        # .cpu() 对 cuda 张量返回副本, 不动模型内的原参数, 训练可继续。
        if ep % 5 == 0 or ep == args.epochs:
            sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({'nc': model.nc, 'use_point': model.use_point, 'epochs': ep,
                        'hyp': dict(vars(model.criterion.hyp)), 'model': sd,
                        'args': vars(args)}, os.path.join(args.out, 'last.pt'))

    # ---- 存档 (可训参数 + 全部 BN buffer) ----
    # 必须连 BN 的 running_mean/running_var/num_batches_tracked 一起存: 解冻 neck 后训练在
    # train 模式下跑, 模型内所有 BN 的 running 统计都更新成当前数据分布; 若只存可训参数,
    # 评估时载回的是 COCO 预训练的原版 BN 统计 -> 特征分布错位, 分类器瞬间塌缩(训练 loss
    # 健康却检测全垃圾的假象)。故存全部 state_dict(params+buffers)。
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt = {
        'nc': model.nc, 'use_point': model.use_point, 'epochs': args.epochs,
        'hyp': dict(vars(model.criterion.hyp)), 'model': sd,
        'args': vars(args),
    }
    torch.save(ckpt, os.path.join(args.out, 'last.pt'))
    print(f"[save] {os.path.join(args.out, 'last.pt')} (全 state_dict {len(sd)} 项, 含 BN 统计)")


if __name__ == '__main__':
    main()
