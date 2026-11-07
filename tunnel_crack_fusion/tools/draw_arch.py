#!/usr/bin/env python3
"""画双模态（图像 + 点云 attrmap）裂缝检测基线模型结构图。

用法:
    python3 tools/draw_arch.py [输出路径]

产物: docs/figures/model_arch.png (默认)
"""

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, 'docs', 'figures', 'model_arch.png')

# ---------------- 配色 ----------------
C = {
    'input':  ('#E3F1E3', '#4A7C4A'),   # 输入张量
    'frozen': ('#E8ECF2', '#6B7785'),   # 冻结参数
    'train':  ('#DCEBFA', '#2E6DA4'),   # 可训练（编码器/投影）
    'fusion': ('#FCE4CE', '#C46A1B'),   # 融合模块（本项目新增）
    'head':   ('#EADDF2', '#6B4C8A'),   # 检测头
    'out':    ('#FFF3C4', '#9C7A00'),   # 输出
    'panel':  ('#FDF6EE', '#C46A1B'),
    'note':   ('#F7F9FA', '#CBD2D9'),
}
INK = '#333333'
ARROW = '#3A3A3A'


def pick_cjk_font():
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei',
                 'Droid Sans Fallback', 'AR PL UMing CN'):
        if cand in names:
            return cand
    return 'DejaVu Sans'


def rounded(ax, x, y, w, h, kind, lw=1.4, z=2, ls='-'):
    fc, ec = C[kind]
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012',
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z,
                       linestyle=ls)
    ax.add_patch(p)
    return p


def box(ax, x, y, w, h, kind, title, body='', tsize=11, bsize=9, lw=1.4, ls='-'):
    rounded(ax, x, y, w, h, kind, lw=lw, ls=ls)
    cy = y + h / 2
    if body:
        ax.text(x + w / 2, cy + h * 0.16, title, ha='center', va='center',
                fontsize=tsize, fontweight='bold', color=INK, zorder=3)
        ax.text(x + w / 2, cy - h * 0.20, body, ha='center', va='center',
                fontsize=bsize, color='#555555', zorder=3, linespacing=1.45)
    else:
        ax.text(x + w / 2, cy, title, ha='center', va='center',
                fontsize=tsize, fontweight='bold', color=INK, zorder=3,
                linespacing=1.45)


def arrow(ax, p0, p1, lw=1.6, color=ARROW, rad=0.0, z=1):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=13,
                        linewidth=lw, color=color,
                        connectionstyle=f'arc3,rad={rad}', zorder=z,
                        shrinkA=1, shrinkB=1)
    ax.add_patch(a)


def label(ax, x, y, s, size=8.5, color='#666666', ha='center', va='center',
          weight='normal', style='normal', z=4):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, fontstyle=style, zorder=z, linespacing=1.4)


def main(out_path):
    plt.rcParams['font.family'] = pick_cjk_font()
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(19.0, 11.0), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ================= 标题 =================
    ax.text(0.5, 0.972, '双模态裂缝检测基线：RGB + LiDAR attrmap',
            ha='center', va='center', fontsize=21, fontweight='bold', color='#1F2933')
    ax.text(0.5, 0.936,
            'YOLO11l backbone（冻结）· token 级交叉注意力融合 · Detect(nc=1) · imgsz 640',
            ha='center', va='center', fontsize=12.5, color='#52606D')

    # ================= 图例 =================
    lx, ly = 0.712, 0.905
    for i, (kind, txt) in enumerate([('frozen', '冻结'), ('train', '可训练'),
                                     ('fusion', '融合新增')]):
        rounded(ax, lx + i * 0.092, ly, 0.022, 0.018, kind, lw=1.2)
        label(ax, lx + i * 0.092 + 0.028, ly + 0.009, txt, size=9.5,
              color='#52606D', ha='left')

    # ================= 行 A：图像支 =================
    yA, hA = 0.762, 0.100
    box(ax, 0.020, yA, 0.118, hA, 'input', 'RGB 图像',
        'B×3×640×640\nBGR·letterbox')
    box(ax, 0.168, yA - 0.012, 0.152, hA + 0.024, 'frozen',
        'YOLO11l backbone + neck',
        'COCO 预训练 · 冻结\nP3 256@80²  P4 512@40²  P5 512@20²')
    box(ax, 0.350, yA, 0.108, hA, 'train',
        'per-scale 1×1 Conv',
        '256/512/512 → d=256\n8400 tokens')
    label(ax, 0.404, yA - 0.024, '+ scale_emb(3) + mod_emb(img)', size=8.5)

    # ================= 行 B：点云支 =================
    yB, hB = 0.540, 0.102
    box(ax, 0.020, yB, 0.118, hB, 'input', 'LiDAR attrmap',
        'B×3×720×1280\ndepth / reflec / height')
    box(ax, 0.168, yB - 0.010, 0.152, hB + 0.020, 'train',
        'PointCloud CNN',
        '4× Conv3x3 s2: 32/64/128/256\n可训练 · 输出 45×80')
    box(ax, 0.350, yB, 0.108, hB, 'train',
        'tokens 3600×256',
        'flatten(2).transpose\n+ mod_emb(pc)')

    # ================= 融合面板 =================
    px, py, pw, ph = 0.497, 0.398, 0.196, 0.480
    rounded(ax, px, py, pw, ph, 'panel', lw=2.0)
    ax.text(px + pw / 2, py + ph - 0.030, 'CrossModalFusion（可训练）',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='#8A4A0B')
    ax.text(px + pw / 2, py + ph - 0.058, 'd = 256 · 8 heads · FFN 1024',
            ha='center', va='center', fontsize=9.5, color='#A05A1B')

    fx, fw = px + 0.012, pw - 0.024

    box(ax, fx, 0.724, fw, 0.076, 'fusion',
        'Q = concat[img 8400 | pc 3600]',
        '12000 × 256   (图像在前)')
    box(ax, fx, 0.598, fw, 0.096, 'fusion',
        'Pre-LN 多头交叉注意力',
        'Q = 12000 tokens；K/V = 2 个可学习上下文 token\n注意力矩阵 8×12000×2（线性于 token 数）')
    box(ax, fx, 0.500, fw, 0.068, 'fusion',
        '残差 + Pre-LN FFN + 残差',
        'Attention → +Q ；FFN(1024) → +x')
    box(ax, fx, 0.420, fw, 0.052, 'fusion',
        'split → 图像 tokens 8400',
        "→ P3' 256@80² / P4' 256@40² / P5' 256@20²")
    label(ax, px + pw / 2, py - 0.020,
          '点云以 token 形式进 Q；不做 12000² 自注意力',
          size=8.2, color='#A05A1B', style='italic')

    # ================= 检测头 =================
    hx, hw = 0.735, 0.170
    box(ax, hx, 0.640, hw, 0.086, 'train',
        'head_proj（1×1 Conv ×3）',
        '256 → 256 / 512 / 512\n投影回 YOLO neck 通道')
    box(ax, hx, 0.500, hw, 0.100, 'head',
        'Detect head  nc = 1',
        '新建头（预训练头 nc=80 不匹配）\nstride 8/16/32 · DFL reg_max=16')
    box(ax, hx, 0.376, hw, 0.088, 'out',
        '输出：crack 框 + 分数',
        '解码 → NMS（评测 IoU 0.70）')
    box(ax, hx, 0.262, hw, 0.080, 'out',
        '损失：v8DetectionLoss',
        'TAL 分配 + box 7.5 / cls 0.5 / dfl 1.5')

    # ================= 箭头 =================
    # 图像支
    arrow(ax, (0.138, yA + hA / 2), (0.168, yA + hA / 2))
    arrow(ax, (0.320, yA + hA / 2), (0.350, yA + hA / 2))
    arrow(ax, (0.458, yA + hA / 2), (fx, 0.762), rad=0.10)
    # 点云支
    arrow(ax, (0.138, yB + hB / 2), (0.168, yB + hB / 2))
    arrow(ax, (0.320, yB + hB / 2), (0.350, yB + hB / 2))
    arrow(ax, (0.458, yB + hB / 2), (fx, 0.742), rad=-0.10)
    # 融合面板内部
    arrow(ax, (px + pw / 2, 0.724), (px + pw / 2, 0.694))
    arrow(ax, (px + pw / 2, 0.598), (px + pw / 2, 0.568))
    arrow(ax, (px + pw / 2, 0.500), (px + pw / 2, 0.472))
    # 融合 → head_proj
    arrow(ax, (fx + fw, 0.446), (hx, 0.683), rad=-0.10)
    # 检测链路
    arrow(ax, (hx + hw / 2, 0.640), (hx + hw / 2, 0.600))
    arrow(ax, (hx + hw / 2, 0.500), (hx + hw / 2, 0.464))
    arrow(ax, (hx + hw / 2, 0.376), (hx + hw / 2, 0.342))

    # ================= 左下：关键设计 =================
    nx, ny, nw, nh = 0.020, 0.132, 0.450, 0.298
    rounded(ax, nx, ny, nw, nh, 'note', lw=1.3)
    label(ax, nx + 0.014, ny + nh - 0.028, '关键设计决策',
          size=11, color='#1F2933', ha='left', weight='bold')
    notes = [
        ('① 图像主干冻结', '624 帧小数据，端到端微调会冲掉 COCO 迁移特征'),
        ('② 点云投影成 attrmap', '与图像共享像素网格，才能做 token 级对齐'),
        ('③ K/V 用 2 个上下文 token', '替代 12000² 自注意力（约 5.8 GB）'),
        ('④ 检测头新建 nc=1', '复用 v8DetectionLoss（TAL + DFL），不搬 COCO 头'),
    ]
    for i, (t, d) in enumerate(notes):
        yy = ny + nh - 0.068 - i * 0.058
        label(ax, nx + 0.016, yy, t, size=10, color='#2E6DA4', ha='left',
              weight='bold')
        label(ax, nx + 0.016, yy - 0.024, d, size=8.8, color='#555555', ha='left')

    # ================= 消融两档（底部） =================
    ay = 0.020
    ax.add_patch(FancyBboxPatch((0.020, ay - 0.010), 0.960, 0.098,
                                boxstyle='round,pad=0.006,rounding_size=0.012',
                                linewidth=1.2, edgecolor='#CBD2D9',
                                facecolor='#F7F9FA', zorder=1))
    label(ax, 0.032, ay + 0.070, '消融两档（共享同一套代码 / 权重初始化 / 训练超参）',
          size=10.5, color='#1F2933', ha='left', weight='bold')

    cards = [
        ('A · rgb', '图像单支', '跳过融合面板，scale_proj 特征直通 head_proj', 'frozen'),
        ('B · rgb+pc', '图像 + 点云（基线）',
         'Q = 图像+点云 tokens；K/V = 2 个可学习上下文 token', 'train'),
    ]
    for i, (tag, name, desc, kind) in enumerate(cards):
        cx = 0.034 + i * 0.478
        rounded(ax, cx, ay - 0.004, 0.460, 0.052, kind, lw=1.4)
        label(ax, cx + 0.012, ay + 0.036, tag, size=9.5, ha='left',
              color='#52606D', weight='bold')
        label(ax, cx + 0.095, ay + 0.036, name, size=11.5, ha='left',
              color=INK, weight='bold')
        label(ax, cx + 0.012, ay + 0.012, desc, size=8.8, ha='left',
              color='#555555')

    label(ax, 0.980, ay + 0.070,
          '数据：output_crack_big · 624 帧配对（image / attrmap / label）· nc=1 · 负样本 25.6%',
          size=8.5, ha='right', color='#7B8794')

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor='white', bbox_inches='tight')
    print('saved:', out_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
