#!/usr/bin/env python3
"""分块 + 梯度检查点与整批计算必须等价（前向输出、反向梯度）。"""

import torch

from tunnel_tri.fusion import CrossModalFusion
from tunnel_tri import config as C


def make(chunk, dropout=0.0):
    torch.manual_seed(0)
    m = CrossModalFusion(dropout=dropout, chunk=chunk, device='cpu')
    # 让 K/V 与投影偏离初始化，避免恒等掩盖差异
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


def run(chunk, train):
    torch.manual_seed(1)
    img = torch.randn(2, C.N_IMG, C.D_MODEL)
    pc = torch.randn(2, C.N_PC, C.D_MODEL)
    m = make(chunk)
    m.train(train)
    out = m(img, pc)
    loss = out.tokens.square().mean() + out.image_scales['P3'].square().sum() * 1e-4
    loss.backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return out.tokens.detach(), loss.item(), grads


def cmp(a, b, what):
    d = (a - b).abs().max().item()
    print(f'  {what}: max|Δ| = {d:.3e}')
    assert d < 1e-5, f'{what} 不等价: {d}'


for train in (False, True):
    print(f'[{"train" if train else "eval"}]')
    t_full, l_full, g_full = run(C.N_IMG + C.N_PC, train)
    t_blk, l_blk, g_blk = run(2048, train)
    cmp(t_full, t_blk, 'tokens')
    assert abs(l_full - l_blk) < 1e-5, (l_full, l_blk)
    assert set(g_full) == set(g_blk)
    for n in g_full:
        d = (g_full[n] - g_blk[n]).abs().max().item()
        scale = g_full[n].abs().max().item()
        rel = d / scale if scale > 0 else d
        assert d < 1e-6 or rel < 1e-5, f'{n} 梯度不等价: abs={d} rel={rel}'
    print(f'  梯度 {len(g_full)} 个张量全部一致')

print('FUSION CHUNK EQUIVALENT OK')
