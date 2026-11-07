#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PQA 搜索损失的权重超参 (γ, β) —— 伪量子退火 × RangeQFL
================================================================
能量函数 = 探针训练后的**真留出集连续 AP** 的负值。每个候选都要真训一遍,
所以: (a) 结果按比特串落盘缓存, 重启不重训; (b) 预算必须小。

★ 关键前提: 探针必须**在短预算下就能排出正确名次**, 否则整个搜索是在优化噪声。
   先用 --check-proxy 验这一点: 拿 (γ=0,β=0)=原版 与 (γ=2,β=0.5)=已训完的
   RangeQFL 两个锚点各跑一次短探针, 看短探针的排名是否与 60 epoch 全量训练一致。
   不一致 ⇒ 探针不可用, PQA 的结果不能当证据。

用法:
  # 1. 验探针可信度 (2 次训练)
  .venv/bin/python fusion/pqa_search.py --check-proxy --epochs 8
  # 2. 跑退火搜索
  .venv/bin/python fusion/pqa_search.py --epochs 8 --budget 8 --steps 10
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import sys

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import numpy as np  # noqa: E402

from fusion.pseudo_quantum_anneal import (  # noqa: E402
    BitParamSpace, CachedEnergy, PseudoQuantumAnnealer,
)

IMG_CKPT = os.environ.get(
    'IMG_CKPT',
    os.path.join(_DIR, 'runs', 'yolo11l_crack_1cls', 'weights', 'best.pt'),
)
DATA = 'data_gen/output_crack_big'
PY = os.path.join(_DIR, '.venv', 'bin', 'python')

SPACE = BitParamSpace({'gamma': [0.0, 1.0, 2.0, 3.0],
                       'beta': [0.0, 0.25, 0.5, 0.75]})
ROOT = os.path.join(_DIR, 'runs', 'pqa')

# 已知参照: 60 epoch 全量训练、heldout(帧0-123) 连续 AP
FULL_RUN_AP = {(0.0, 0.0): 0.8869, (2.0, 0.5): 0.9152}


def probe_train(gamma: float, beta: float, epochs: int, out: str) -> str:
    """跑一次探针训练, 返回 ckpt 路径。已存在则跳过。"""
    ck = os.path.join(out, 'last.pt')
    if os.path.exists(ck):
        print(f'[probe] 复用已有 {ck}')
        return ck
    os.makedirs(out, exist_ok=True)
    cmd = [PY, 'fusion/train_assist.py',
           '--epochs', str(epochs), '--batch', '8', '--seed', '42', '--val-ratio', '0.2',
           '--loss', 'range-qfl', '--qfl-gamma', str(gamma), '--qfl-beta', str(beta),
           '--img-ckpt', IMG_CKPT,
           '--point-ckpt', 'runs/attrmap_ae_crack/point_encoder.pt',
           '--data', DATA, '--out', out]
    log = os.path.join(out, 'probe.log')
    for attempt in (1, 2):
        print(f'[probe] 训练 gamma={gamma} beta={beta} epochs={epochs} -> {out}'
              + (f' (重试 {attempt})' if attempt > 1 else ''))
        with open(log, 'w') as f:
            r = subprocess.run(cmd, cwd=_DIR, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            return ck
        tail = open(log, errors='ignore').read()[-800:]
        # 长时连续训练后偶发 CUDA device-level 故障(unspecified launch failure),
        # 不是 OOM 也不是逻辑错误; 冷却后重试一次即可, 不值得让整轮搜索作废
        if attempt == 1 and 'CUDA' in tail:
            print('[probe] CUDA 设备级故障, 冷却 30s 后重试一次')
            time.sleep(30)
            continue
        raise RuntimeError(f'探针训练失败 ({r.returncode}), 见 {log}')
    return ck


def eval_ap(ckpt: str) -> float:
    """真留出集(帧 0..123) 连续 AP。"""
    import torch
    sys.path.insert(0, os.path.abspath(os.path.join(_DIR, '..', 'tunnel_crack_mm')))
    from compare_branches import (DATA as TDATA, gt_from_fusion_val,
                                  infer_assist, continuous_ap)
    from fusion.evaluate_pr import (ap_from_curve, curve_for_class, load_model,
                                    per_class_pr)
    n_img = len([f for f in os.listdir(os.path.join(_DIR, TDATA, 'images'))
                 if f.endswith('.jpg')])
    val_idx = list(range(max(1, int(n_img * 0.2))))       # train_assist 的 val 段
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(ckpt, dev)
    preds = infer_assist(model, val_idx, dev)
    gts = gt_from_fusion_val(val_idx)
    det = per_class_pr(preds, gts, 1, 0.5)[0]
    th, prec, rec = curve_for_class(det)
    ap = float(continuous_ap(rec, prec))
    del model
    torch.cuda.empty_cache()
    return ap


def make_objective(epochs: int):
    def evaluate(bits: np.ndarray) -> float:
        p = SPACE.decode(bits)
        key = SPACE.key(bits)
        out = os.path.join(ROOT, f'probe_{key}')
        ck = probe_train(p['gamma'], p['beta'], epochs, out)
        ap = eval_ap(ck)
        print(f'[probe] {SPACE.name(bits)}  AP_cont={ap:.4f}')
        return -ap
    return evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=8, help='探针训练轮数')
    ap.add_argument('--steps', type=int, default=10, help='退火步数')
    ap.add_argument('--budget', type=int, default=8, help='最多真实训练次数')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--T0', type=float, default=0.02,
                    help='初始温度。能量= -AP, 探针实测 AP 差约 0.03, 故 0.02 量级')
    ap.add_argument('--check-proxy', action='store_true',
                    help='只验两个锚点, 比较短探针排名与全量训练排名是否一致')
    a = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)

    evaluate = make_objective(a.epochs)
    energy = CachedEnergy(SPACE, os.path.join(ROOT, 'cache.json'), evaluate)

    if a.check_proxy:
        print('=== 探针可信度检查 (短探针 vs 60ep 全量) ===')
        rows = []
        for g, b in ((0.0, 0.0), (2.0, 0.5)):
            bits = SPACE.encode({'gamma': g, 'beta': b})
            e = energy(bits)
            rows.append((g, b, -e, FULL_RUN_AP[(g, b)]))
        probe_order = sorted(range(len(rows)), key=lambda i: -rows[i][2])
        full_order = sorted(range(len(rows)), key=lambda i: -rows[i][3])
        for g, b, p_, f_ in rows:
            print(f'  gamma={g} beta={b}: 探针AP={p_:.4f}  全量AP={f_:.4f}')
        same = probe_order == full_order
        print(f'  排名一致: {same}  (探针 {probe_order} vs 全量 {full_order})')
        print('  ⇒ ' + ('探针可用, 可以做搜索' if same
                        else '探针排名与全量不符, PQA 结果不能当证据'))
        return

    # 显式给 T0: auto_T0 会额外消耗一次昂贵评估, 预算耗尽时会把温度估到 1e9
    annealer = PseudoQuantumAnnealer(SPACE, steps=a.steps, T0=a.T0, seed=a.seed)

    def budgeted(bits):
        if energy.n_calls >= a.budget:
            # 预算耗尽: 不训练, 给一个惩罚能量(极差), 让退火停在已有最优
            return 1e9
        return energy(bits)

    res = annealer.optimize(budgeted, init={'gamma': 2.0, 'beta': 0.5})
    dst = os.path.join(ROOT, f'pqa_result_s{a.seed}_e{a.epochs}.json')
    with open(dst, 'w') as f:
        json.dump({'best_params': res.best_params,
                   'best_energy': res.best_energy,
                   'best_ap': -res.best_energy,
                   'n_train_calls': energy.n_calls,
                   'n_cache_hits': energy.n_hits,
                   'history': res.history,
                   'full_run_reference': {f'{k[0]},{k[1]}': v
                                          for k, v in FULL_RUN_AP.items()}}, f,
                  ensure_ascii=False, indent=2)
    print(f'\n[pqa] 最优: {res.best_params}  AP_cont={-res.best_energy:.4f}')
    print(f'[pqa] 真实训练 {energy.n_calls} 次, 缓存命中 {energy.n_hits} 次')
    print(f'[pqa] 结果 -> {dst}')


if __name__ == '__main__':
    main()
