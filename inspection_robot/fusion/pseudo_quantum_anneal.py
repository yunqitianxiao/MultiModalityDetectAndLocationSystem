# -*- coding: utf-8 -*-
"""
伪量子退火 (Pseudo-Quantum Annealing, PQA) —— 横向场多比特翻转退火
================================================================================
**边界声明**: 经典机器上跑不出真正的量子隧穿, 这里实现的是"量子启发式退火"
(quantum-inspired annealing), 属于元启发式搜索, 不是量子计算。

模型: 把待优化参数编码成自旋串 s ∈ {-1,+1}^n, 能量 E(s) 由外部黑盒目标给出。
每一步做一次"自旋翻转提议":

    Γ(t) = Γ0 · (1 − t/T)^p            横向场强度, 单调衰减
    p_multi = Γ / (1 + Γ)              多比特(集群/隧穿式)翻转概率
    T(t)   = T0 · (T1/T0)^(t/T)        几何降温

    - 以 p_multi 概率一次翻转 k ≥ 2 个比特  ⇒ 允许跨过局部极小(隧穿式大跳)
    - 否则只翻 1 个比特                      ⇒ 退化成经典模拟退火
    - 接受准则: ΔE ≤ 0 或 U[0,1] < exp(−ΔE / T)

前期 Γ 大 ⇒ 大步探索; 后期 Γ→0、T→T1 ⇒ 局部精细搜索。这一条正是它与经典 SA
的唯一实质区别: **经典 SA 只能单点移动, 这里早期允许整簇同时翻转**。

为什么用在这儿: 检测损失里的 (γ, β) 不在连续空间里平滑可导 —— 目标是"训练完
之后的 AP", 只能通过整段训练观测 ⇒ 黑盒、非凸、带噪、评估昂贵。正是元启发式
适用(也只能用)的形态。**代价**: 每次能量评估 = 一次探针训练, 很贵, 所以必须
缓存 + 小预算。
================================================================================
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np


class BitParamSpace:
    """把若干取有限候选值的参数编码成自旋串, 并支持反解。

    spaces = {'gamma': [0.0, 1.0, 2.0, 3.0], 'beta': [0.0, 0.25, 0.5]}
    n_bits = Σ ceil(log2(len(candidates))) —— 每个参数用定长比特编码,
    解码时对越界码字取模(保证任何比特串都有定义, 退火不会走进无效态)。
    """

    def __init__(self, spaces: dict[str, list]):
        self.names = list(spaces)
        self.values = [list(spaces[n]) for n in self.names]
        self.widths = [max(1, int(np.ceil(np.log2(len(v))))) for v in self.values]
        self.offsets = np.cumsum([0] + self.widths[:-1]).tolist()
        self.n_bits = int(sum(self.widths))

    def decode(self, bits: np.ndarray) -> dict:
        out = {}
        for name, vals, w, off in zip(self.names, self.values, self.widths, self.offsets):
            idx = 0
            for k in range(w):
                if bits[off + k] > 0:            # -1 -> 0, +1 -> 1
                    idx |= 1 << k
            out[name] = vals[idx % len(vals)]
        return out

    def encode(self, values: dict) -> np.ndarray:
        bits = np.ones(self.n_bits, dtype=np.int8)
        for name, vals, w, off in zip(self.names, self.values, self.widths, self.offsets):
            idx = vals.index(values[name])
            for k in range(w):
                bits[off + k] = 1 if (idx >> k) & 1 else -1
        return bits

    def key(self, bits: np.ndarray) -> str:
        return ''.join('1' if b > 0 else '0' for b in bits)

    def name(self, bits: np.ndarray) -> str:
        d = self.decode(bits)
        return '|'.join(f'{k}={v}' for k, v in d.items())


@dataclass
class AnnealResult:
    best_bits: np.ndarray
    best_energy: float
    best_params: dict
    history: list = field(default_factory=list)
    n_evals: int = 0
    from_cache: int = 0


class PseudoQuantumAnnealer:
    """横向场多比特翻转退火 (见模块 docstring)。

    参数:
        T0      初始温度。能量尺度未知时用 auto_T0=True 由首轮探测量级估计
        T1      终止温度
        gamma0  初始横向场强度 (p_multi = Γ/(1+Γ); Γ0=1 时约 50% 提议为多比特)
        p_field 横向场衰减指数 (2 = 二次衰减, 前期大跳更多)
    """

    def __init__(self, space: BitParamSpace, steps: int = 20, T0: float | None = None,
                 T1: float = 0.01, gamma0: float = 1.0, p_field: float = 2.0,
                 seed: int = 0):
        self.space = space
        self.steps = steps
        self.T0 = T0
        self.T1 = T1
        self.gamma0 = gamma0
        self.p_field = p_field
        self.rng = np.random.default_rng(seed)

    def optimize(self, energy, init: dict | None = None,
                 init_state: np.ndarray | None = None,
                 log=print) -> AnnealResult:
        """energy(bits: np.ndarray) -> float, 越小越好。会被调用约 steps+1 次。"""
        n = self.space.n_bits
        if init_state is not None:
            s = init_state.copy()
        elif init is not None:
            s = self.space.encode(init)
        else:
            s = self.rng.choice([-1, 1], size=n).astype(np.int8)

        E = float(energy(s))
        best_E, best_s = E, s.copy()
        hist = [{'t': 0, 'state': self.space.name(s), 'E': E, 'accepted': True,
                 'T': None, 'gamma': self.gamma0}]

        T0 = self.T0
        if T0 is None:
            # 用一次随机邻域探测定温度量级, 避免能量尺度未知导致接受率全 0 或全 1
            probe = self._perturb(s, p_multi=0.5)
            dE = abs(float(energy(probe)) - E)
            T0 = max(dE, 1e-6) / 2.0
            log(f'[pqa] auto T0 = {T0:.4f} (由首个邻域 ΔE={dE:.4f} 估计)')

        for t in range(1, self.steps + 1):
            frac = t / self.steps
            T = T0 * (self.T1 / T0) ** frac
            G = self.gamma0 * (1.0 - frac) ** self.p_field
            p_multi = G / (1.0 + G)

            cand = self._perturb(s, p_multi)
            Ec = float(energy(cand))
            dE = Ec - E
            acc = dE <= 0 or self.rng.random() < np.exp(-dE / max(T, 1e-12))
            if acc:
                s, E = cand, Ec
                if E < best_E:
                    best_E, best_s = E, s.copy()
            hist.append({'t': t, 'state': self.space.name(s), 'E': E, 'dE': dE,
                         'accepted': bool(acc), 'T': T, 'gamma': G, 'p_multi': p_multi})
            log(f'[pqa] t={t:02d}/{self.steps} T={T:.4f} Γ={G:.3f} '
                f'{self.space.name(cand)} E={Ec:.4f} dE={dE:+.4f} '
                f'{"accept" if acc else "reject"} | best={best_E:.4f}')

        return AnnealResult(best_bits=best_s, best_energy=best_E,
                            best_params=self.space.decode(best_s), history=hist)

    def _perturb(self, s: np.ndarray, p_multi: float) -> np.ndarray:
        n = len(s)
        if n > 1 and self.rng.random() < p_multi:
            k = int(self.rng.integers(2, n + 1))
            idx = self.rng.choice(n, size=k, replace=False)
        else:
            idx = [int(self.rng.integers(0, n))]
        out = s.copy()
        out[idx] *= -1
        return out


class CachedEnergy:
    """能量函数缓存: 每次评估 = 一次探针训练, 重复状态必须直接命中。

    用法: energy = CachedEnergy(space, path, evaluate_fn); annealer.optimize(energy)
    """

    def __init__(self, space: BitParamSpace, cache_path: str, evaluate_fn):
        self.space = space
        self.cache_path = cache_path
        self.evaluate_fn = evaluate_fn
        self.cache: dict[str, float] = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)
        self.n_calls = 0
        self.n_hits = 0

    def __call__(self, bits: np.ndarray) -> float:
        k = self.space.key(bits)
        if k in self.cache:
            self.n_hits += 1
            print(f'[pqa] (cache) {self.space.name(bits)} -> {self.cache[k]:.4f}')
            return self.cache[k]
        self.n_calls += 1
        v = float(self.evaluate_fn(bits))
        self.cache[k] = v
        self._save()
        return v

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp = self.cache_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.cache_path)
