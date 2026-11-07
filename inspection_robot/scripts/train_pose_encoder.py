#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
位姿 MLP 编码器训练脚本
================================================================
自监督目标: 用当前时刻状态向量预测下一时刻状态向量(时间动力学的代理任务)。
在没有配对图像数据之前, 先用这个任务让编码器学到有意义的位姿动态特征;
以后接多模态融合时, 编码器再随融合头端到端微调。

数据路径:
    1) 录真实数据:
       ros2 run inspection_robot pose_state_node.py --ros-args -p log_csv:=~/pose_state.csv
       ... 遥控小车跑一会(直线+转弯+往返) ...
    2) 训练:
       python3 scripts/train_pose_encoder.py --csv ~/pose_state.csv --out ~/pose_mlp.pt
       (没有真实数据时先用合成轨迹验证链路:
        python3 scripts/train_pose_encoder.py --demo --out ~/pose_mlp.pt)
    3) 部署:
       ros2 run inspection_robot pose_encoder_node.py --ros-args -p ckpt:=~/pose_mlp.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from data_gen.pose_mlp import PoseMLP, save_ckpt, DIM_IN  # noqa: E402

DIM_OUT = DIM_IN  # 预测下一时刻, 输出维度=输入维度


def load_csv(path):
    """读 pose_state_node 落盘的 CSV -> (N, 12) 状态矩阵(去掉第 0 列时间)。"""
    arr = np.loadtxt(path, delimiter=',', skiprows=1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, 1:1 + DIM_IN].astype(np.float32)


def make_demo_data(n=2000, dt=0.05, seed=0):
    """合成"直线 + 缓慢转弯"轨迹, 用于没有真实数据时验证训练链路。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    vx = 0.4 + 0.08 * np.sin(t * 0.3)          # 前进速度小幅波动
    yaw = 0.15 * np.sin(t * 0.2)               # 缓慢摆头
    x = np.cumsum(vx * dt)
    y = 0.1 * np.sin(t * 0.2)
    pos_scale = 10.0
    state = np.stack([
        x / pos_scale, y / pos_scale, np.zeros_like(t),
        np.zeros_like(t), np.ones_like(t),        # roll=0
        np.zeros_like(t), np.ones_like(t),        # pitch=0
        np.sin(yaw), np.cos(yaw),
        vx, np.gradient(y, dt), np.gradient(yaw, dt),
    ], axis=1).astype(np.float32)
    return state


def train(state, epochs=60, lr=1e-3, batch=64, seed=0):
    """自监督: 编码器 + 预测头 拟合状态增量 d[i] = state[i+1] - state[i] (MSE)。
    预测增量而非原始状态: 各通道量级均衡(位置增量~v·dt, 角度增量~w·dt),
    头无需输出原始米数, 编码器被迫编码速度/角速度等动力学。训练完弃预测头。"""
    X = state[:-1]
    Y = state[1:] - state[:-1]
    n = len(X)
    torch.manual_seed(seed)

    model = PoseMLP()
    head = nn.Linear(model.dim_emb, DIM_OUT)   # 128 嵌入 -> 12 预测, 仅训练用
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=lr)
    mse = nn.MSELoss()
    n_batches = max(1, n // batch)

    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for b in range(n_batches):
            idx = perm[b * batch:(b + 1) * batch]
            xb = torch.from_numpy(X[idx])
            yb = torch.from_numpy(Y[idx])
            opt.zero_grad()
            loss = mse(head(model(xb)), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f'epoch {ep + 1:3d}/{epochs}  loss {total / n_batches:.5f}')
    return model


def main():
    ap = argparse.ArgumentParser(description='训练位姿 MLP 编码器(自监督: 预测下一时刻状态)')
    ap.add_argument('--csv', default='', help='pose_state CSV 路径(真实数据)')
    ap.add_argument('--demo', action='store_true', help='用合成轨迹代替真实数据(验证链路)')
    ap.add_argument('--out', default=os.path.expanduser('~/pose_mlp.pt'))
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--dt', type=float, default=0.05)
    args = ap.parse_args()

    if args.demo:
        state = make_demo_data(dt=args.dt)
        print(f'合成数据: {len(state)} 帧')
    elif args.csv and os.path.exists(args.csv):
        state = load_csv(args.csv)
        print(f'加载 CSV: {len(state)} 帧 (取后 {DIM_IN} 列)')
    else:
        ap.error('需要 --csv <文件> 或 --demo')

    model = train(state, epochs=args.epochs)
    save_ckpt(model, args.out)
    print(f'已保存: {args.out} (嵌入维度 {model.dim_emb})')


if __name__ == '__main__':
    main()
