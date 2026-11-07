# -*- coding: utf-8 -*-
"""
融合检测 → 路径决策解码器 · 离线验证
================================================================
读服务器 dump 的 detections.json(每帧 {cls,class_name,conf,distance,azimuth_deg,bbox}),
按帧序(000000..)喂给【同一个】DecisionDecoder(保留状态锁存, 模拟实时决策), 断言解码行为,
并可视化成 4 张 PNG(原图叠框 / 速度-距离 / 状态时间线 / 俯视BEV)。

验证点(与用户裁决一致):
  1. 状态只可能是 {FREE, APPROACH, INSPECT} —— 无 steer/避让 分支。
  2. 前方存在 conf>=阈值且 |az|<80° 的目标 -> 必是 APPROACH 或 INSPECT(接近-检查), 不会 FREE。
  3. INSPECT: d0<=STOP -> 停(yaw=0); 迟滞锁存沿(上一帧 INSPECT, d0 仍<STOP+HIST)也保持停。
  4. APPROACH: 速度符合减速斜坡(0<=v<=CRUISE); |az|>死区 -> yaw 指向目标(居中, 左正右负)。
  5. 低于类阈值的目标不驱动决策。
  6. 异物类(任意类 id)同样"接近-检查"而非避让【合成断言】。

用法(用 pro/2 的 venv, 有 numpy+matplotlib):
  /home/ylf/workspace/pro/2/.venv/bin/python verify_fusion_decision.py \
      --det /home/ylf/workspace/pro/3/detections_big_s3.json \
      --viz-dir /home/ylf/workspace/pro/3/viz_frames
非零退出码 = 断言失败。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from localization.decision_decoder import DecisionDecoder, Target  # noqa: E402
from localization.tunnel_params import (  # noqa: E402
    AVOID_BODY_FRONT, AVOID_SLOW_DIST, AVOID_STOP_DIST, AVOID_V_CRUISE,
    AVOID_YAW_MAX, AZ_ABEAM_DEG, AZ_DEAD_CENTER_DEG, DEC_CONF_THR,
)

CLASS_NAMES = ['裂纹', '渗漏', '剥落']
COLORS = {0: (0.90, 0.30, 0.30), 1: (0.20, 0.60, 0.90), 2: (0.95, 0.65, 0.15)}
EPS = 1e-6


def _targets(dets):
    out = []
    for x in dets:
        ci = int(x['cls'])
        name = x.get('class_name') or (CLASS_NAMES[ci] if ci < len(CLASS_NAMES) else str(ci))
        out.append(Target(cls=ci, conf=float(x['conf']), distance=float(x['distance']),
                          azimuth_deg=float(x['azimuth_deg']), class_name=name))
    return out


def _ref_front(targets):
    """参照实现(与 DecisionDecoder 同规则): 首个 conf>=thr 且 |az|<80 的目标, 按距离升序。"""
    cands = [t for t in targets if t.conf >= DEC_CONF_THR.get(t.cls, 0.25)
             and abs(t.azimuth_deg) < AZ_ABEAM_DEG]
    if not cands:
        return None
    cands.sort(key=lambda t: t.distance)
    return cands[0]


def verify(dec, targets, frame, fails):
    """单帧断言; 用同一个 dec(保留锁存, 不 reset)。"""
    d = dec.decide(targets)
    ref = _ref_front(targets)

    def chk(name, ok, detail=''):
        if not ok:
            fails.append((frame, name, detail))

    chk('状态合法', d.state in {'FREE', 'APPROACH', 'INSPECT'}, d.state)
    chk('速度范围', 0.0 <= d.target_speed <= AVOID_V_CRUISE + EPS, f'{d.target_speed:.3f}')
    chk('|yaw|上限', abs(d.target_yaw_rate) <= AVOID_YAW_MAX + EPS, f'{d.target_yaw_rate:.3f}')

    if ref is None:
        chk('无目标→FREE', d.state == 'FREE', d.state)
        chk('无目标→巡航', abs(d.target_speed - AVOID_V_CRUISE) < EPS, f'{d.target_speed:.3f}')
        chk('无目标→yaw0', abs(d.target_yaw_rate) < EPS, f'{d.target_yaw_rate:.4f}')
        return d, ref

    chk('有目标→非FREE', d.state != 'FREE', d.state)
    d0 = ref.distance - AVOID_BODY_FRONT
    if d0 <= AVOID_STOP_DIST + 1e-9:
        chk('近距→INSPECT', d.state == 'INSPECT', f'{d.state} d0={d0:.2f}')
        chk('INSPECT→停', abs(d.target_speed) < EPS, f'{d.target_speed:.3f}')
        chk('INSPECT→yaw0', abs(d.target_yaw_rate) < EPS, f'{d.target_yaw_rate:.4f}')
    elif d.state == 'INSPECT':
        # 迟滞锁存沿: 上一帧 INSPECT, 本帧 d0 仍 < STOP+HIST(尚未恢复)
        chk('锁存INSPECT→停', abs(d.target_speed) < EPS, f'{d.target_speed:.3f}')
        chk('锁存INSPECT→yaw0', abs(d.target_yaw_rate) < EPS, f'{d.target_yaw_rate:.4f}')
    else:
        chk('中远距→APPROACH', d.state == 'APPROACH', f'{d.state} d0={d0:.2f}')
        exp = (AVOID_V_CRUISE * (d0 - AVOID_STOP_DIST) / (AVOID_SLOW_DIST - AVOID_STOP_DIST)
               if d0 < AVOID_SLOW_DIST else AVOID_V_CRUISE)
        exp = float(min(AVOID_V_CRUISE, max(0.0, exp)))
        chk('斜坡速度', abs(d.target_speed - exp) < 1e-3, f'{d.target_speed:.3f} vs {exp:.3f}')
        if abs(ref.azimuth_deg) > AZ_DEAD_CENTER_DEG:
            chk('转向朝目标', abs(d.target_yaw_rate) > EPS and
                np.sign(d.target_yaw_rate) == np.sign(ref.azimuth_deg),
                f'yaw={d.target_yaw_rate:.4f} az={ref.azimuth_deg:.1f}')
        else:
            chk('正前→不转', abs(d.target_yaw_rate) < EPS, f'{d.target_yaw_rate:.4f}')
    return d, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default='/home/ylf/workspace/pro/3/detections_big_s3.json')
    ap.add_argument('--viz-dir', default='/home/ylf/workspace/pro/3/viz_frames')
    args = ap.parse_args()

    with open(args.det) as f:
        data = json.load(f)
    stems = sorted(data.keys(), key=lambda s: int(s))
    print(f'[det] {args.det}: {len(stems)} 帧')

    # 真实序列: 一个 decoder 跨帧保留锁存
    dec = DecisionDecoder(class_names=CLASS_NAMES)
    fails, rows = [], []
    state_counts = {'FREE': 0, 'APPROACH': 0, 'INSPECT': 0}
    for stem in stems:
        targets = _targets(data[stem] or [])
        d, ref = verify(dec, targets, stem, fails)
        rows.append((stem, d, ref))
        state_counts[d.state] += 1

    # ---------------- 合成"异物不避让"断言 ----------------
    syn_fails = []

    def synth(name, ok, detail=''):
        if not ok:
            syn_fails.append((name, detail))

    decs2 = DecisionDecoder(class_names=CLASS_NAMES)
    r = decs2.decide([Target(cls=0, conf=0.9, distance=0.6, azimuth_deg=0.0, class_name='异物')])
    synth('异物近距→INSPECT(检查非避让)', r.state == 'INSPECT', r.state)
    r = decs2.decide([Target(cls=0, conf=0.9, distance=2.0, azimuth_deg=20.0, class_name='异物')])
    synth('异物中距→APPROACH(不避让)', r.state == 'APPROACH', r.state)
    synth('异物中距→转向朝目标(避让则反号)', r.target_yaw_rate > 0.0, f'{r.target_yaw_rate:.3f}')

    # ---------------- 报告 ----------------
    print('\n===== 路径决策解码器 · 离线验证 =====')
    print(f'帧数 {len(stems)} | 状态 {state_counts} | 断言失败 {len(fails)} 处')
    for f in fails:
        print(f'  [FAIL] {f[0]} :: {f[1]} -> {f[2]}')
    for s in syn_fails:
        print(f'  [SYN-FAIL] {s[0]} -> {s[1]}')
    print(f'>>> {"全部通过" if not fails and not syn_fails else f"失败 {len(fails)}+{len(syn_fails)} 处"}')

    step = max(1, len(rows) // 15)
    print('\n=== 决策抽样 ===')
    for stem, d, ref in rows[::step]:
        fn = f"{ref.class_name}@{ref.distance:.2f}m/{ref.azimuth_deg:+.0f}°" if ref else '无'
        sp = '巡航' if d.state == 'FREE' else ('停' if d.state == 'INSPECT' else f'{d.target_speed:.2f}')
        print(f'  {stem}  {d.state:8s} v={sp:>6s} yaw={d.target_yaw_rate:+.3f}  front={fn}')

    # ---------------- 可视化 ----------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP',
                                                  'WenQuanYi Zen Hei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from PIL import Image

        os.makedirs(args.viz_dir, exist_ok=True)

        # Fig1: 原图+框 4 联
        pick = ['000000', '000001', '000101', '000135']
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for ax, stem in zip(axes.ravel(), pick):
            ip = os.path.join(args.viz_dir, f'{stem}.jpg')
            if not os.path.isfile(ip):
                ax.axis('off'); ax.set_title(f'{stem} (无图)'); continue
            ax.imshow(np.asarray(Image.open(ip).convert('RGB')))
            for x in data[stem]:
                x1, y1, x2, y2 = x['bbox']
                c = COLORS.get(int(x['cls']), (0.3, 0.3, 0.3))
                ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                                edgecolor=c, lw=1.6))
                ax.text(x1, max(0, y1 - 5), f"{x['class_name']} {x['conf']:.2f}", color=c, fontsize=8)
            dd = next(r for r in rows if r[0] == stem)
            ax.set_title(f'{stem}  {dd[1].state}  v={dd[1].target_speed:.2f} '
                         f'yaw={dd[1].target_yaw_rate:+.2f}', fontsize=9)
            ax.axis('off')
        fig.tight_layout()
        p1 = os.path.join(args.viz_dir, 'fig_overlay.png')
        fig.savefig(p1, dpi=110, bbox_inches='tight'); plt.close(fig)

        # Fig2: 速度-目标距离 曲线 + 减速斜坡
        fig, ax = plt.subplots(figsize=(8, 5))
        good = [(r[2].distance, r[1].target_speed) for r in rows if r[2] is not None]
        if good:
            ds, vs = zip(*good)
            ax.scatter(ds, vs, s=18, c='#2a6fbb', alpha=0.6, label='真实帧决策')
        d = np.linspace(AVOID_BODY_FRONT + 0.05, 14, 400)
        d0 = d - AVOID_BODY_FRONT
        exp = np.clip(AVOID_V_CRUISE * (d0 - AVOID_STOP_DIST) / (AVOID_SLOW_DIST - AVOID_STOP_DIST), 0, AVOID_V_CRUISE)
        ax.plot(d, exp, 'r-', lw=2, label='减速斜坡理论')
        ax.axvline(AVOID_BODY_FRONT + AVOID_STOP_DIST, color='r', ls='--', lw=1, label='INSPECT 区界')
        ax.set_xlabel('前方目标距离 (m)'); ax.set_ylabel('前进速度 v (m/s)')
        ax.set_title('速度-目标距离 (接近-检查)'); ax.legend(); ax.grid(alpha=0.3)
        p2 = os.path.join(args.viz_dir, 'fig_speed_curve.png')
        fig.savefig(p2, dpi=110, bbox_inches='tight'); plt.close(fig)

        # Fig3: 状态时间线
        fig, ax = plt.subplots(figsize=(12, 3.2))
        smap = {'FREE': 0, 'APPROACH': 1, 'INSPECT': 2}
        cmap = ['#2a6fbb', '#e0a800', '#c0392b']
        ys = [smap[r[1].state] for r in rows]
        ax.scatter(list(range(len(rows))), ys, c=[cmap[i] for i in ys], s=16)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['FREE', 'APPROACH', 'INSPECT'])
        ax.set_xlabel('帧序'); ax.set_title('状态时间线'); ax.grid(axis='x', alpha=0.2)
        p3 = os.path.join(args.viz_dir, 'fig_timeline.png')
        fig.savefig(p3, dpi=110, bbox_inches='tight'); plt.close(fig)

        # Fig4: 俯视 BEV(车 + 前方目标点 + 演示帧决策向量)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.add_patch(mpatches.Rectangle((-AVOID_BODY_FRONT, -0.175), AVOID_BODY_FRONT * 2, 0.35,
                                        fill=False, edgecolor='k', lw=2, label='车(base)'))
        demo = next((r for r in rows if r[0] == '000135'), None)
        for _, d, ref in rows:
            if ref is None:
                continue
            ex = ref.distance * np.cos(np.radians(ref.azimuth_deg))
            ey = ref.distance * np.sin(np.radians(ref.azimuth_deg))
            c = COLORS.get(ref.cls, (0.5, 0.5, 0.5))
            ax.plot(ex, ey, 'o', color=c, ms=9 if ref.conf > 0.6 else 5)
        if demo and demo[2]:
            ref = demo[2]
            ex = ref.distance * np.cos(np.radians(ref.azimuth_deg))
            ey = ref.distance * np.sin(np.radians(ref.azimuth_deg))
            ax.plot([0, ex], [0, ey], 'k-', lw=1.5, alpha=0.4)
            ax.plot([0, 1.0 * np.sin(np.radians(max(8.0, ref.azimuth_deg)))],
                    [0, 1.0 * np.cos(np.radians(max(8.0, ref.azimuth_deg)))],
                    'r-', lw=2, label='决策转向分量')
        ax.set_xlim(-0.5, 9); ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('前方 x (m)'); ax.set_ylabel('横向 y (m)')
        ax.set_title('俯视: 前方目标 + 车(左正右负方位)'); ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3); ax.set_aspect('equal')
        p4 = os.path.join(args.viz_dir, 'fig_bev.png')
        fig.savefig(p4, dpi=110, bbox_inches='tight'); plt.close(fig)
        print(f'\n[图] {p1}\n     {p2}\n     {p3}\n     {p4}')
    except Exception as e:
        import traceback
        print(f'\n[图] 生成失败: {e}')
        traceback.print_exc()

    return 0 if not fails and not syn_fails else 1


if __name__ == '__main__':
    raise SystemExit(main())
