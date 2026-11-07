# -*- coding: utf-8 -*-
"""
三模态融合(单类裂缝) → 决策 → cmd_vel · 离线闭合回环(回放仿真)
================================================================
源自 pro/3/run_fusion_closed_loop.py, 改单类裂缝: CLASS_NAMES=['crack']、
DecisionDecoder 类别数按 len(CLASS_NAMES)。其余逻辑一字未动。

把 dump_detections_crack.py 产出的 detections_crack.json(每帧
{cls,conf,distance,azimuth_deg})与世界系锚定, 再让一个【模拟小车】积分 cmd_vel,
每步用"最近录制帧"当感知, 重新相对化目标的距离/方位 -> DecisionDecoder -> (v,wz)
-> 运动学前进。

不是对录制路径做纯跟随(那种永远复现"全APPROACH"): 世界系锚定后, 车真正按 cmd_vel 走,
若在缺陷附近减速/停, 车会滞留在该帧(感知不变), 呈现真实的接近-检查行为。

用法(本地, inspection_robot/.venv = torch cu130 GPU):
  python -u run_closed_loop_crack.py --det detections_crack.json \
      --poses ../inspection_robot/data_gen/output_crack/poses --conf-thr 0.25 --dt 0.1 --limit 0 \
      --out runs/closed_loop.json --viz runs/closed_loop.png
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

_P = os.path.dirname(os.path.abspath(__file__))
if _P not in sys.path:
    sys.path.insert(0, _P)
_IR = os.environ.get('IR_ROOT', os.path.abspath(os.path.join(_P, '..', 'inspection_robot')))
if _IR not in sys.path:
    sys.path.append(_IR)            # 取 inspection_robot 的 localization/
from localization.decision_decoder import DecisionDecoder, Target  # noqa: E402
from localization.tunnel_params import (  # noqa: E402
    AVOID_BODY_FRONT, AVOID_V_CRUISE, AZ_ABEAM_DEG,
)

CLASS_NAMES = ['crack']
EPS = 1e-6
HOLD_TICKS = 40          # INSPECT 停下查看的 tick 数(@dt=0.1 → 4s)后才标记"已检查"放行


def yaw_from_pose(p):
    qx, qy, qz, qw = float(p.get('qx', 0)), float(p.get('qy', 0)), float(p.get('qz', 0)), float(p.get('qw', 1))
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def load_poses(dirp, stems=None):
    files = sorted(glob.glob(os.path.join(dirp, '*.json')))
    poses = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stems is not None and stem not in stems:
            continue
        p = json.load(open(f))
        poses[stem] = (float(p['x']), float(p['y']), yaw_from_pose(p))
    return poses   # 仅真实帧: stem -> (x, y, yaw)


def anchor_target(t, pose):
    """录制帧车体系(distance/azimuth) -> 世界系锚点。azimuth 左正右负, 前向沿 yaw。"""
    x, y, yaw = pose
    a = math.radians(t.azimuth_deg)
    fwd = t.distance * math.cos(a)          # 前后分量
    left = t.distance * math.sin(a)         # 左右分量(左正)
    wx = x + fwd * math.cos(yaw) - left * math.sin(yaw)
    wy = y + fwd * math.sin(yaw) + left * math.cos(yaw)
    return np.array([wx, wy])


def relativize(wpos, pose):
    """世界点 -> 我的车体系 (fwd, left)。返回 (dist, azimuth_deg)。"""
    x, y, yaw = pose
    dx, dy = wpos[0] - x, wpos[1] - y
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    left = -dx * math.sin(yaw) + dy * math.cos(yaw)
    dist = math.hypot(fwd, left)
    az = math.degrees(math.atan2(left, fwd))
    return dist, az


def nearest_pose(x, y, poses):
    best, bd = None, None
    for stem, (px, py, pyaw) in poses.items():
        if stem.endswith('__stamp'):
            continue
        d = (px - x) ** 2 + (py - y) ** 2
        if bd is None or d < bd:
            bd, best = d, stem
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', default=os.path.join(_P, 'detections_big_s3.json'))
    ap.add_argument('--poses', default=os.path.join(_P, 'pcdata', 'output_big_s3', 'poses'))
    ap.add_argument('--conf-thr', type=float, default=0.25)
    # 裂缝贴在拱顶/墙面上部, 车在地面最近仅能到 ~3.2m(实测), 沿用避障的 0.5m 阈值会使
    # INSPECT 永不触发。改按"可拍摄距离": 真净距 <=3.5m 停车记录、<6m 开始减速。
    ap.add_argument('--stop-dist', type=float, default=3.5, help='前向目标真净距 <= 此值 -> 停车检查')
    ap.add_argument('--slow-dist', type=float, default=6.0, help='真净距 < 此值 -> 减速斜坡')
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--max-steps', type=int, default=4000)
    ap.add_argument('--free-break', type=int, default=600,
                    help='纯 FREE 巡航超过此步数且前方已无未检查缺陷 -> 巡检结束(缺缺陷间隔远时可调大)')
    ap.add_argument('--limit', type=int, default=0, help='>0 只取前 N 帧(探路)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--viz', default=None)
    args = ap.parse_args()

    dets = json.load(open(args.det))
    stems = sorted(dets.keys(), key=lambda s: int(s))
    if args.limit > 0:
        stems = stems[:args.limit]
    poses = load_poses(args.poses, set(stems))
    print(f'[data] {len(stems)} 帧 @ {args.det}', flush=True)

    dec = DecisionDecoder(conf_thr={i: args.conf_thr for i in range(len(CLASS_NAMES))},
                          class_names=CLASS_NAMES,
                          stop_dist=args.stop_dist, slow_dist=args.slow_dist)

    # ---- 世界系锚定感知: stem -> list[np(2) world target, Target] ----
    world_targets = {}
    for stem in stems:
        pose = poses.get(stem)
        if pose is None:
            continue
        tl = []
        for x in (dets[stem] or []):
            t = Target(cls=int(x['cls']), conf=float(x['conf']), distance=float(x['distance']),
                       azimuth_deg=float(x['azimuth_deg']), class_name=x.get('class_name'))
            tl.append((anchor_target(t, pose), t))
        world_targets[stem] = tl

    # ---- 闭环: 车从首帧位姿出发 ----
    start = poses[stems[0]]
    x, y, yaw = float(start[0]), float(start[1]), float(start[2])
    cur_stem = stems[0]
    rows, states = [], {'FREE': 0, 'APPROACH': 0, 'INSPECT': 0}
    v = 0.0
    visited = set()                 # 已检查的世界锚点(放行后不再触发)
    inspecting_anchor = None        # 当前正停下查看的锚点
    hold_ticks = 0
    free_idle = 0                   # 连续纯巡航步数(前方无缺陷) -> 巡检收尾
    last_inspect_step = -10000
    for step in range(args.max_steps):
        # 感知: 用当前车姿相对化"当前帧"的世界目标(当前帧=最近录制帧); 已访问的剔除
        cur_stem = nearest_pose(x, y, poses)
        cur_pairs = []
        for wpos, t in world_targets.get(cur_stem, []):
            key = (round(float(wpos[0]), 2), round(float(wpos[1]), 2), t.cls)
            if key in visited:
                continue
            dist, az = relativize(wpos, (x, y, yaw))
            cur_pairs.append((key, wpos, t, dist, az))
        targets = [Target(cls=t.cls, conf=t.conf, distance=dist, azimuth_deg=az,
                          class_name=t.class_name) for (_, _, t, dist, az) in cur_pairs]
        d = dec.decide(targets, v_now=v, nav_yaw=yaw)
        states[d.state] += 1
        v, wz = d.target_speed, d.target_yaw_rate
        # 运动学(dt)
        x += v * math.cos(yaw) * args.dt
        y += v * math.sin(yaw) * args.dt
        yaw += wz * args.dt
        # 纯巡航收尾: 前方已无未检查缺陷且连续 FREE -> 巡检结束
        free_idle = free_idle + 1 if d.state == 'FREE' else 0
        front = d.front
        # INSPECT 快门: 停下查看 HOLD_TICKS 后标记该锚点已访问 -> 放行到下一缺陷
        if d.state == 'INSPECT' and front is not None:
            if inspecting_anchor is None:
                bestk, bdiff = None, 1e9
                for (key, wpos, t, dist, az) in cur_pairs:
                    if t.cls != front.cls:
                        continue
                    dd = abs(dist - front.distance) + abs(az - front.azimuth_deg)
                    if dd < bdiff:
                        bdiff, bestk = dd, key
                inspecting_anchor = bestk
            hold_ticks += 1
            last_inspect_step = step
            if inspecting_anchor is not None and hold_ticks >= HOLD_TICKS:
                visited.add(inspecting_anchor)
                inspecting_anchor = None; hold_ticks = 0
                dec.reset()          # 清 INSPECT 锁存, 放行
        else:
            inspecting_anchor = None; hold_ticks = 0
        rows.append({'step': step, 'stem': cur_stem, 'state': d.state, 'x': x, 'y': y,
                     'yaw': yaw, 'v': v, 'vz': wz, 'n': len(targets),
                     'reason': d.reason, 'front': (front.class_name if front else None),
                     'fd': (front.distance if front else None),
                     'faz': (front.azimuth_deg if front else None),
                     'n_inspected': len(visited)})
        if (step % 200) == 0:
            print(f'  step {step} stem={cur_stem} {d.state:8s} v={v:.3f} '
                  f'yaw={yaw:+.2f} 已检查={len(visited)} front={d.reason[:24]}', flush=True)
        # 结束: 车遍历完可达缺陷(前方无目标且纯巡航一段) -> 巡检完成; 飞到路径野即收尾
        if free_idle >= args.free_break:
            break

    print(f'\n===== 闭环结束: {len(rows)} 步 =====')
    print(f'状态分布 {states}')
    # 速度剖面
    vs = [r['v'] for r in rows]
    dmin = min(vs)
    cruise_frac = sum(1 for v_ in vs if v_ >= AVOID_V_CRUISE - EPS) / max(1, len(vs))
    print(f'速度: min={dmin:.3f} 巡航占比={cruise_frac:.2f} 平均值={np.mean(vs):.3f}')
    inspects = [r for r in rows if r['state'] == 'INSPECT']
    print(f'INSPECT 步数={len(inspects)} | APPROACH={(len(rows)-len(inspects)-states["FREE"]):.2f}'
          f' | FREE={states["FREE"]}')
    dist = math.hypot(rows[-1]['x'] - rows[0]['x'], rows[-1]['y'] - rows[0]['y'])
    print(f'总位移 {dist:.2f} m | 终点 ({rows[-1]["x"]:.2f},{rows[-1]["y"]:.2f}) yaw={rows[-1]["yaw"]:+.2f}')

    # ---- 抽样决策 ----
    step = max(1, len(rows) // 16)
    print('\n=== 决策抽样 ===')
    for r in rows[::step]:
        fn = f"{r['front']}@{r['fd']:.2f}m/{r['faz']:+.0f}°" if r['front'] else '无'
        sp = '巡航' if r['state'] == 'FREE' else ('停' if r['state'] == 'INSPECT' else f"{r['v']:.2f}")
        print(f"  {r['step']:>4} {r['state']:8s} v={sp:>6s} yaw={r['yaw']:+.3f} x={r['x']:+.2f} "
              f"y={r['y']:+.2f} front={fn}")

    # ---- 可视化 ----
    if args.viz:
        try:
            import matplotlib
            matplotlib.use('Agg')
            matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP',
                                                      'WenQuanYi Zen Hei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 1, figsize=(11, 9))
            smap = {'FREE': 0, 'APPROACH': 1, 'INSPECT': 2}
            cmap = np.array(['#2a6fbb', '#e0a800', '#c0392b'])
            ys = [smap[r['state']] for r in rows]
            xs = [r['x'] for r in rows]; xy = [r['y'] for r in rows]
            # 1) 世界轨迹(按状态上色) + 目标世界点
            ax = axes[0]
            for stem, tl in world_targets.items():
                for wpos, _ in tl:
                    ax.plot(wpos[0], wpos[1], '.', color='#7f7f7f', ms=3, alpha=0.35)
            # 逐段画(单点用 '-' 画不出线, 车轨迹会整条消失): 每步与上一步连一小段, 按本步状态着色
            for i in range(1, len(rows)):
                ax.plot(xs[i - 1:i + 1], xy[i - 1:i + 1], '-',
                        color=cmap[ys[i]], lw=1.4, alpha=0.9)
            ax.plot(xs[0], xy[0], 'ko', ms=6, label='起')
            ax.plot(xs[-1], xy[-1], 'k^', ms=8, label='终')
            ax.set_title('世界轨迹(点=世界系缺陷目标, 线=车, 色=状态)')
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=0.3); ax.set_aspect('equal')
            # 2) 速度-时间
            ax = axes[1]
            ax.plot([r['step'] for r in rows], vs, '-', color='#2a6fbb', lw=1.8)
            ax.axhline(AVOID_V_CRUISE, color='k', ls='--', lw=1, label=f'巡航 {AVOID_V_CRUISE}')
            for r in rows:
                if r['state'] == 'INSPECT':
                    ax.axvline(r['step'], color='#c0392b', alpha=0.25, lw=0.6)
            ax.set_xlabel('step'); ax.set_ylabel('v (m/s)'); ax.set_title('cmd_vel 前进速度')
            ax.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)
            # 3) 状态时间线
            ax = axes[2]
            ax.scatter([r['step'] for r in rows], ys, c=[cmap[i] for i in ys], s=14)
            ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['FREE', 'APPROACH', 'INSPECT'])
            ax.set_xlabel('step'); ax.set_title('决策状态时间线'); ax.grid(axis='x', alpha=0.2)
            fig.tight_layout()
            fig.savefig(args.viz, dpi=110, bbox_inches='tight')
            plt.close(fig)
            print(f'\n[图] {args.viz}')
        except Exception as e:
            import traceback
            print(f'\n[图] 失败: {e}')
            traceback.print_exc()

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'states': states, 'rows': rows,
                       'summary': {'steps': len(rows), 'dmin': float(dmin),
                                   'cruise_frac': float(cruise_frac),
                                   'inspect_steps': len(inspects), 'dist': float(dist)}},
                      f, ensure_ascii=False, indent=2)
        print('[存入]', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
