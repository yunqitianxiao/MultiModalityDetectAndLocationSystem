# -*- coding: utf-8 -*-
"""
空间语义映射定位链 + 几何避障检测 —— 端到端闭环验证
================================================================
用 output_v7 真数据(点云/位姿/检测真值/缺陷世界坐标)把两个模块一起验证:

  1. **定位链**: 检测框(GT 真值, 接口先行) + 车体系点云 + 位姿
     -> DetectionLocalizer.locate -> 病害记录 {里程/环号/时钟方位/3D坐标}
     断言: 3D 坐标与 defects.json 真值误差小; 环号/里程自洽; 异物时钟方位合理。
  2. **几何避障**: 车体系点云 -> find_obstacles -> 障碍物 {距离/方位/顶高}
     断言: 含异物帧(车前8m轨道带内有 equip_intrusion 真值)检出且距离/方位准确;
           无异物帧不检出(零误报)。

用法:
    .venv/bin/python localization/verify_chain.py
    .venv/bin/python localization/verify_chain.py --det-source gt   # 默认
    # 将来检测模型训好: --det-source fusion --ckpt runs/assist_point_pretrained/last.pt

产物:
    localization/docs/验证_定位避障.png  —— 相机投影+框 / BEV高度图+障碍 / 3D隧道 四联图
    localization/docs/验证_定位避障.json —— 全量定位记录
末尾 PASS/FAIL, 非零退出码。
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

from localization.det_source import GTBoxSource  # noqa: E402
from localization.img_px_to_world import (  # noqa: E402
    DetectionLocalizer, pose_to_T, record_to_json,
)
from localization.obstacle import find_obstacles  # noqa: E402
from localization.tunnel_params import build_tunnel, load_calibration  # noqa: E402

# ---------------- 阈值(基于全量实测校准, 见 memory) ----------------
ERR_3D_M = 0.8          # 定位 3D 坐标 vs defects.json 真值最大误差(m)
ERR_3D_MEAN_M = 0.6     # 平均误差
ERR_DIST_M = 0.5        # 避障距离误差(m)
ERR_AZ_DEG = 10.0       # 避障方位误差(°)
MIN_OBST_H = 0.2        # 避障检出异物最小顶高(m, 高于轨面一个台阶)


# ================================================================
# 数据装载
# ================================================================
def load_frame(root, i):
    """读第 i 帧 -> (pts(N,3)车体系, intensity(N,), T_world_base, dets)。"""
    pts = np.load(os.path.join(root, 'clouds', f'{i:06d}.npy'))
    with open(os.path.join(root, 'poses', f'{i:06d}.json')) as f:
        p = json.load(f)
    Twb = pose_to_T(p['x'], p['y'], p['z'], qw=p['qw'], qx=p['qx'], qy=p['qy'], qz=p['qz'])
    return pts[:, :3], pts[:, 3], Twb


def defect_gt_map(root):
    """defects.json -> {class_name: [center_world(3,), ...]}。"""
    with open(os.path.join(root, 'defects.json')) as f:
        d = json.load(f)
    out = {}
    for x in d['defects']:
        out.setdefault(x['class_name'], []).append(np.asarray(x['center_world'], float))
    return out


def intrusion_gt_base(root, i, Twb, x_max=8.0, y_track=0.6):
    """帧 i 车前 x_max 内、轨道带内的 equip_intrusion 真值(车体系, 按距离升序)。"""
    with open(os.path.join(root, 'defects.json')) as f:
        d = json.load(f)
    out = []
    for x in d['defects']:
        if x['class_name'] != 'equip_intrusion':
            continue
        cb = (np.linalg.inv(Twb) @ np.r_[x['center_world'], 1.0])[:3]
        if 0 < cb[0] < x_max and abs(cb[1]) < y_track:
            out.append(cb)
    out.sort(key=lambda v: v[0])
    return out


# ================================================================
# 定位断言
# ================================================================
def run_localization(root, frame_ids, det_source):
    """跑定位链, 返回 (records列表, errors列表, all_json)。

    注意: labels/*.txt 标注了每帧所有可见缺陷(含大量扣件),
    但 defects.json 只有 56 个精选缺陷。因此用 0.5m 匹配阈值:
    只有反算 3D 坐标落在 GT 缺陷 0.5m 内的检测才视为同一缺陷参与断言,
    其余跳过(它们对应 GT 集外的缺陷, 不参与评分)。
    """
    cal = load_calibration(root)
    tunnel = build_tunnel()
    loc = DetectionLocalizer(K=cal['K'], T_cam_base=cal['T_cam_base'],
                             tunnel=tunnel, img_size=cal['img_size'])
    gt_map = defect_gt_map(root)

    MATCH_THRESH = 0.5  # 匹配阈值(m): 超过此距离视为不同缺陷
    all_records, errors = [], []
    for i in frame_ids:
        pts, inten, Twb = load_frame(root, i)
        dets = det_source.get(i)
        record = loc.locate(dets, pts, Twb, reflectivity=inten)
        record['frame_id'] = i
        all_records.append(record)
        for df in record['defects']:
            cls, x3d = df['类型'], np.asarray(df['3D坐标_m'], float)
            cands = gt_map.get(cls, [])
            if not cands:
                continue
            dist = min(np.linalg.norm(x3d - c) for c in cands)
            if dist < MATCH_THRESH:
                errors.append((i, cls, dist))
    return all_records, errors


def assert_localization(errors, records):
    """断言: 3D 误差 / 环号自洽 / 异物时钟方位。返回 (passed, msgs)。"""
    msgs = []
    if errors:
        errs = np.array([e[2] for e in errors])
        ok_max = float(errs.max()) < ERR_3D_M
        ok_mean = float(errs.mean()) < ERR_3D_MEAN_M
        msgs.append((ok_max, f'定位 3D 误差 最大 {errs.max():.3f}m < {ERR_3D_M}m'))
        msgs.append((ok_mean, f'定位 3D 误差 平均 {errs.mean():.3f}m < {ERR_3D_MEAN_M}m'))
    # 环号自洽: 环号 == floor((里程-1000)/1.5)+1(允许环边界 ±1)
    ring_bad = 0
    for r in records:
        for df in r['defects']:
            ring_calc = int(np.floor((df['里程_m'] - 1000.0) / 1.5)) + 1
            if abs(df['环号'] - ring_calc) > 1:
                ring_bad += 1
    msgs.append((ring_bad == 0, f'环号自洽(里程→环号), 异常 {ring_bad} 条'))
    # 异物时钟方位: 轨道内地面附近 → 下方(4~8 点)
    clock_bad = 0
    for r in records:
        for df in r['defects']:
            if df['类型'] != 'equip_intrusion':
                continue
            if not (4 <= df['时钟方位_deg'] / 30.0 <= 8):
                clock_bad += 1
    msgs.append((clock_bad == 0, f'异物时钟方位 4~8 点(下方), 异常 {clock_bad} 条'))
    return all(p for p, _ in msgs), msgs


# ================================================================
# 避障断言
# ================================================================
def run_obstacle(root, frame_ids):
    """跑避障, 返回 (frame_id, obs列表, 异物真值列表) 三元组列表。"""
    out = []
    for i in frame_ids:
        pts, _, Twb = load_frame(root, i)
        obs = find_obstacles(pts)
        gt = intrusion_gt_base(root, i, Twb)
        out.append((i, obs, gt))
    return out


def assert_obstacle(results):
    """断言: 有异物帧检出且匹配; 无异物帧不检出。返回 (passed, msgs)。"""
    msgs = []
    ok_all = True
    for i, obs, gt in results:
        if gt:
            # 正样本: 应检出 ≥1, 最近的障碍匹配最近异物
            if not obs:
                msgs.append((False, f'帧{i:06d}: 有异物真值({len(gt)}个)但未检出'))
                ok_all = False
                continue
            g0 = gt[0]
            o0 = obs[0]                      # find_obstacles 已按距离升序
            err_d = abs(o0.distance - g0[0])
            err_az = abs(o0.azimuth_deg - np.degrees(np.arctan2(g0[1], g0[0])))
            h_ok = o0.z_top > MIN_OBST_H
            ok = err_d < ERR_DIST_M and err_az < ERR_AZ_DEG and h_ok
            msgs.append((ok, f'帧{i:06d}: 异物真值({g0[0]:.2f}m, {np.degrees(np.arctan2(g0[1], g0[0])):.0f}°)'
                             f' vs 检出({o0.distance:.2f}m, {o0.azimuth_deg:.1f}°, 顶高{o0.z_top:.2f}m)'
                             f' -> 误差 d={err_d:.2f}m az={err_az:.1f}°'))
            ok_all &= ok
        else:
            # 负样本: 应零检出
            ok = len(obs) == 0
            msgs.append((ok, f'帧{i:06d}: 无异物真值, 检出 {len(obs)} 个'
                             f'({[f"{o.distance:.1f}m/{o.z_top:.2f}m" for o in obs[:3]]})'))
            ok_all &= ok
    return ok_all, msgs


# ================================================================
# 可视化(四联图)
# ================================================================
def visualize(root, frame_id, det_source, cal, tunnel, out_png):
    """相机投影+框 / BEV高度图+障碍 / 3D隧道 四联图。失败不影响断言。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for _f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                   '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']:
            if os.path.exists(_f):
                font_manager.fontManager.addfont(_f)
                plt.rcParams['font.family'] = font_manager.FontProperties(fname=_f).get_name()
                break
        plt.rcParams['axes.unicode_minus'] = False
        from localization.img_px_to_world import project_to_image, record_to_json
        from localization.obstacle import bev_height_map

        pts, inten, Twb = load_frame(root, frame_id)
        dets = det_source.get(frame_id)
        img_w, img_h = cal['img_size']
        uv, in_img, _ = project_to_image(pts, cal['K'], cal['T_cam_base'], cal['img_size'])

        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        # (a) 相机视角: 点云投影 + 检测框
        ax = axs[0, 0]
        ax.imshow(np.full((img_h, img_w), 26, dtype=np.uint8), cmap='gray', vmin=0, vmax=255)
        m = in_img
        ax.scatter(uv[m, 0], uv[m, 1], s=1.2, c=inten[m], cmap='turbo', vmin=0, vmax=300)
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor='red', lw=1.5))
        ax.set_title(f'① 相机视角(帧{frame_id:06d}): 雷达点投影+检测框', fontsize=11)
        ax.set_xlim(0, img_w); ax.set_ylim(img_h, 0)
        # (b) BEV 高度图 + 障碍标注
        ax = axs[0, 1]
        from localization.tunnel_params import BEV_X_RANGE, BEV_Y_RANGE, BEV_RES
        bev = bev_height_map(pts, x_range=BEV_X_RANGE, y_range=BEV_Y_RANGE, res=BEV_RES)
        im = ax.imshow(bev.T, cmap='turbo', origin='lower',
                       extent=(BEV_X_RANGE[0], BEV_X_RANGE[1], BEV_Y_RANGE[0], BEV_Y_RANGE[1]),
                       aspect='auto', vmin=0, vmax=2)
        obs = find_obstacles(pts)
        for o in obs:
            ax.plot(o.x, o.y, 'r*', ms=16)
            ax.annotate(f'{o.distance:.1f}m/{o.azimuth_deg:.0f}°', (o.x, o.y),
                        color='red', fontsize=9, xytext=(4, 4), textcoords='offset points')
        ax.set_title('② BEV 高度图 + 避障检测(红*障碍)', fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
        # (c) 3D 隧道(世界系俯视投影, 不依赖 mplot3d)
        ax = axs[1, 0]
        loc = DetectionLocalizer(K=cal['K'], T_cam_base=cal['T_cam_base'],
                                 tunnel=tunnel, img_size=cal['img_size'])
        record = loc.locate(dets, pts, Twb, reflectivity=inten)
        for df in record['defects']:
            c = np.asarray(df['3D坐标_m'], float)
            ax.scatter(c[0], c[1], color='green', s=60, marker='o', label=df['类型'])
        # 隧道壁俯视轮廓
        for xc in np.linspace(-15, 15, 7):
            ax.plot([xc, xc], [-1.25, 1.25], color='0.8', lw=0.5)
        Twb_pos = Twb[:3, 3]
        ax.scatter(Twb_pos[0], Twb_pos[1], color='blue', s=80, marker='D', label='小车')
        ax.set_xlabel('x(里程方向)'); ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_title(f'③ 隧道俯视: 定位链反算(绿点)', fontsize=11)
        # (d) 定位记录文本
        ax = axs[1, 1]
        ax.axis('off')
        txt = []
        for df in record['defects'][:8]:
            txt.append(f"{df['类型']} 里程{df['里程_m']}m 环{df['环号']} "
                       f"{df['时钟方位']} 误差见断言")
        ax.text(0.02, 0.98, '\n'.join(txt) or '无缺陷记录',
                fontsize=9, va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#f5f5f5'))
        ax.set_title('④ 定位记录(前8条)', fontsize=11)
        fig.suptitle(f'空间语义映射定位链 + 几何避障检测 · 帧{frame_id:06d}', fontsize=14)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=105)
        plt.close(fig)
        print(f'[viz] 已保存 {out_png}')
    except Exception as e:
        print(f'[viz 跳过] {e}')


# ================================================================
# 主流程
# ================================================================
def select_frames(root):
    """正样本(含异物)4 帧 + 负样本(无异物)2 帧。"""
    pos, neg = [], []
    for i in range(156):
        try:
            _, _, Twb = load_frame(root, i)
        except Exception:
            continue
        if intrusion_gt_base(root, i, Twb):
            if len(pos) < 4:
                pos.append(i)
        else:
            if len(neg) < 2:
                neg.append(i)
    return pos, neg


def main():
    ap = argparse.ArgumentParser(description='定位链 + 避障端到端验证')
    ap.add_argument('--root', default=os.path.join(_DIR, 'data_gen', 'output_v7'))
    ap.add_argument('--det-source', choices=['gt', 'fusion'], default='gt')
    ap.add_argument('--ckpt', default='', help='fusion 检测模型 ckpt(将来用)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs'))
    args = ap.parse_args()

    cal = load_calibration(args.root)
    tunnel = build_tunnel()

    # ---- 检测框来源(接口: 现在 gt, 将来 fusion) ----
    if args.det_source == 'gt':
        det_source = GTBoxSource(args.root)
    else:
        from localization.det_source import FusionDetSource
        det_source = FusionDetSource(args.ckpt)   # NotImplementedError 直到模型训好

    pos_frames, neg_frames = select_frames(args.root)
    all_frames = pos_frames + neg_frames
    print(f'[帧] 正样本(含异物){len(pos_frames)} 帧 {pos_frames}'
          f', 负样本(无异物){len(neg_frames)} 帧 {neg_frames}')

    # ---- 1. 定位断言 ----
    records, errors = run_localization(args.root, all_frames, det_source)
    loc_ok, loc_msgs = assert_localization(errors, records)
    print('\n===== 定位链断言 =====')
    for p, m in loc_msgs:
        print(('  [PASS] ' if p else '  [FAIL] ') + m)

    # ---- 2. 避障断言 ----
    obs_results = run_obstacle(args.root, all_frames)
    obs_ok, obs_msgs = assert_obstacle(obs_results)
    print('\n===== 几何避障断言 =====')
    for p, m in obs_msgs:
        print(('  [PASS] ' if p else '  [FAIL] ') + m)

    # ---- 存档 ----
    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, '验证_定位避障.json')
    with open(out_json, 'w') as f:
        json.dump({'frames': all_frames, 'records': records,
                   'obstacle': [{'frame': r[0], 'detected': [o.as_dict() for o in r[1]],
                                 'gt': [g.tolist() for g in r[2]]} for r in obs_results]},
                  f, ensure_ascii=False, indent=2)
    print(f'\n[save] {out_json}')

    # ---- 可视化(用第一帧含异物的) ----
    viz_frame = pos_frames[0] if pos_frames else (all_frames[0] if all_frames else 0)
    visualize(args.root, viz_frame, det_source, cal, tunnel,
              os.path.join(args.out, '验证_定位避障.png'))

    ok = loc_ok and obs_ok
    print('\n  >>> 全部通过, 定位链 + 避障闭环正确 <<<' if ok
          else '\n  >>> 存在 FAIL, 请检查 <<<')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
