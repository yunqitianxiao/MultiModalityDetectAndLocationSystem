# -*- coding: utf-8 -*-
"""
合并 6 个评测 JSON (3 支 × {full, ablate}) → 逐支【融合 vs 基线】对照表。
全 帧 同 套 框 (output_big_s3 / output_rail_s1), 同 一 decode, 唯一变量 = 是否注入 attrmap+nav。

说明 基线 定义:
  - 独立 RGB 检测器 best_<支>.pt 在这批融合帧上退化(见 pro3-fusion-recall 备注): [0,255] 饱和/ [0,1] 死。
    故 基线 用【融合架构图像单模态】= 同一 wrapper, 清零 attr+nav (cond=ablate)。
  - 它共享 encoders/adapters/Detect 头/帧/[0,1] 输入, 差异纯粹是 点云+导航 条件 —— 隔离"融合模块"的净增益。
"""
import json, os, sys

_P = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(_P, 'runs')

def load(tag):
    p = os.path.join(RUNS, f'{tag}.json')
    if not os.path.isfile(p):
        return None
    return json.load(open(p, encoding='utf-8'))

def row_metrics(j):
    t = j['total']
    return t

def report(branch, names, full, ab):
    print(f'\n================ 支: {branch}  ================')
    print(f"  test_set = {full['test_set']}   imgsz = {full['imgsz']}   conf = {full['conf']}")
    f, a = full['classes'], ab['classes']
    hdr = f"  {'类别':<12}{'P_full':>8}{'R_full':>8}{'|':>2}{'P_abl':>8}{'R_abl':>8}{'|':>2}{'ΔR':>7}{'ΔP':>7}"
    print(hdr)
    tot_full = row_metrics(full); tot_abl = row_metrics(ab)
    for ci, nm in enumerate(names):
        fk, ak = f.get(nm), a.get(nm)
        if fk is None or ak is None:
            continue
        dR = ak['r'] - fk['r']; dP = ak['p'] - fk['p']
        print(f"  {nm:<12}{fk['p']:>8.3f}{fk['r']:>8.3f}{'|':>2}"
              f"{ak['p']:>8.3f}{ak['r']:>8.3f}{'|':>2}{dR:>+7.3f}{dP:>+7.3f}   "
              f"(FG:{fk['gt']:>3}/{ak['gt']:>3}  TP:{fk['tp']:>3}/{ak['tp']:>3}  F1:{fk['f1']:.3f}/{ak['f1']:.3f})")
    dR = tot_abl['r'] - tot_full['r']; dP = tot_abl['p'] - tot_full['p']
    print(f"  {'整体':<12}{tot_full['p']:>8.3f}{tot_full['r']:>8.3f}{'|':>2}"
          f"{tot_abl['p']:>8.3f}{tot_abl['r']:>8.3f}{'|':>2}{dR:>+7.3f}{dP:>+7.3f}   "
          f"(本支融合 召回短差={tot_full['fn']}/{tot_full['gt']}={tot_full['fn']/max(tot_full['gt'],1):.3f})")

    def cls_sum(j):
        return {c: d for c, d in j['classes'].items()}
    return {'branch': branch, 'full': tot_full, 'ablate': tot_abl, 'classes': {'full': cls_sum(full), 'ablate': cls_sum(ab)}}

def main():
    spec = [
        ('土建', ['裂纹', '渗漏', '剥落'], 'evalfusion_土建_full', 'evallate_土建_ablate'),
        ('轨道', ['断裂', '丢失', '松动歪斜'], 'evalfusion_轨道_full', 'evallate_轨道_ablate'),
        ('设备', ['异物入侵', '管线支架松脱'], 'evalfusion_设备_full', 'evallate_设备_ablate'),
    ]
    out = {}
    for branch, names, ftag, atag in spec:
        f = load(ftag); a = load(atag)
        if f is None or a is None:
            missing = [x for x in (ftag, atag) if not os.path.isfile(os.path.join(RUNS, x + '.json'))]
            print(f'\n[skip] {branch}: 缺 {missing}', file=sys.stderr)
            continue
        out[branch] = report(branch, names, f, a)
    with open(os.path.join(RUNS, 'fusion_vs_baseline.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print('\n[存入] runs/fusion_vs_baseline.json')

if __name__ == '__main__':
    main()
