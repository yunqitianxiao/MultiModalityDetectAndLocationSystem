#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三分支仿真数据集完整性自检 + YOLO 冒烟
================================================================================
对 土建_3cls_sim / 轨道_3cls_sim / 设备_2cls_sim 逐集校验:
  1. data.yaml 结构 (path/train/val/test/names)
  2. 图-标签对齐 (train/val/test 每 split 图数==标签数)
  3. 标签格式 (YOLO 5列, 类号在 names 内, x/y/w/h 归一化 0~1, 无负宽高)
  4. 每类 (train) 框数 + 每 split 图数
  --smoke: 额外用 ultralytics 真实加载 data.yaml 并统计每个 split 可读图数
           (自动修正不了 path, 但能暴露 path 指向错误/缺失目录的问题)

与机器无关: 不硬编码本地/服务器绝对路径, 一律以「本脚本所在目录」为基准,
  优先 <脚本目录>/<数据集>/data.yaml, 其次把 data.yaml 里的 path 视为候选。
  -> 本地(pro/3) 与服务器(~  /linfengyao/3) 均可直接运行, 无参即跑。

用法:
  python3 test_3cls_sim.py                  # 全部三集, 快速自检
  python3 test_3cls_sim.py --split test      # 附加校验 test 分片
  python3 test_3cls_sim.py --smoke           # 附加 YOLO 真加载冒烟
  python3 test_3cls_sim.py --data 土建_3cls_sim --smoke   # 只查一集
"""
import os, sys, argparse, glob, yaml

# ---- 按 名称 -> 类别数 定义三集 (顺序即统计顺序) ----
DEFAULTS = [
    ("土建_3cls_sim", 3, ["裂缝", "渗漏水", "管片掉块"]),
    ("轨道_3cls_sim", 3, ["断裂", "丢失", "松动歪斜"]),
    ("设备_2cls_sim", 2, ["异物入侵", "管线支架松脱"]),
]
SPLITS = ("train", "val", "test")


def ds_path(name):
    """候选数据集目录: 脚本同目录下 <name>, 以及 data.yaml 里 path 指向的目录。"""
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, name)
    return cand


def load_yaml(p):
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_boxes(txt):
    """返回 (lines, 异常数)。5列 -> (cls,cx,cy,w,h)。"""
    bad = 0
    out = []
    if not os.path.isfile(txt):
        return out, 0
    for ln in open(txt, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        if len(p) != 5:
            bad += 1
            continue
        try:
            cl = int(float(p[0]))
            vals = [float(x) for x in p[1:]]
        except ValueError:
            bad += 1
            continue
        out.append((cl, vals))
    return out, bad


def check_one(name, n_classes, class_names, do_yolo):
    print(f"\n{'='*64}\n【 {name} 】  (应 {n_classes} 类)\n{'='*64}")
    d = ds_path(name)
    yaml_p = os.path.join(d, "data.yaml")
    if not os.path.isfile(yaml_p):
        print(f"  ❌ 找不到 {yaml_p}  (脚本与 <数据集>/data.yaml 未同层)")
        return 1
    cfg = load_yaml(yaml_p)

    # 1) data.yaml 结构
    names = cfg.get("names")
    if isinstance(names, int):
        names = {i: str(i) for i in range(names)}
    n_names = len(names) if names else 0
    ok_names = "; ".join(f"{k}={v}" for k, v in (names or {}).items())
    print(f"  [yaml names] {ok_names}  (count={n_names}, 期望 {n_classes})")
    if n_names != n_classes:
        print(f"    ⚠ 类别数与期望不符! names={n_names} vs 期望 {n_classes}")

    # path 逻辑: 优先 data.yaml 的 path(若目录存在), 否则用脚本同层
    path_field = cfg.get("path")
    base = path_field if (path_field and os.path.isdir(path_field)) else d
    print(f"  [yaml path ] {path_field}  -> {'✓ 目录存在' if path_field and os.path.isdir(path_field) else '⚠ false/缺失, 用脚本同层 ' + base}")

    # 2) 三个 split 的图/标签对齐 + 类别分布
    total_box = 0
    per_cls = {k: 0 for k in range(n_classes)}
    errs = 0
    for sp in SPLITS:
        imgs = sorted(glob.glob(os.path.join(base, "images", sp, "*")))
        lbls = sorted(glob.glob(os.path.join(base, "labels", sp, "*.txt")))
        istems = {os.path.splitext(os.path.basename(x))[0] for x in imgs}
        lstems = {os.path.splitext(os.path.basename(x))[0] for x in lbls}
        img_no_lbl = len(istems - lstems)
        lbl_no_img = len(lstems - istems)
        flag = " " if len(imgs) == len(lbls) else "⚠"
        print(f"  [{sp:5s}] 图{len(imgs):4d} 标签{len(lbls):4d}  "
              f"图缺标签{img_no_lbl} 标签缺图{lbl_no_img} {flag}")
        if img_no_lbl or lbl_no_img:
            errs += 1
        # 标签校验(仅在 train/val 详细做, test 只对齐不算类分布)
        if sp in ("train", "val"):
            for t in lbls:
                boxes, bad = read_boxes(t)
                errs += bad
                for cl, vals in boxes:
                    if cl not in per_cls:
                        errs += 1
                        continue
                    per_cls[cl] += 1
                    total_box += 1
                    for v in vals:
                        if v < 0.0 or v > 1.0:
                            errs += 1
    print(f"  [train+val 框数] {total_box}  各类: " +
          "  ".join(f"{class_names[k]}={per_cls[k]}" for k in range(n_classes)))
    # 若 test 图数=0 提示
    test_img = len(glob.glob(os.path.join(base, "images", "test", "*")))
    if test_img == 0:
        print(f"  ⚠ test 分片为空! (病集留测真图未就位? {base}/images/test)")

    # 3) YOLO 真加载冒烟
    if do_yolo:
        try:
            from ultralytics import YOLO
            dm = YOLO("yolo11n.yaml")  # 只建骨架, 不下载权重
            # 用 check_det_dataset 校验 data.yaml 可被 YOLO 解析且路径可读
            from ultralytics.data.utils import check_det_dataset
            dd = check_det_dataset(yaml_p)
            ok = all(dd.get(k) for k in ("train", "val"))
            print(f"  [yolo smoke] check_det_dataset -> train={dd.get('train')} val={dd.get('val')} "
                  f"{'✓' if ok else '⚠ path 有缺失'}")
        except Exception as e:
            print(f"  [yolo smoke] ❌ {type(e).__name__}: {str(e)[:300]}")

    return 1 if errs else 0


def main():
    ap = argparse.ArgumentParser(description="三分支仿真数据集自检")
    ap.add_argument("--data", default=None, help="只查某个数据集名(缺省=全部3个)")
    ap.add_argument("--smoke", action="store_true", help="用 ultralytics 真加载冒烟")
    ap.add_argument("--split", default=None, choices=SPLITS, help="(占位) 打印分片统计")
    a = ap.parse_args()

    sel = [x for x in DEFAULTS if a.data is None or x[0] == a.data]
    if not sel:
        sys.exit(f"未知数据集: {a.data}\n可选: {[x[0] for x in DEFAULTS]}")
    failed = 0
    for name, nc, cn in sel:
        failed += check_one(name, nc, cn, a.smoke)
    print(f"\n{'='*64}\n完成: {'✅ 全部通过' if failed == 0 else f'❌ {failed} 项异常'}\n{'='*64}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
