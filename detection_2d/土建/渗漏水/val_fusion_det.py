#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合模型评测(A) —— 在分支验证/测试集上算融合后的 mAP50 / mAP50-95 / 精度 / 召回
================================================================================
【为什么手动驱动】融合包装器不是标准 YOLO, 不能走 model.val()(里面用 AutoBackend 包一层)。
但 A 要和基线 B 同口径比较, 所以这里复用 ultralytics DetectionValidator 的
preprocess / postprocess / update_metrics / get_stats —— 与 model.val() 完全同一套 mAP 计算,
保证 A-B 的差值纯粹来自"额外分支的上下文", 而不是评估口径差异。

【和 train_fusion_det.py 的关系】
  - train:  编码器(冻) → 融合/适配/检测头 微调, 存 wrapper.state_dict()。
  - val:    用 build_model() 重建同结构 wrapper -> load_state_dict(保存的权重) -> 手动前向评测。
  - 前向走 det.eval()+end2end=False: Detect.forward 返回 (y, preds), 取 y=out[0] 喂 postprocess(NMS)。

【用法(服务器 8×4090)】
  # 轨道(权重已跑完, 存 guidao_fusion.pt)
  python val_fusion_det.py --branch 轨道 --split test --imgsz 1024 --batch 2 --device 0
  # 土建/设备 同理(权重 tujian_fusion.pt / shebei_fusion.pt)
  python val_fusion_det.py --branch 土建 --split test --imgsz 1024 --batch 2 --device 1
  # 快速接线检查(只跑前 N 个 batch):
  python val_fusion_det.py --branch 轨道 --imgsz 320 --batch 4 --device cpu --limit 2

【注意】--split 要与基线 B(test_detect.py --split)一致才有可比性; B 默认 test。
"""
from __future__ import annotations

import argparse, os, glob
from pathlib import Path

import torch
import torch.nn as nn

# 复用训练脚本里的构建逻辑, 保证结构/编码器/数据集定位完全一致
from train_fusion_det import _find_repo, _parsed_device, _BRANCHES, build_model

_REPO = _find_repo()

# 各分支数据集目录名(含 data.yaml), 用于自动定位
_BRANCH_DATASETS = {
    "轨道": "轨道_3cls_均衡",
    "土建": "土建_病害_yolo_3cls",
    "设备": "设备_异物支架_合并_yolo",
}


def _fusion_weight(branch: str) -> str:
    """由分支基线权重推导融合权重名: guidao_best.pt -> guidao_fusion.pt。
    去掉 '_best' 后缀(文件名规则是 <分支>_fusion.pt, 与 train --save 一致); 顺带兼容不带 _best 的情况。"""
    conf = _BRANCHES[branch]
    stem = conf["weight"].rsplit(".", 1)[0]                 # guidao_best
    if stem.endswith("_best"):
        stem = stem[: -len("_best")]                        # guidao
    return os.path.join(_REPO, stem + "_fusion.pt")


def _collect_data_yamls(repo, maxdepth=5):
    """在仓库里搜所有 data.yaml(限深度), 返回绝对路径列表。"""
    repo = os.path.abspath(repo)
    hits = []
    for root, dirs, files in os.walk(repo):
        depth = root[len(repo):].count(os.sep)
        if depth > maxdepth:
            dirs[:] = []
            continue
        if "data.yaml" in files:
            hits.append(os.path.join(root, "data.yaml"))
    return hits


def find_data_dir(branch, repo, data_override=None):
    """自动定位分支数据集目录(内含 data.yaml)。不再依赖 data1/ 还是 data/:
    先看 --data 显式给的; 再按常见位置试; 最后全盘按目录名关键字找。返回含 data.yaml 的【绝对】目录。"""
    base = _BRANCH_DATASETS[branch]
    cands = []
    if data_override:
        cands.append(os.path.abspath(data_override))
    for sub in ("data", "data1", ""):
        cands.append(os.path.join(repo, sub, base))
        cands.append(os.path.join(repo, sub, base.replace("_均衡", "")))
    for c in cands:
        if os.path.isfile(os.path.join(c, "data.yaml")):
            return os.path.abspath(c)
    # 全盘按关键字兜底
    kw = base[:2] if base else branch                     # 取目录名前缀做关键字, 如 '轨道'
    for yaml_p in _collect_data_yamls(repo):
        if kw in os.path.basename(os.path.dirname(yaml_p)):
            return os.path.dirname(os.path.abspath(yaml_p))
    raise FileNotFoundError(
        "自动找不到该分支数据集。请在服务器执行:\n"
        f"  find {repo} -maxdepth 5 -name data.yaml\n"
        f"把含 {base} 数据的那个目录用 --data 传给我。")


def build_validator(yaml_path, args):
    """构造一个可直接驱动循环的 DetectionValidator(复用与 B 完全一致的指标口径)。"""
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.detect.val import DetectionValidator

    data = check_det_dataset(yaml_path)
    validator = DetectionValidator(
        save_dir=Path(os.path.join(_REPO, "runs", "val")),
        args={
            "task": "detect",
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": str(args.device),
            "split": args.split,
            "batch": args.batch,
            "max_det": args.max_det,
            "plots": False,
            "save_json": False,
            "val": True,
            "verbose": True,
        },
    )
    validator.device = _parsed_device(args.device)
    validator.data = data          # init_metrics 需要 self.data.get(self.args.split)
    validator.args.imgsz = args.imgsz
    return validator, data


def main():
    ap = argparse.ArgumentParser(description="融合模型评测(A)")
    ap.add_argument("--branch", default="轨道", choices=list(_BRANCHES))
    ap.add_argument("--weights", default=None, help="融合权重 wrapper.pt; 缺省用 <branch>_fusion.pt")
    ap.add_argument("--data", default=None, help="覆盖数据集 yaml 所在目录")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.001, help="NMS 置信度阈值(与 val 默认一致)")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"),
                    help="评测哪个 split(须与基线 B 一致)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 batch(0=全部), 用于快速接线检查")
    args = ap.parse_args()

    weights = args.weights or _fusion_weight(args.branch)
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"融合权重不存在: {weights}\n  --weights 手动指定, 或先跑 train_fusion_det.py --save")
    print(f"[A 评测] branch={args.branch}  weight={weights}")
    print(f"[A 评测] split={args.split}  imgsz={args.imgsz}  batch={args.batch}  "
          f"conf={args.conf}  device={args.device}")

    device = _parsed_device(args.device)
    data_dir = find_data_dir(args.branch, _REPO, args.data)
    print(f"[数据] 自动定位: {data_dir}")
    wrapper, yaml_path, nc = build_model(args.branch, data_override=data_dir)
    wrapper.to(device)
    for _md in wrapper.encs.values():        # 编码器是普通 dict, nn.Module.to 不递归, 要手动搬
        for _enc in _md.values():
            _enc.to(device)

    # 载入训练好的融合权重(融合/适配/检测头 全由 state_dict 恢复)
    sd = torch.load(weights, map_location=device, weights_only=False)
    wrapper.load_state_dict(sd)

    # 评测前向: det.eval() + end2end=False -> Detect.forward 返回 (y, preds)
    det = wrapper.model.model[-1]
    det.eval()
    if hasattr(det, "end2end"):
        det.end2end = False

    validator, data = build_validator(yaml_path, args)
    validator.stride = max(int(det.stride.max()), 32)
    validator.init_metrics(wrapper.model)

    from ultralytics.data.build import build_yolo_dataset, build_dataloader
    gs = max(int(det.stride.max()), 32)
    val_ds = build_yolo_dataset(wrapper.model.args, data[args.split], args.batch, data,
                                mode="val", rect=True, stride=gs)
    loader = build_dataloader(val_ds, batch=args.batch, workers=args.workers,
                              shuffle=False, rank=-1, device=device)
    n_img = len(val_ds)
    print(f"[数据] {yaml_path}  split={args.split}  图片数={n_img}(未限)")

    torch.set_grad_enabled(False)
    done = 0
    for i, batch in enumerate(loader):
        if args.limit and i >= args.limit:
            break
        batch = validator.preprocess(batch)           # img -> [0,1] float, 其它张量去 device
        out = wrapper(batch["img"])                   # (y, preds)
        y = out[0] if isinstance(out, (tuple, list)) else out
        preds = validator.postprocess(y)              # NMS -> [{'bboxes','conf','cls','extra'}, ...]
        validator.update_metrics(preds, batch)
        done += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}] 已处理 {i + 1} 批", flush=True)
    torch.set_grad_enabled(True)

    stats = validator.get_stats()                     # metrics.process -> 计算 p/r/all_ap, 返回 results_dict
    m = validator.metrics.box

    print("\n===== A 融合模型 整体指标 =====")
    print(f"  mAP50-95 : {stats.get('metrics/mAP50-95(B)', m.map):.4f}")
    print(f"  mAP50    : {stats.get('metrics/mAP50(B)', m.map50):.4f}")
    print(f"  平均精度 : {stats.get('metrics/precision(B)', m.mp):.4f}")
    print(f"  平均召回 : {stats.get('metrics/recall(B)', m.mr):.4f}")

    print("\n===== 逐类指标 =====")
    names = wrapper.model.names if hasattr(wrapper.model, "names") else validator.names
    for i, c in enumerate(validator.metrics.ap_class_index):
        n = names.get(c, str(c))
        print(f"  类{c} {n}: P={m.p[i]:.3f}  R={m.r[i]:.3f}  "
              f"mAP50={m.ap50[i]:.3f}  mAP50-95={m.ap[i]:.3f}")

    print(f"\n处理批次={done}  图片数={validator.seen}")
    print("VAL_FUSION_DONE")


if __name__ == "__main__":
    main()
