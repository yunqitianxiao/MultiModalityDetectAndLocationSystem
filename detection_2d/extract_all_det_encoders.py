#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键摘取三个 2D 检测编码器 —— 覆盖原有的编码器 pt
================================================================================
把三个已训练的 YOLO11l 检测权重，分别摘成「检测编码器」（去掉 Detect 检测头），
输出特征金字塔 P3/P4/P5，供 ④ 跨模态融合层（交叉注意力）接入。

三个分支:
  轨道 guidao_best.pt   -> 断裂/丢失/松动歪斜 (3类)   -> 轨道固件_4cls_stage/
  土建 tujian_best.pt   -> 裂缝/渗漏水/管片掉块 (3类)  -> 土建_3cls_stage/
  设备 shebei_best.pt   -> 异物/支架松脱 (2类)         -> 设备_2cls_stage/  (新建)

每个分支输出两种模式(与原逻辑一致):
  <前缀>_backbone_encoder.pt  纯骨干(层0..10, 到 C2PSA),  P3/P4/P5, C=512/512/512
  <前缀>_neck_encoder.pt      骨干+颈(层0..22, 检测头前), P3'/P4'/P5', C=256/512/512

编码器对象额外带 ./.branch ./names 元数据(便于融合层区分来源), 不影响原有加载。
复用 yolo11_encoder.build_encoder —— 即"本来的"摘取逻辑。

用法(默认 python3 即可跑, CPU 能 forward 校验):
  python3 extract_all_det_encoders.py
  python3 extract_all_det_encoders.py --imgsz 512 --out 统一输出目录

输出统一收集到 --out 时, 跨模态融合层可直接从单个目录加载三个编码器。
"""
import os, sys, argparse

import torch

# 让脚本能找到本来的 yolo11_encoder.py
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "土建", "渗漏水"))
from yolo11_encoder import build_encoder, describe_encoder  # noqa: E402

# 三个分支 = (中文名, 权重路径, stage目录, 输出前缀, 类名)
BRANCHES = [
    ("轨道", os.path.join(_HERE, "guidao_best.pt"),
     os.path.join(_HERE, "data1", "轨道固件_4cls_stage"), "yolo11l_轨道3cls",
     {"断裂", "丢失", "松动歪斜"}),
    ("土建", os.path.join(_HERE, "tujian_best.pt"),
     os.path.join(_HERE, "data1", "土建_3cls_stage"), "yolo11l_土建3cls",
     {"裂缝", "渗漏水", "管片掉块"}),
    ("设备", os.path.join(_HERE, "shebei_best.pt"),
     os.path.join(_HERE, "data1", "设备_2cls_stage"), "yolo11l_设备2cls",
     {"foreign_object", "bracket_loose"}),
]


def main():
    ap = argparse.ArgumentParser(description="一键摘取三个 2D 检测编码器并覆盖原有 pt")
    ap.add_argument("--mode", choices=["backbone", "neck", "both"], default="both",
                    help="both=骨干+颈(默认); backbone/neck 单独")
    ap.add_argument("--imgsz", type=int, default=512, help="验证 forward 用输入尺寸(H/W 为 32 倍数), 默认 512")
    ap.add_argument("--out", default=None,
                    help="=None 覆盖到各分支 stage 目录(默认/本来的); 指定则统一收集到该目录")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    all_reports = []

    for branch, weight, stage_dir, prefix, names in BRANCHES:
        print("=" * 72)
        print(f"[{branch}] 权重: {weight}")
        if not os.path.isfile(weight):
            print(f"  !! 权重缺失, 跳过: {weight}")
            continue
        if not args.out:
            os.makedirs(stage_dir, exist_ok=True)
            out_base = stage_dir
        else:
            os.makedirs(args.out, exist_ok=True)
            out_base = args.out

        for mode in (("backbone", "neck") if args.mode == "both" else (args.mode,)):
            out_pt = os.path.join(out_base, f"{prefix}_{mode}_encoder.pt")
            print(f"--- [{branch}] mode={mode} -> {out_pt}")
            enc = build_encoder(weight, mode=mode, imgsz=args.imgsz, device="cpu")
            enc.branch = branch     # 元数据: 分支来源(融合层用于区分)
            enc.names = names       # 元数据: 该分支类别

            # 前向校验: 输入 -> 三尺度特征
            dummy = torch.rand(1, 3, args.imgsz, args.imgsz) * 255.0
            outs = enc(dummy)
            strides = list(enc.out_strides)
            shapes = [tuple(o.shape) for o in outs]
            print(f"    输入 (1,3,{args.imgsz},{args.imgsz}) -> {len(outs)} 尺度")
            for i, (s, sh) in enumerate(zip(strides, shapes)):
                print(f"      输出[{i}] stride={s}  形状={sh}  dtype={outs[i].dtype}")
            assert len(outs) == 3, f"{branch}/{mode}: 应输出 3 尺度, 实测 {len(outs)}"
            assert strides == [8, 16, 32], \
                f"{branch}/{mode}: stride 应为 8/16/32, 实测 {strides} —— 层索引与此 YOLO 版本不符"

            torch.save(enc, out_pt)
            print(f"    已保存并覆盖 -> {out_pt}")
            all_reports.append((branch, mode, out_pt, strides, shapes))

    # 汇总
    print("\n" + "=" * 72)
    print("✅ 全部完成。汇总(均输出 P3/P4/P5, stride 8/16/32):")
    for branch, mode, out_pt, strides, shapes in all_reports:
        print(f"  [{branch}/{mode}] {os.path.basename(out_pt)}")
        print(f"      输入 (1,3,{args.imgsz},{args.imgsz}) -> "
              + ", ".join(f"stride{s}({sh[2]}x{sh[3]})" for s, sh in zip(strides, shapes)))
    print("\n加载方式(融合层):")
    print("  import yolo11_encoder   # 需在 sys.path 里能找到该模块")
    print("  enc = torch.load('<xx_encoder.pt>', weights_only=False)")
    print(f"  feats = enc(img)       # img: (1,3,H,W) RGB[0,255], H/W 为 32 倍数, 返回 [P3,P4,P5]")
    print("  enc.branch / enc.names  # 分支与类别元数据")


if __name__ == "__main__":
    main()
