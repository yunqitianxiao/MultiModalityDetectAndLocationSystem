#!/usr/bin/env python3
"""
从 YOLO11 检测模型摘出编码器(特征提取器), 去掉检测头。

用法:
  python3 extract_yolo11_encoder.py --weights yolo11l.pt            # 骨干编码器 (默认)
  python3 extract_yolo11_encoder.py --weights .../best.pt --mode neck  # 骨干+颈
  python3 extract_yolo11_encoder.py --weights yolo11l.pt --onnx     # 顺带导出 ONNX

输出:
  yolo11l_encoder.pt    (torch.save 整个编码器, 加载需 import yolo11_encoder)
  yolo11l_encoder.onnx  (仅 --onnx, 自包含, 推理端不用装 torch/ultralytics)

在共享服务器上对训练好的模型执行:
  python3 extract_yolo11_encoder.py \
      --weights ~/linfengyao/2/2/土建/渗漏水/yolo_runs/detect/yolo11l_imgsz512_8x4090/weights/best.pt
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser(description="从 YOLO11 检测模型摘出编码器")
    ap.add_argument("--weights", required=True, help="YOLO11 权重 (.pt), 预训练或训练好的 best.pt")
    ap.add_argument("--mode", choices=["backbone", "neck"], default="backbone",
                    help="backbone=纯骨干(默认); neck=骨干+颈(检测头前)")
    ap.add_argument("--imgsz", type=int, default=512, help="训练/推理输入尺寸, 默认 512")
    ap.add_argument("--device", default="cpu", help="验证 forward 用设备, 默认 cpu")
    ap.add_argument("--out", default=None, help="输出 .pt 路径 (默认同目录 <weights 名>_<mode>_encoder.pt)")
    ap.add_argument("--onnx", action="store_true", help="顺带导出 ONNX")
    args = ap.parse_args()

    from yolo11_encoder import build_encoder, describe_encoder

    print(f"[1/3] 加载 {args.weights} ...")
    enc = build_encoder(args.weights, mode=args.mode, imgsz=args.imgsz, device=args.device)

    print("[2/3] 验证 forward ...")
    # 输入统一 [0,1](与训练一致); 直接喂 [0,255] 会让特征爆炸 ~25x
    dummy = torch.rand(1, 3, args.imgsz, args.imgsz, device=args.device)
    with torch.no_grad():
        outs = enc(dummy)
    print("  输入 (1,3,%d,%d) RGB[0,255] ->" % (args.imgsz, args.imgsz))
    for i, o in enumerate(outs):
        print(f"    输出[{i}] stride={enc.out_strides[i]}  形状 {tuple(o.shape)}  "
              f"dtype {o.dtype}")
    assert len(outs) == 3, "应输出三尺度特征"
    assert enc.out_strides == [8, 16, 32], \
        f"输出 stride 应为 8/16/32, 实测 {enc.out_strides} —— 层索引与该 YOLO 版本不匹配, 请检查 yolo11_encoder.py 的 _BACKBONE_OUT/_NECK_OUT"

    if args.out is None:
        base = os.path.splitext(os.path.basename(args.weights))[0]
        args.out = f"{base}_{args.mode}_encoder.pt"
    print(f"[3/3] 保存编码器 -> {args.out}")
    torch.save(enc, args.out)
    print("\n" + describe_encoder(enc))
    print(f"\n✅ 完成。加载方式:\n"
          f"    import yolo11_encoder\n"
          f"    enc = torch.load('{args.out}', weights_only=False)\n"
          f"    feats = enc(img)   # img: (1,3,H,W) RGB[0,255], H/W 为 32 倍数, 返回 [P3,P4,P5]")

    if args.onnx:
        onnx_path = os.path.splitext(args.out)[0] + ".onnx"
        print(f"\n[可选] 导出 ONNX -> {onnx_path}")
        torch.onnx.export(enc, dummy, onnx_path,
                          input_names=["images"], output_names=["P3", "P4", "P5"],
                          dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}},
                          opset_version=17)
        print(f"   ONNX 已导出, 推理端可用 onnxruntime 直接加载。")


if __name__ == "__main__":
    main()
