"""
本机 torchvision 为纯 CPU 构建 (0.28.0+cpu), CUDA 上缺 `torchvision::nms` 算子。
ultralytics 原生训练过 epoch 后在内部 validate() 的 postprocess 会对 CUDA 检测框做 NMS,
直接 NotImplementedError 崩掉 (训练本身不涉及 NMS, 故只有验证步会触发)。

本 shim 把 nms 的 boxes/scores 挪到 CPU 计算、再把返回值(索引)挪回原设备,
对上层代码完全无感知 —— 与工程里 evaluate_pr / ood_gate "绕开 CUDA nms 缺失" 的
既有做法一致 (fusion/evaluate_pr.py 是显式把 grid 丢到 CPU, 这里则是打补丁兜底)。

用法: 在 import ultralytics 之前 import 并调用一次即可:
    from fix_nms import patch_torchvision_nms
    patch_torchvision_nms()
"""

import torchvision

_PATCHED = False


def patch_torchvision_nms():
    """把 torchvision.ops.nms 替换为"CPU 计算, 索引挪回原设备"。幂等, 可重复调用。"""
    global _PATCHED
    if _PATCHED:
        return
    _orig = torchvision.ops.nms

    def _nms(boxes, scores, iou_threshold):
        dev = boxes.device
        idx = _orig(boxes.detach().cpu(), scores.detach().cpu(), iou_threshold)
        return idx.to(dev)

    torchvision.ops.nms = _nms
    _PATCHED = True
