# -*- coding: utf-8 -*-
"""
检测框来源 —— 可插拔接口(定位链/避障的输入)
================================================================
定位链吃的是"检测框列表", 而检测框可以来自不同来源。本文件定义统一接口
DetectionSource, 让 verify_chain.py 与将来路径决策不关心框从哪来:

    frame_id -> list[{class_name, bbox:[x1,y1,x2,y2] @1280×720, confidence}]

- GTBoxSource   : 现在用。output_v7 的 labels/*.txt 是检测真值(由缺陷布置精确
                  生成), 直接当"假检测框"喂给定位链/避障 —— 接口先行, 先把
                  定位链和避障算法的数学验证闭环, 不依赖模型检出质量。
- FusionDetSource: 将来用。融合模型训练完后, 重建 PointCloudAssistedYOLO
                  (last.pt 可训参数 + 冻结 YOLO 主干) + eval + decode/NMS 出框,
                  替换 GTBoxSource, 验证脚本代码零改动。
"""
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from localization.tunnel_params import load_calibration  # noqa: E402


class DetectionSource:
    """检测框来源接口。子类实现 get(frame_id)。"""

    def get(self, frame_id: int) -> list[dict]:
        """返回该帧检测框列表, 每项 {class_name, bbox:[x1,y1,x2,y2], confidence}。"""
        raise NotImplementedError


class GTBoxSource(DetectionSource):
    """从 output_v7/labels/*.txt 读检测真值当"假检测框"。

    labels 是 YOLO 归一化 xywh(相对原图 1280×720), 转回像素 xyxy@1280。
    confidence 填 1.0(真值)。空 label 文件 -> 空列表(该帧无检测)。
    """

    def __init__(self, root: str):
        self.root = root
        cal = load_calibration(root)
        self.W, self.H = cal['img_size']
        self.class_names = cal.get('class_names', [])

    def get(self, frame_id: int) -> list[dict]:
        lbl = os.path.join(self.root, 'labels', f'{frame_id:06d}.txt')
        if not os.path.exists(lbl) or os.path.getsize(lbl) == 0:
            return []
        labels = np.loadtxt(lbl, dtype=np.float32).reshape(-1, 5)
        dets = []
        for row in labels:
            cls_id = int(row[0])
            xc, yc, w, h = row[1] * self.W, row[2] * self.H, row[3] * self.W, row[4] * self.H
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            dets.append({
                'class_name': cls_name,
                'bbox': [float(xc - w / 2), float(yc - h / 2),
                         float(xc + w / 2), float(yc + h / 2)],
                'confidence': 1.0,
            })
        return dets


class FusionDetSource(DetectionSource):
    """融合检测模型推理框(检测模型训练完成后实现)。

    TODO(检测模型训练完):
      1. 重建 PointCloudAssistedYOLO: 从 runs/assist_point_pretrained/last.pt
         载入 118 个可训参数(load_state_dict strict=False), 冻结主干从
         YOLO11L_CKPT 重建(见 fusion/image_encoder.py)。
      2. model.eval(); 前向 imgs(BGR@640) + attrmap + pose_state。
      3. Detect eval 模式返回 (y, preds) 元组: y=(B, 11, 8400) xywh+cls,
         dbox 是 xywh @640(相对 letterbox 画布), cls 已 sigmoid。
      4. ultralytics.utils.ops.non_max_suppression(y, conf, iou) 出框,
         再按 letterbox 变换缩回 1280×720 原图像素坐标。
    """

    def __init__(self, ckpt: str, conf: float = 0.35, iou: float = 0.45):
        self.ckpt, self.conf, self.iou = ckpt, conf, iou
        raise NotImplementedError(
            'FusionDetSource 待检测模型训练完成后实现(接口已预留)。当前用 GTBoxSource。')


if __name__ == '__main__':
    # 冒烟: GTBoxSource 读第 0 帧, 应返回几个检测框(000000 含异物)
    root = os.path.join(_DIR, 'data_gen', 'output_v7')
    src = GTBoxSource(root)
    dets = src.get(0)
    print(f'GTBoxSource 帧000000: {len(dets)} 个框')
    for d in dets[:4]:
        print(' ', d['class_name'], [round(v, 1) for v in d['bbox']])
