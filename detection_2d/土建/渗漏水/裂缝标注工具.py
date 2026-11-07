#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
裂缝(及土建病害)检测标签标注工具 —— 输出 YOLO 格式, 与现有数据集完全兼容
=====================================================================
交互式矩形框标注器 (OpenCV GUI):
  - 拖拽画框 / 右键删框 / d 删最近 / z 撤销 / 0-9 切类别
  - 滚轮缩放 + 中键拖拽平移 (标细裂缝必需)
  - 显示并复用已有 YOLO 标签(同名 .txt), 增量修正
  - 可选 --weights: 模型预标注, 人工只改错漏 → 半自动流程
  - 保存: 每行 `cls cx cy w h`(归一化), 兼容 土建_裂缝渗漏水_yolo 等

用法:
  # 纯人工标注
  python3 裂缝标注工具.py --src /path/to/待标图片 --names "crack,leakage"

  # 半自动 (只保留 --names 里出现的类)
  python3 裂缝标注工具.py --src /path/to/待标图片 --weights /path/best.pt --conf 0.12

操作:
  左键拖拽   = 画框             右键点框   = 删除该框
  滚轮       = 缩放(以鼠标为中心)  中键拖拽   = 平移视图
  0..9       = 切类别(顶部显示当前类)
  s          = 保存当前标签      w / → / 空格 = 保存并下一张
  ← / a      = 上一张            q / Esc     = 退出
  d          = 删最近框          z           = 撤销
  r          = 清空当前图        e           = 删除当前标签文件
"""
import os
# cv2 用 Qt 后端时静默字体目录警告 (Linux 桌面一般有 /usr/share/fonts)
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import argparse
import sys
import numpy as np
import cv2

# ---------- 模块级常量 (waitKeyEx 方向键码, 兼容不同后端) ----------
KEY_LEFT = (2424832, 16777234, 63234)      # ←
KEY_RIGHT = (2555904, 16777236, 63235)     # →
KEY_SPACE = 32

DEFAULT_NAMES = ["crack", "leakage", "spalling"]
PALETTE = [
    (0, 0, 255),      # 红  crack
    (0, 215, 255),    # 黄  leakage
    (255, 0, 0),      # 蓝  spalling
    (0, 255, 0),      # 绿
    (255, 0, 255),    # 洋红
    (0, 165, 255),    # 橙
    (128, 128, 255),  # 紫
    (0, 255, 255),    # 青
]
MIN_SCALE_RATIO = 1.0    # 最小缩放 = 适配窗口
MAX_SCALE_RATIO = 12.0   # 最大放大倍数(相对适配)

_NOW_NS = None


def now():
    global _NOW_NS
    import time
    return time.monotonic()


class Labeler:
    def __init__(self, src, out, names, default_cls, model=None, conf=0.12,
                 device="0", win_w=1280, win_h=800):
        self.src = os.path.abspath(src)
        self.out = os.path.abspath(out)
        self.names = names
        self.cur_cls = max(0, min(default_cls, len(names) - 1))
        self.model = model
        self.conf = conf
        self.device = device
        self.win_w, self.win_h = win_w, win_h

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.imgs = sorted(
            f for f in os.listdir(self.src)
            if f.lower().endswith(exts))
        if not self.imgs:
            sys.exit(f"\n✗ {self.src} 里没有图片\n")
        os.makedirs(self.out, exist_ok=True)

        self.idx = 0
        self.boxes = []          # [(cls, (x1,y1,x2,y2) 原图像素坐标)]
        self.dirty = False
        self.dragging = False
        self.drag_start = None
        self.pan_drag = False
        self.hover_i = -1
        self.msg = ""
        self.msg_t = 0.0
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        cv2.namedWindow("CrackLabeler", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CrackLabeler", win_w, win_h)
        self._load(self.idx)

    # ================= IO (兼容中文路径) =================
    def _imread(self, path):
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def _lbl_path(self, img_name):
        return os.path.join(self.out, os.path.splitext(img_name)[0] + ".txt")

    def _read_labels(self, img_name, W, H):
        p = self._lbl_path(img_name)
        if not os.path.isfile(p):
            return []
        boxes = []
        for line in open(p, encoding="utf-8").read().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls, cx, cy, bw, bh = map(float, parts)
            except ValueError:
                continue
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            boxes.append((int(cls), (x1, y1, x2, y2)))
        return boxes

    def _save(self, img_name, W, H):
        p = self._lbl_path(img_name)
        lines = []
        for cls, (x1, y1, x2, y2) in self.boxes:
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            x1 = max(0.0, x1); y1 = max(0.0, y1)
            x2 = min(W, x2);  y2 = min(H, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n" if lines else "")
        self.dirty = False

    # ================= 加载 / 导航 =================
    def _load(self, idx):
        self.img_name = self.imgs[idx]
        self.img = self._imread(os.path.join(self.src, self.img_name))
        if self.img is None:
            sys.exit(f"\n✗ 无法读取图片: {self.img_name}\n")
        self.H, self.W = self.img.shape[:2]
        self.boxes = self._read_labels(self.img_name, self.W, self.H)
        self.dirty = False
        self._reset_view()
        if self.model is not None and not self.boxes:
            self._autolabel()

    def _reset_view(self):
        """适配窗口的初始视图。"""
        fit = min(self.win_w / self.W, self.win_h / self.H)
        self.scale = fit
        self.pan_x = (self.win_w - self.W * fit) / 2
        self.pan_y = (self.win_h - self.H * fit) / 2

    def _autolabel(self):
        self._set_msg("预标注中…", 2.0)
        res = self.model.predict(
            source=os.path.join(self.src, self.img_name),
            conf=self.conf, iou=0.5, imgsz=512,
            device=self.device, verbose=False)[0]
        new = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            cl = res.boxes.cls.cpu().numpy()
            for bx, cid in zip(xyxy, cl):
                cid = int(cid)
                if cid < len(self.names):        # 只保留标注类别内的框
                    new.append((cid, tuple(map(float, bx))))
        self.boxes = new
        self.dirty = bool(new)
        self._set_msg(f"预标注 {len(new)} 框 (conf≥{self.conf}), 请人工修正", 3.0)

    # ================= 渲染 =================
    def _to_view(self, ox, oy):
        return self.pan_x + ox * self.scale, self.pan_y + oy * self.scale

    def _to_img(self, mx, my):
        return (mx - self.pan_x) / self.scale, (my - self.pan_y) / self.scale

    def _clip_pan(self):
        dw = self.W * self.scale
        dh = self.H * self.scale
        margin = 60
        if dw <= self.win_w - 2 * margin:
            self.pan_x = (self.win_w - dw) / 2
        else:
            self.pan_x = max(min(self.pan_x, margin), self.win_w - dw - margin)
        if dh <= self.win_h - 2 * margin:
            self.pan_y = (self.win_h - dh) / 2
        else:
            self.pan_y = max(min(self.pan_y, margin), self.win_h - dh - margin)

    def _render(self):
        canvas = np.full((self.win_h, self.win_w, 3), 24, np.uint8)
        iw = int(self.W * self.scale)
        ih = int(self.H * self.scale)
        if iw >= 2 and ih >= 2:
            disp = cv2.resize(self.img, (iw, ih))
            x0 = int(round(self.pan_x))
            y0 = int(round(self.pan_y))
            x1 = max(0, x0); y1 = max(0, y0)
            x2 = min(self.win_w, x0 + iw); y2 = min(self.win_h, y0 + ih)
            if x2 > x1 and y2 > y1:
                canvas[y1:y2, x1:x2] = disp[y1 - y0:y2 - y0, x1 - x0:x2 - x0]

        # 已有框
        for i, (cls, (x1, y1, x2, y2)) in enumerate(self.boxes):
            color = PALETTE[cls % len(PALETTE)]
            ox1, oy1 = self._to_view(min(x1, x2), min(y1, y2))
            ox2, oy2 = self._to_view(max(x1, x2), max(y1, y2))
            sel = (i == self.hover_i)
            cv2.rectangle(canvas, (int(ox1), int(oy1)), (int(ox2), int(oy2)),
                          (255, 255, 255) if sel else color,
                          2 if sel else 1, cv2.LINE_AA)
            label = self.names[cls]
            th = 15
            lw = len(label) * 13 + 6
            cv2.rectangle(canvas, (int(ox1), int(oy1 - th)),
                          (int(ox1) + lw, int(oy1)), color, -1)
            cv2.putText(canvas, label, (int(ox1) + 3, int(oy1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 拖拽中的新框
        if self.dragging and self.drag_start:
            color = PALETTE[self.cur_cls % len(PALETTE)]
            sx, sy = self._to_view(*self.drag_start)
            ex, ey = self._to_view(*self._mouse_img)
            cv2.rectangle(canvas, (int(sx), int(sy)), (int(ex), int(ey)),
                          color, 2, cv2.LINE_AA)

        # 状态栏
        zoom = self.scale / max(min(self.win_w / self.W, self.win_h / self.H), 1e-9)
        bar = (f"[{self.idx + 1}/{len(self.imgs)}] {self.img_name} | "
               f"{self.names[self.cur_cls]}({self.cur_cls}) | "
               f"框={len(self.boxes)} | 缩放x{zoom:.1f} | "
               f"s存 w存+下一 q退 d删 z撤销 r清空")
        if self.msg_t > now():
            bar = self.msg + " | " + bar
        cv2.putText(canvas, bar, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("CrackLabeler", canvas)

    def _set_msg(self, text, seconds):
        self.msg = text
        self.msg_t = now() + seconds

    # ================= 鼠标 =================
    def _on_mouse(self, event, mx, my, flags, param):
        self._mx, self._my = mx, my
        ox, oy = self._to_img(mx, my)

        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:      # Ctrl+左键 = 平移
                self.pan_drag = True
                self.pan_anchor = (mx, my)
            else:
                self.dragging = True
                self.drag_start = (max(0, min(self.W, ox)), max(0, min(self.H, oy)))
                self._mouse_img = self.drag_start
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            ox1, oy1 = self.drag_start
            ox2, oy2 = self._mouse_img
            if abs(ox2 - ox1) > 3 and abs(oy2 - oy1) > 3:
                self.boxes.append((self.cur_cls,
                                   (min(ox1, ox2), min(oy1, oy2),
                                    max(ox1, ox2), max(oy1, oy2))))
                self.dirty = True
                self._set_msg(f"已添加 {self.names[self.cur_cls]} 框", 1.2)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self._mouse_img = (max(0, min(self.W, ox)), max(0, min(self.H, oy)))
            elif self.pan_drag:
                dx = mx - self.pan_anchor[0]
                dy = my - self.pan_anchor[1]
                self.pan_x += dx
                self.pan_y += dy
                self.pan_anchor = (mx, my)
                self._clip_pan()
            else:
                self.hover_i = self._hit_box(ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and self.pan_drag:
            self.pan_drag = False
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.pan_drag = True
            self.pan_anchor = (mx, my)
        elif event == cv2.EVENT_MBUTTONUP:
            self.pan_drag = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            i = self._hit_box(ox, oy)
            if i >= 0:
                self.boxes.pop(i)
                self.dirty = True
                self.hover_i = -1
                self._set_msg("已删除框", 1.2)
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) if flags else 0       # >0 向上
            fit = min(self.win_w / self.W, self.win_h / self.H)
            factor = 1.18 if delta > 0 else 1 / 1.18
            new_scale = self.scale * factor
            new_scale = max(fit * MIN_SCALE_RATIO,
                            min(fit * MAX_SCALE_RATIO, new_scale))
            # 以鼠标位置为锚点缩放
            self.pan_x = mx - ox * new_scale
            self.pan_y = my - oy * new_scale
            self.scale = new_scale
            self._clip_pan()

    def _hit_box(self, ox, oy):
        for i in range(len(self.boxes) - 1, -1, -1):
            cls, (x1, y1, x2, y2) = self.boxes[i]
            if min(x1, x2) <= ox <= max(x1, x2) and min(y1, y2) <= oy <= max(y1, y2):
                return i
        return -1

    # ================= 键盘 =================
    def _on_key(self, k):
        for ci in range(min(10, len(self.names))):
            if k == ord(str(ci)):
                if self.cur_cls != ci:
                    self.cur_cls = ci
                    self._set_msg(f"类别 → {self.names[ci]}", 1.2)
                return True
        if k in (ord('+'), ord('=')):
            self._zoom(1.5)
            return True
        if k in (ord('-'), ord('_')):
            self._zoom(1 / 1.5)
            return True
        if k == ord('f'):
            self._reset_view()
            self._set_msg("视图复位", 1.0)
            return True

        if k in (ord('q'), 27):
            self._save(self.img_name, self.W, self.H)
            print(f"保存 {self.img_name} → {self._lbl_path(self.img_name)}")
            return False
        if k == ord('s'):
            self._save(self.img_name, self.W, self.H)
            self._set_msg("已保存 " + os.path.basename(self._lbl_path(self.img_name)), 1.5)
            return True
        if k in (ord('w'), KEY_SPACE) or k in KEY_RIGHT:
            self._save(self.img_name, self.W, self.H)
            self.idx = min(self.idx + 1, len(self.imgs) - 1)
            self._load(self.idx)
            return True
        if k in KEY_LEFT or k == ord('a'):
            if self.dirty:
                self._save(self.img_name, self.W, self.H)
            self.idx = max(self.idx - 1, 0)
            self._load(self.idx)
            return True
        if k == ord('d'):
            if self.boxes:
                self.boxes.pop()
                self.dirty = True
                self._set_msg("已删除最近框", 1.2)
            return True
        if k == ord('r'):
            self.boxes = []
            self.dirty = True
            self._set_msg("已清空当前图", 1.2)
            return True
        if k == ord('e'):
            p = self._lbl_path(self.img_name)
            if os.path.isfile(p):
                os.remove(p)
                self.boxes = []
                self.dirty = False
                self._set_msg("已删除标签文件", 1.5)
            return True
        return True

    def _zoom(self, factor):
        fit = min(self.win_w / self.W, self.win_h / self.H)
        cx = self.win_w / 2
        cy = self.win_h / 2
        ox, oy = self._to_img(cx, cy)
        self.scale = max(fit * MIN_SCALE_RATIO,
                         min(fit * MAX_SCALE_RATIO, self.scale * factor))
        self.pan_x = cx - ox * self.scale
        self.pan_y = cy - oy * self.scale
        self._clip_pan()

    # ================= 主循环 =================
    def run(self):
        cv2.setMouseCallback("CrackLabeler", self._on_mouse)
        print(f"图片 {len(self.imgs)} 张 | 类别 {self.names} | 输出 → {self.out}")
        print("左键拖拽画框 | 滚轮缩放 | 中键(或Ctrl+左键)平移 | 右键删框")
        print("0-9切类 | s存 | w存并下一 | ←→/a导航 | q退出")
        if self.model is not None:
            print(f"预标注模式: conf≥{self.conf}, 无标签图自动生成初稿")
        print("=" * 70)
        while True:
            self._render()
            k = cv2.waitKeyEx(1)
            if k < 0:
                continue
            if not self._on_key(k):
                break
        cv2.destroyAllWindows()
        print(f"\n完成。标签已保存到: {self.out}")


def main():
    ap = argparse.ArgumentParser(
        description="裂缝检测标签标注工具 (YOLO 格式, 兼容现有数据)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--src", required=True, help="待标注图片目录(必填)")
    ap.add_argument("--out", default=None,
                    help="标签输出目录(默认 <src>/../labels, 兼容 train/images→train/labels)")
    ap.add_argument("--names", default=",".join(DEFAULT_NAMES),
                    help="类别名, 逗号分隔 (与训练 data.yaml names 顺序一致!)")
    ap.add_argument("--cls", type=int, default=0, help="默认类别索引")
    ap.add_argument("--weights", default=None, help="预标注模型 best.pt (半自动模式)")
    ap.add_argument("--conf", type=float, default=0.12, help="预标注置信度阈值")
    ap.add_argument("--device", default="0", help="预标注推理设备: 0 / cpu")
    ap.add_argument("--win", default="1280x800", help="窗口尺寸 WxH")
    args = ap.parse_args()

    names = [x.strip() for x in args.names.split(",") if x.strip()]
    if not names:
        sys.exit("✗ --names 为空\n")
    win_w, win_h = (int(x) for x in args.win.lower().split("x"))

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.src)), "labels")

    model = None
    if args.weights:
        if not os.path.isfile(args.weights):
            sys.exit(f"\n✗ 找不到预标注权重: {args.weights}\n")
        from ultralytics import YOLO
        print(f"加载预标注模型: {args.weights} …")
        model = YOLO(args.weights)
        mnames = list(model.names.values()) if model.names else []
        print(f"  模型类别: {mnames} | 标注类别: {names}")
        if mnames and len(mnames) > len(names):
            print(f"  ⚠ 模型前{len(names)}类: {mnames[:len(names)]} 与 --names={names} 需一致, "
                  f"超出的类({mnames[len(names):]})会被忽略")

    lab = Labeler(args.src, out, names, args.cls, model=model, conf=args.conf,
                  device=args.device, win_w=win_w, win_h=win_h)
    lab.run()


if __name__ == "__main__":
    main()
