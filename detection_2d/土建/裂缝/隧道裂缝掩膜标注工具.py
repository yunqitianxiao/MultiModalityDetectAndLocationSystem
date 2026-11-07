#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
隧道裂缝掩膜标注工具
- 目标: data/土建_裂缝/06*.jpg|.jpeg (12张隧道图, 含 0601 —— 标完它就能进数据集)
- 约定: 裂缝涂白(255), 背景涂黑(0)  ← 与所有训练脚本一致(掩膜 >127 = 病害)
        千万不要涂反, 否则模型会把裂缝当背景学
- 保存: data/土建_裂缝_手动标注的掩膜/mask_06xx.png
        PNG 无损、纯 255; 保存时自动删除旧的 .jpg/.jpeg 低质量掩膜
- 操作:
    左键拖动 : 画裂缝(白色)        右键拖动 : 擦除(黑色)
    滚轮     : 缩放(以光标为中心)  中键拖动 : 平移
    [  /  ]  : 画笔变小 / 变大      U : 撤销    C : 清空
    A / D    : 上一张 / 下一张(自动保存)
    S        : 保存                 F : 适应窗口
    Q / ESC  : 退出(自动保存)
"""
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider,
                             QGroupBox)
from PyQt5.QtCore import Qt, QPointF, QTimer
from PyQt5.QtGui import QImage, QPainter, QPen, QColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "土建_裂缝")
MASK_DIR = os.path.join(PROJECT_ROOT, "data", "土建_裂缝_手动标注的掩膜")
os.makedirs(MASK_DIR, exist_ok=True)


class MaskCanvas(QWidget):
    """画布: 全分辨率掩膜上作画, 显示时缩放"""
    def __init__(self):
        super().__init__()
        self.image = None          # BGR uint8 全分辨率
        self.mask = None           # uint8 全分辨率, 0/255
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.brush = 20            # 屏幕像素半径
        self.painting = False
        self.erasing = False
        self.panning = False
        self.last_img_pos = None
        self.cursor = QPointF(-1, -1)
        self.undo_stack = []
        self.on_change = None      # 回调: 画笔变化/画布变化
        self.setMouseTracking(True)
        self.setMinimumSize(520, 520)

    # ---------- 数据 ----------
    def set_image_mask(self, image, mask):
        self.image = image
        self.mask = mask
        self.undo_stack = []
        self.fit_view()
        self.update()

    def fit_view(self):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        if self.height() <= 0 or self.width() <= 0:
            return
        self.scale = min(self.height() / h, self.width() / w)
        dw, dh = w * self.scale, h * self.scale
        self.offset = QPointF((self.width() - dw) / 2, (self.height() - dh) / 2)
        self.update()

    def widget_to_img(self, pos):
        if self.image is None:
            return None
        h, w = self.image.shape[:2]
        x = (pos.x() - self.offset.x()) / self.scale
        y = (pos.y() - self.offset.y()) / self.scale
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return (int(x), int(y))

    def _brush_r_img(self):
        return max(1, int(round(self.brush / self.scale)))

    def _stamp(self, p, color):
        cv2.circle(self.mask, p, self._brush_r_img(), color, -1)

    def _stroke(self, p1, p2, color):
        cv2.line(self.mask, p1, p2, color,
                 thickness=2 * self._brush_r_img(), lineType=cv2.LINE_AA)

    # ---------- 撤销 ----------
    def push_undo(self):
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)

    def undo(self):
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.update()
            self._notify()

    def clear(self):
        if self.mask is None:
            return
        self.push_undo()
        self.mask[:] = 0
        self.update()
        self._notify()

    def _notify(self):
        if self.on_change:
            self.on_change()

    # ---------- 事件 ----------
    def mousePressEvent(self, ev):
        if self.image is None:
            return
        p = self.widget_to_img(ev.pos())
        if ev.button() == Qt.LeftButton:
            self.push_undo()
            self.painting = True
            if p:
                self._stamp(p, 255)
            self.last_img_pos = p
        elif ev.button() == Qt.RightButton:
            self.push_undo()
            self.erasing = True
            if p:
                self._stamp(p, 0)
            self.last_img_pos = p
        elif ev.button() == Qt.MiddleButton:
            self.panning = True
            self.pan_last = ev.pos()
        self.update()

    def mouseMoveEvent(self, ev):
        self.cursor = ev.pos()
        if self.panning:
            d = ev.pos() - self.pan_last
            self.offset += QPointF(d)
            self.pan_last = ev.pos()
        elif self.painting or self.erasing:
            color = 255 if self.painting else 0
            p = self.widget_to_img(ev.pos())
            if p and self.last_img_pos:
                self._stroke(self.last_img_pos, p, color)
            elif p:
                self._stamp(p, color)
            self.last_img_pos = p
        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.painting = False
        if ev.button() == Qt.RightButton:
            self.erasing = False
        if ev.button() == Qt.MiddleButton:
            self.panning = False
        self.last_img_pos = None
        self.update()
        self._notify()

    def wheelEvent(self, ev):
        if self.image is None:
            return
        factor = 1.25 ** (ev.angleDelta().y() / 120.0)
        new_scale = min(64.0, max(0.02, self.scale * factor))
        pos = ev.pos()
        img_x = (pos.x() - self.offset.x()) / self.scale
        img_y = (pos.y() - self.offset.y()) / self.scale
        self.scale = new_scale
        self.offset = QPointF(pos.x() - img_x * self.scale,
                              pos.y() - img_y * self.scale)
        self.update()

    def paintEvent(self, ev):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        dw = max(1, int(round(w * self.scale)))
        dh = max(1, int(round(h * self.scale)))

        img = self.image if (dw, dh) == (w, h) else \
            cv2.resize(self.image, (dw, dh), interpolation=cv2.INTER_AREA)
        img = img.copy()

        # 绿色半透明叠加掩膜前景
        if self.mask is not None:
            if self.mask.shape[:2] != (h, w):
                self.mask = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)
            m = self.mask
            if m.shape[0] != dh or m.shape[1] != dw:
                m = cv2.resize(m, (dw, dh), interpolation=cv2.INTER_NEAREST)
            fg = m > 127
            img[fg] = (img[fg].astype(np.float32) * 0.55
                       + np.array([0, 255, 0]) * 0.45).astype(np.uint8)

        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      rgb.strides[0], QImage.Format_RGB888)
        painter = QPainter(self)
        painter.drawImage(QPointF(self.offset.x(), self.offset.y()), qimg)

        if self.cursor.x() >= 0 and self.mask is not None:
            painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.DashLine))
            painter.drawEllipse(self.cursor, self.brush, self.brush)
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("隧道裂缝掩膜标注工具 (白=裂缝 / 黑=背景)")
        self.setFocusPolicy(Qt.StrongFocus)

        self.canvas = MaskCanvas()
        self.canvas.on_change = self.update_status
        self.image_files = self._list_images()
        self.current = 0

        self._build_ui()
        if self.image_files:
            self.load_image(0)
            QTimer.singleShot(120, self.canvas.fit_view)  # 等布局定型后适配
        else:
            print("❌ 未找到 data/土建_裂缝/06*.jpg 图片, 请检查目录")

    def _list_images(self):
        if not os.path.isdir(IMAGE_DIR):
            return []
        return sorted(f for f in os.listdir(IMAGE_DIR)
                      if f.lower().startswith("06")
                      and f.lower().endswith((".jpg", ".jpeg", ".png")))

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        # 画布
        root.addWidget(self.canvas, stretch=1)

        # 控制栏
        ctl = QHBoxLayout()
        btn = lambda text, slot: (lambda b: (b.clicked.connect(slot), ctl.addWidget(b)))(QPushButton(text))
        for t, s in (("上一张(A)", self.prev), ("下一张(D)", self.next_),
                     ("保存(S)", self.save_current), ("撤销(U)", self.canvas.undo),
                     ("清空(C)", self.canvas.clear), ("适应窗口(F)", self.canvas.fit_view)):
            btn(t, s)

        self.brush_label = QLabel("笔刷: 20")
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(3, 120)
        self.brush_slider.setValue(20)
        self.brush_slider.valueChanged.connect(self.change_brush)
        ctl.addWidget(self.brush_label)
        ctl.addWidget(self.brush_slider)
        root.addLayout(ctl)

        # 状态与提示
        info = QGroupBox("提示")
        vl = QVBoxLayout(info)
        self.status_label = QLabel("")
        vl.addWidget(self.status_label)
        vl.addWidget(QLabel(
            "左键=涂裂缝(白)  右键=擦除(黑)  滚轮=缩放  中键拖动=平移  [ ]=笔刷  "
            "U=撤销  C=清空  A/D=翻页  S=保存  Q=退出"))
        root.addWidget(info)

        self.setCentralWidget(central)
        self.resize(1100, 800)

    # ---------- 逻辑 ----------
    def current_base(self):
        return os.path.splitext(self.image_files[self.current])[0]

    def load_image(self, idx):
        if self.current != idx:
            self.save_current()          # 切图前自动保存
        self.current = idx
        fname = self.image_files[idx]
        img = cv2.imread(os.path.join(IMAGE_DIR, fname))
        if img is None:
            print(f"❌ 读取失败: {fname}")
            return
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        base = self.current_base()
        for ext in (".png", ".jpg", ".jpeg"):
            mp = os.path.join(MASK_DIR, f"mask_{base}{ext}")
            if os.path.exists(mp):
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask[m > 127] = 255
                break
        self.canvas.set_image_mask(img, mask)
        self.update_status()

    def save_current(self):
        if not self.image_files or self.canvas.mask is None:
            return
        base = self.current_base()
        out = os.path.join(MASK_DIR, f"mask_{base}.png")
        os.makedirs(MASK_DIR, exist_ok=True)
        cv2.imwrite(out, self.canvas.mask)
        for ext in (".jpg", ".jpeg"):   # 删除旧的低质量掩膜, 避免配对误用
            old = os.path.join(MASK_DIR, f"mask_{base}{ext}")
            if os.path.exists(old):
                os.remove(old)
                print(f"  已删除旧掩膜: {old}")
        print(f"  ✅ 已保存: {out}")

    def update_status(self):
        if not self.image_files or self.canvas.mask is None:
            return
        base = self.current_base()
        fg = int((self.canvas.mask > 0).sum())
        total = len(self.image_files)
        self.status_label.setText(
            f"[{self.current + 1}/{total}] {base}   已标裂缝像素: {fg}   "
            f"笔刷(屏幕): {self.canvas.brush}  缩放: {self.canvas.scale:.2f}")
        self.brush_label.setText(f"笔刷: {self.canvas.brush}")
        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(self.canvas.brush)
        self.brush_slider.blockSignals(False)

    def change_brush(self, v):
        if v is not None and not isinstance(v, bool):
            self.canvas.brush = int(v)
            self.update_status()

    def prev(self):
        if self.image_files:
            self.load_image((self.current - 1) % len(self.image_files))

    def next_(self):
        if self.image_files:
            self.load_image((self.current + 1) % len(self.image_files))

    def keyPressEvent(self, ev):
        k = ev.key()
        if k in (Qt.Key_A, Qt.Key_Left):
            self.prev()
        elif k in (Qt.Key_D, Qt.Key_Right):
            self.next_()
        elif k == Qt.Key_S:
            self.save_current()
        elif k == Qt.Key_U:
            self.canvas.undo()
        elif k == Qt.Key_C:
            self.canvas.clear()
        elif k == Qt.Key_F:
            self.canvas.fit_view()
        elif k == Qt.Key_BracketLeft:
            self.change_brush(max(3, self.canvas.brush - 3))
        elif k == Qt.Key_BracketRight:
            self.change_brush(min(120, self.canvas.brush + 3))
        elif k in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self.save_current()
        super().closeEvent(ev)


if __name__ == "__main__":
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()
