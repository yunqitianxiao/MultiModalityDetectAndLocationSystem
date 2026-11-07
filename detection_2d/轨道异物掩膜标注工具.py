#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轨道异物掩膜标注工具
- 目标: data/轨道_异物/ 下按类别子目录组织的异物图像(如 person/ stone/ vehicle/ ...)
- 约定: 异物涂白(255), 背景涂黑(0)  ← 与所有训练脚本一致(掩膜 >127 = 病害)
- 保存: data/轨道_异物_手动标注的掩膜/<类别子目录>/<同名>.png(结构镜像输入目录)
- 操作:
    左键拖动 : 画异物(白色)         右键拖动 : 擦除(黑色)
    滚轮     : 缩放(以光标为中心)   中键拖动 : 平移
    [  /  ]  : 画笔变小 / 变大       U : 撤销    C : 清空
    A / D    : 上一张 / 下一张(自动保存)
    S        : 保存                  F : 适应窗口
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
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IMAGE_ROOT = os.path.join(PROJECT_ROOT, "data", "设备_异物入侵")
MASK_ROOT = os.path.join(PROJECT_ROOT, "data", "设备_异物入侵_手动标注的掩膜")
os.makedirs(MASK_ROOT, exist_ok=True)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


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
        self.on_change = None
        self.setMouseTracking(True)
        self.setMinimumSize(520, 520)

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
        self.setWindowTitle("轨道异物掩膜标注工具 (白=异物 / 黑=背景)")
        self.setFocusPolicy(Qt.StrongFocus)

        self.canvas = MaskCanvas()
        self.canvas.on_change = self.update_status
        self.entries = self._list_images()   # (相对子目录, 文件名)
        self.current = 0

        self._build_ui()
        if self.entries:
            self.load_image(0)
            QTimer.singleShot(120, self.canvas.fit_view)
        else:
            print(f"❌ 未找到 {IMAGE_ROOT}/ 下的图片, 请检查目录结构")

    def _list_images(self):
        if not os.path.isdir(IMAGE_ROOT):
            return []
        out = []
        for root, _, files in os.walk(IMAGE_ROOT):
            rel = os.path.relpath(root, IMAGE_ROOT)
            for f in sorted(files):
                if f.lower().endswith(IMAGE_EXTS):
                    out.append((rel, f))
        return out

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.addWidget(self.canvas, stretch=1)

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

        info = QGroupBox("提示")
        vl = QVBoxLayout(info)
        self.status_label = QLabel("")
        vl.addWidget(self.status_label)
        vl.addWidget(QLabel(
            "左键=涂异物(白)  右键=擦除(黑)  滚轮=缩放  中键拖动=平移  [ ]=笔刷  "
            "U=撤销  C=清空  A/D=翻页  S=保存  Q=退出"))
        root.addWidget(info)

        self.setCentralWidget(central)
        self.resize(1100, 800)

    # ---------- 逻辑 ----------
    def current_rel(self):
        return self.entries[self.current]

    def load_image(self, idx):
        if self.current != idx:
            self.save_current()
        self.current = idx
        rel, fname = self.entries[idx]
        img = cv2.imread(os.path.join(IMAGE_ROOT, rel, fname))
        if img is None:
            print(f"❌ 读取失败: {rel}/{fname}")
            return
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        base = os.path.splitext(fname)[0]
        for ext in (".png", ".jpg", ".jpeg"):
            mp = os.path.join(MASK_ROOT, rel, f"{base}{ext}")
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
        if not self.entries or self.canvas.mask is None:
            return
        rel, fname = self.entries[self.current]
        base = os.path.splitext(fname)[0]
        out_dir = os.path.join(MASK_ROOT, rel)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{base}.png")
        cv2.imwrite(out, self.canvas.mask)
        for ext in (".jpg", ".jpeg"):
            old = os.path.join(out_dir, f"{base}{ext}")
            if os.path.exists(old):
                os.remove(old)
        print(f"  ✅ 已保存: {out}")

    def update_status(self):
        if not self.entries or self.canvas.mask is None:
            return
        rel, fname = self.entries[self.current]
        fg = int((self.canvas.mask > 0).sum())
        total = len(self.entries)
        self.status_label.setText(
            f"[{self.current + 1}/{total}] 类别={rel or '(根目录)'}  {fname}   "
            f"已标异物像素: {fg}  笔刷: {self.canvas.brush}  缩放: {self.canvas.scale:.2f}")
        self.brush_label.setText(f"笔刷: {self.canvas.brush}")
        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(self.canvas.brush)
        self.brush_slider.blockSignals(False)

    def change_brush(self, v):
        if v is not None and not isinstance(v, bool):
            self.canvas.brush = int(v)
            self.update_status()

    def prev(self):
        if self.entries:
            self.load_image((self.current - 1) % len(self.entries))

    def next_(self):
        if self.entries:
            self.load_image((self.current + 1) % len(self.entries))

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
