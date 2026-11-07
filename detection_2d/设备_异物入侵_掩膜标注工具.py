#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
设备_异物入侵 掩膜人工标注工具(纯手工, 和土建_管片掉块 spalling 标注方式一致)

数据流:
  源图      data/设备_异物入侵/*.jpg
  人工掩膜   data/设备_异物入侵_人工掩膜/{名}_mask.png   (本工具保存到这里)
  完成记录   data/设备_异物入侵_已复核.txt

操作:
  左键拖动  涂白(把区域标为异物)
  右键拖动  擦黑(把区域标为背景)
  S         保存人工掩膜 + 标记已复核
  空格       标记已复核(无目标)
  C         清空掩膜(无目标)
  Z         撤销
  +/-       笔刷大小  ←→ 切图
  显示掩膜开关 显示/隐藏叠加层

列表颜色: 红=未标注  橙=已保存掩膜  绿=已复核
"""
import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QGroupBox, QListWidget, QListWidgetItem,
    QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QImage, QPixmap, QColor

import warnings
warnings.filterwarnings("ignore")

SRC_DIR = "data/设备_异物入侵"
MASK_DIR = "data/设备_异物入侵_人工掩膜"
REVIEWED_FILE = "data/设备_异物入侵_已复核.txt"


class DeviceIntrusionMaskTool(QMainWindow):
    def __init__(self):
        super().__init__()
        root = os.path.dirname(os.path.abspath(__file__))
        self.src_dir = os.path.join(root, SRC_DIR)
        self.mask_dir = os.path.join(root, MASK_DIR)
        self.reviewed_path = os.path.join(root, REVIEWED_FILE)
        os.makedirs(self.mask_dir, exist_ok=True)

        self.reviewed = {}
        if os.path.exists(self.reviewed_path):
            with open(self.reviewed_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.reviewed[line.strip()] = True

        self.image_files = []
        self.current_idx = 0
        self.original_image = None
        self.display_image = None
        self.mask_layer = None
        self.display_scale = 1.0
        self.image_rect = QRect()

        self.is_drawing = False
        self.last_point = None
        self.brush_size = 16
        self.history = []
        self.max_history = 50

        self.load_image_list()
        self.init_ui()
        if self.image_files:
            self.load_image(0)

    def load_image_list(self):
        files = []
        for fn in sorted(os.listdir(self.src_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                name = os.path.splitext(fn)[0]
                files.append({"name": name, "path": os.path.join(self.src_dir, fn)})
        self.image_files = files
        print(f"共 {len(files)} 张源图")

    def mask_state(self, name):
        if name in self.reviewed:
            return 2
        if os.path.exists(os.path.join(self.mask_dir, f"{name}_mask.png")):
            return 1
        return 0

    def update_file_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for f in self.image_files:
            name = f["name"]
            st = self.mask_state(name)
            tag = {0: "", 1: " [有掩膜]", 2: " [已复核]"}[st]
            item = QListWidgetItem(f"{name}{tag}")
            if st == 2:
                item.setForeground(QColor(120, 220, 120))
            elif st == 1:
                item.setForeground(QColor(240, 180, 90))
            else:
                item.setForeground(QColor(230, 100, 100))
            if name == self.image_files[self.current_idx]["name"]:
                item.setBackground(QColor(0, 100, 200, 80))
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)

    def init_ui(self):
        self.setWindowTitle("设备_异物入侵 掩膜标注工具")
        self.setGeometry(100, 100, 1250, 900)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_label = QLabel()
        self.canvas_label.setAlignment(Qt.AlignCenter)
        self.canvas_label.setMinimumSize(850, 720)
        self.canvas_label.setStyleSheet("background-color: #1a1a2e; border: 2px solid #333;")
        self.canvas_label.setMouseTracking(True)
        self.canvas_label.mousePressEvent = self.mouse_press
        self.canvas_label.mouseMoveEvent = self.mouse_move
        self.canvas_label.mouseReleaseEvent = self.mouse_release
        left_layout.addWidget(self.canvas_label)
        self.info_label = QLabel("就绪")
        self.info_label.setStyleSheet("color: #aaa; font-size: 12px; padding: 5px;")
        left_layout.addWidget(self.info_label)
        layout.addWidget(left, stretch=3)

        right = QWidget()
        right.setFixedWidth(300)
        right_layout = QVBoxLayout(right)

        g_list = QGroupBox("图片列表")
        gl = QVBoxLayout(g_list)
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(
            lambda item: self.load_image(self.image_files[self.file_list.row(item)]["name"]))
        gl.addWidget(self.file_list)
        right_layout.addWidget(g_list)

        g_brush = QGroupBox("笔刷")
        bl = QVBoxLayout(g_brush)
        row = QHBoxLayout()
        row.addWidget(QLabel("大小:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(3, 100)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.valueChanged.connect(self.on_brush_changed)
        row.addWidget(self.brush_slider)
        self.brush_label = QLabel(f"{self.brush_size}")
        self.brush_label.setFixedWidth(30)
        row.addWidget(self.brush_label)
        bl.addLayout(row)
        right_layout.addWidget(g_brush)

        g_view = QGroupBox("显示")
        vl = QVBoxLayout(g_view)
        self.show_mask_cb = QCheckBox("显示掩膜叠加")
        self.show_mask_cb.setChecked(True)
        self.show_mask_cb.stateChanged.connect(self.update_canvas)
        vl.addWidget(self.show_mask_cb)
        right_layout.addWidget(g_view)

        g_act = QGroupBox("操作")
        al = QVBoxLayout(g_act)
        b_undo = QPushButton("撤销 (Z)"); b_undo.clicked.connect(self.undo); al.addWidget(b_undo)
        b_clear = QPushButton("清空掩膜 (C)"); b_clear.clicked.connect(self.clear_mask); al.addWidget(b_clear)
        b_pass = QPushButton("直接通过/不改 (空格)"); b_pass.clicked.connect(self.mark_reviewed); al.addWidget(b_pass)
        b_save = QPushButton("保存 + 标记已复核 (S)")
        b_save.setStyleSheet("background-color: #2d6a4f; color: white; font-size: 14px; padding: 10px;")
        b_save.clicked.connect(self.save_mask); al.addWidget(b_save)
        nav = QHBoxLayout()
        b_prev = QPushButton("◀ 上一张"); b_prev.clicked.connect(self.prev_image)
        b_next = QPushButton("下一张 ▶"); b_next.clicked.connect(self.next_image)
        nav.addWidget(b_prev); nav.addWidget(b_next)
        al.addLayout(nav)
        right_layout.addWidget(g_act)

        hint = QLabel(
            "左键=涂白(标异物)  右键=擦除\n"
            "S 保存+已复核  空格 无目标/跳过\n"
            "Z 撤销  C 清空  +/- 笔刷  ←→ 切换"
        )
        hint.setStyleSheet("color:#888; font-size:11px; background-color:#2d2d44; padding:10px; border-radius:5px;")
        right_layout.addWidget(hint)
        right_layout.addStretch()
        layout.addWidget(right, stretch=1)

        self.update_file_list()

    def load_image(self, name):
        for i, f in enumerate(self.image_files):
            if f["name"] == name:
                self.current_idx = i
                break
        info = self.image_files[self.current_idx]
        img = cv2.imread(info["path"])
        if img is None:
            QMessageBox.warning(self, "错误", f"无法加载: {info['path']}")
            return
        self.original_image = img.copy()
        h, w = img.shape[:2]

        max_w, max_h = 860, 720
        self.display_scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * self.display_scale), int(h * self.display_scale)
        self.display_image = cv2.cvtColor(cv2.resize(img, (new_w, new_h)), cv2.COLOR_BGR2RGB)

        # 纯手工标注: 画布从空掩膜开始(不加载任何已有掩膜)
        self.mask_layer = np.zeros((new_h, new_w), np.uint8)
        self.history = []

        self.update_canvas()
        self.update_status()
        self.update_file_list()

    def update_canvas(self, *args):
        if self.display_image is None or self.mask_layer is None:
            return
        canvas = self.display_image.copy()
        if self.show_mask_cb.isChecked() and self.mask_layer.max() > 0:
            green = np.zeros_like(canvas)
            green[self.mask_layer > 127] = [0, 255, 120]
            canvas = cv2.addWeighted(canvas, 0.65, green, 0.35, 0)
        h, w, ch = canvas.shape
        qt = QImage(canvas.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt)
        self.canvas_label.setPixmap(pixmap)
        size = self.canvas_label.size()
        self.image_rect = QRect(
            (size.width() - pixmap.width()) // 2,
            (size.height() - pixmap.height()) // 2,
            pixmap.width(), pixmap.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 窗口尺寸变化后重算画布位置, 防止点击坐标错位
        if self.display_image is not None:
            self.update_canvas()

    def update_status(self):
        info = self.image_files[self.current_idx]
        st = self.mask_state(info["name"])
        status = {0: "无掩膜", 1: "有掩膜(未确认)", 2: "已复核"}[st]
        pct = (self.mask_layer > 127).sum() / max(1, self.mask_layer.size) * 100
        self.info_label.setText(
            f"[{self.current_idx+1}/{len(self.image_files)}] {info['name']}  状态:{status}  "
            f"掩膜占比:{pct:.1f}%  笔刷:{self.brush_size}px")
        self.setWindowTitle(f"设备_异物入侵掩膜标注 - {info['name']}")

    def get_image_pos(self, pos):
        x = pos.x() - self.image_rect.left()
        y = pos.y() - self.image_rect.top()
        if x < 0 or y < 0 or x >= self.image_rect.width() or y >= self.image_rect.height():
            return None
        return x, y

    def _push_history(self):
        self.history.append(self.mask_layer.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def mouse_press(self, event):
        pos = self.get_image_pos(event.pos())
        if pos is None:
            self.info_label.setText(
                f"点击在画布外 (event={event.pos().x()},{event.pos().y()}, "
                f"rect={self.image_rect.x()},{self.image_rect.y()},{self.image_rect.width()}x{self.image_rect.height()})")
            return
        self.is_drawing = True
        self.last_point = pos
        self._draw_value = 255 if event.button() == Qt.LeftButton else 0
        self._push_history()
        self.paint(pos)

    def mouse_move(self, event):
        if self.is_drawing:
            pos = self.get_image_pos(event.pos())
            if pos is not None:
                cv2.line(self.mask_layer,
                         (int(self.last_point[0]), int(self.last_point[1])),
                         (int(pos[0]), int(pos[1])),
                         self._draw_value, self.brush_size)
                self.last_point = pos
                self.update_canvas()

    def mouse_release(self, event):
        self.is_drawing = False

    def paint(self, pos):
        cv2.circle(self.mask_layer, (int(pos[0]), int(pos[1])),
                   self.brush_size, self._draw_value, -1)
        self.update_canvas()

    def undo(self):
        if not self.history:
            return
        self.mask_layer = self.history.pop()
        self.update_canvas()
        self.update_status()

    def clear_mask(self):
        self._push_history()
        self.mask_layer[:] = 0
        self.update_canvas()
        self.update_status()

    def mark_reviewed(self):
        info = self.image_files[self.current_idx]
        self.reviewed[info["name"]] = True
        self.write_reviewed()
        self.update_status()
        self.update_file_list()

    def save_mask(self):
        if self.original_image is None:
            return
        info = self.image_files[self.current_idx]
        h, w = self.original_image.shape[:2]
        mask = cv2.resize(self.mask_layer, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        out = os.path.join(self.mask_dir, f"{info['name']}_mask.png")
        cv2.imwrite(out, mask)
        if mask.max() == 0 and os.path.exists(out):
            os.remove(out)
        self.mark_reviewed()
        print(f"已保存: {out}")

    def write_reviewed(self):
        with open(self.reviewed_path, "w", encoding="utf-8") as f:
            for name in sorted(self.reviewed):
                f.write(name + "\n")

    def prev_image(self):
        if self.current_idx > 0:
            self.load_image(self.image_files[self.current_idx - 1]["name"])

    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.load_image(self.image_files[self.current_idx + 1]["name"])

    def on_brush_changed(self, value):
        self.brush_size = value
        self.brush_label.setText(f"{value}")

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Left:
            self.prev_image()
        elif k == Qt.Key_Right:
            self.next_image()
        elif k == Qt.Key_S:
            self.save_mask()
        elif k == Qt.Key_Z:
            self.undo()
        elif k == Qt.Key_C:
            self.clear_mask()
        elif k == Qt.Key_Space:
            self.mark_reviewed()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.brush_size = min(100, self.brush_size + 4)
            self.brush_slider.setValue(self.brush_size)
        elif k == Qt.Key_Minus:
            self.brush_size = max(3, self.brush_size - 4)
            self.brush_slider.setValue(self.brush_size)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet("""
    QMainWindow { background-color: #1a1a2e; }
    QLabel { color: #eee; }
    QGroupBox { color: #eee; border: 1px solid #444; border-radius: 8px;
                margin-top: 10px; padding-top: 10px; font-weight: bold; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
    QPushButton { background-color: #2d2d44; color: #eee; border: 1px solid #444;
                  border-radius: 5px; padding: 8px; font-weight: bold; }
    QPushButton:hover { background-color: #3d3d5c; }
    QListWidget { background-color: #1a1a2e; color: #eee; border: 1px solid #444; border-radius: 5px; }
    QSlider::handle { background: #3d3d5c; border: 1px solid #555; }
    QCheckBox { color: #eee; }
    """)
    window = DeviceIntrusionMaskTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
