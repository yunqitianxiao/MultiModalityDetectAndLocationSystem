#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
画笔掩膜标注工具 - 修正版
直接在图片上涂抹进行标注
"""
import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QGroupBox,
    QMessageBox, QListWidget, QListWidgetItem, QColorDialog
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QImage, QPixmap, QColor

import warnings
warnings.filterwarnings("ignore")


class BrushMaskAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()

        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
        DATA_DIR = os.path.join(PROJECT_ROOT, "data")

        self.image_dir = os.path.join(DATA_DIR, "土建_管片破损掉块")
        self.output_dir = os.path.join(DATA_DIR, "土建_管片破损掉块_掩膜")
        os.makedirs(self.output_dir, exist_ok=True)

        self.image_files = []
        self.current_idx = 0
        self.original_image = None
        self.display_image = None
        self.drawing_layer = None
        self.display_scale = 1.0
        self.image_rect = QRect()

        self.is_drawing = False
        self.brush_size = 20
        self.brush_color = QColor(0, 255, 0, 128)

        self.load_image_list()
        self.init_ui()

        if self.image_files:
            self.load_image(0)

    def load_image_list(self):
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
            print(f"已创建目录: {self.image_dir}")
            return

        self.image_files = [
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        self.image_files.sort()
        print(f"找到 {len(self.image_files)} 张图片")

    def init_ui(self):
        self.setWindowTitle("画笔掩膜标注工具")
        self.setGeometry(100, 100, 1200, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas_label = QLabel()
        self.canvas_label.setAlignment(Qt.AlignCenter)
        self.canvas_label.setMinimumSize(800, 700)
        self.canvas_label.setStyleSheet("background-color: #1a1a2e; border: 2px solid #333;")
        self.canvas_label.setMouseTracking(True)
        self.canvas_label.mousePressEvent = self.mouse_press
        self.canvas_label.mouseMoveEvent = self.mouse_move
        self.canvas_label.mouseReleaseEvent = self.mouse_release

        left_layout.addWidget(self.canvas_label)

        self.info_label = QLabel("就绪")
        self.info_label.setStyleSheet("color: #aaa; font-size: 12px; padding: 5px;")
        left_layout.addWidget(self.info_label)

        right_widget = QWidget()
        right_widget.setFixedWidth(280)
        right_layout = QVBoxLayout(right_widget)

        group_file = QGroupBox("图片列表")
        file_layout = QVBoxLayout(group_file)
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(lambda item: self.load_image(self.file_list.row(item)))
        file_layout.addWidget(self.file_list)
        right_layout.addWidget(group_file)

        group_brush = QGroupBox("画笔设置")
        brush_layout = QVBoxLayout(group_brush)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("笔刷:"))
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setMinimum(5)
        self.brush_size_slider.setMaximum(50)
        self.brush_size_slider.setValue(self.brush_size)
        self.brush_size_slider.valueChanged.connect(self.on_brush_size_changed)
        size_layout.addWidget(self.brush_size_slider)
        self.brush_size_label = QLabel(f"{self.brush_size}px")
        size_layout.addWidget(self.brush_size_label)
        brush_layout.addLayout(size_layout)

        self.color_btn = QPushButton("选择颜色")
        self.color_btn.setStyleSheet("background-color: #00ff00;")
        self.color_btn.clicked.connect(self.choose_color)
        brush_layout.addWidget(self.color_btn)

        right_layout.addWidget(group_brush)

        group_action = QGroupBox("操作")
        action_layout = QVBoxLayout(group_action)

        btn_clear = QPushButton("清除画布")
        btn_clear.clicked.connect(self.clear_canvas)
        action_layout.addWidget(btn_clear)

        btn_save = QPushButton("保存掩膜")
        btn_save.setStyleSheet("background-color: #2d6a4f; color: white; font-size: 14px; padding: 10px;")
        btn_save.clicked.connect(self.save_mask)
        action_layout.addWidget(btn_save)

        nav_layout = QHBoxLayout()
        btn_prev = QPushButton("◀")
        btn_prev.clicked.connect(self.prev_image)
        btn_next = QPushButton("▶")
        btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        action_layout.addLayout(nav_layout)

        right_layout.addWidget(group_action)

        hint = QLabel(
            "使用说明:\n"
            "1. 选择图片\n"
            "2. 调节笔刷大小\n"
            "3. 在图片上涂抹\n"
            "4. 保存\n\n"
            "← → 切换图片\n"
            "C 清除\n"
            "S 保存"
        )
        hint.setStyleSheet("color: #888; font-size: 11px; background-color: #2d2d44; padding: 10px; border-radius: 5px;")
        right_layout.addWidget(hint)

        right_layout.addStretch()

        main_layout.addWidget(left_widget, stretch=3)
        main_layout.addWidget(right_widget, stretch=1)

        self.update_file_list()

    def update_file_list(self):
        self.file_list.clear()
        for i, f in enumerate(self.image_files):
            item = QListWidgetItem(f"{i + 1}. {f}")
            if i == self.current_idx:
                item.setBackground(QColor(0, 100, 200, 50))
            self.file_list.addItem(item)

    def load_image(self, idx):
        if not self.image_files or idx < 0 or idx >= len(self.image_files):
            return

        self.current_idx = idx
        filename = self.image_files[idx]
        filepath = os.path.join(self.image_dir, filename)

        self.original_image = cv2.imread(filepath)
        if self.original_image is None:
            QMessageBox.warning(self, "错误", f"无法加载图片: {filename}")
            return

        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        h, w = self.original_image.shape[:2]

        max_w = 850
        max_h = 700
        self.display_scale = min(max_w / w, max_h / h, 1.0)

        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)

        self.display_image = cv2.resize(self.original_image, (new_w, new_h))
        self.drawing_layer = np.zeros((new_h, new_w, 4), dtype=np.uint8)

        self.update_canvas()
        self.update_status()
        self.update_file_list()

    def update_canvas(self):
        if self.display_image is None:
            return

        canvas = self.display_image.copy()

        mask = self.drawing_layer[:, :, 3] > 0
        if mask.any():
            overlay = np.zeros_like(canvas)
            overlay[mask] = self.drawing_layer[mask, :3]
            alpha = self.drawing_layer[:, :, 3] / 255.0
            for c in range(3):
                canvas[:, :, c] = (canvas[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha).astype(np.uint8)

        h, w, ch = canvas.shape
        bytes_per_line = ch * w
        qt_img = QImage(canvas.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        self.canvas_label.setPixmap(pixmap)

        label_size = self.canvas_label.size()
        self.image_rect = QRect(
            (label_size.width() - pixmap.width()) // 2,
            (label_size.height() - pixmap.height()) // 2,
            pixmap.width(),
            pixmap.height()
        )

    def update_status(self):
        if self.image_files:
            self.info_label.setText(f"{self.current_idx + 1}/{len(self.image_files)}: {self.image_files[self.current_idx]}")
            self.setWindowTitle(f"掩膜标注 - {self.image_files[self.current_idx]}")

    def get_image_pos(self, pos):
        x = pos.x() - self.image_rect.left()
        y = pos.y() - self.image_rect.top()

        if x < 0 or y < 0 or x >= self.image_rect.width() or y >= self.image_rect.height():
            return None

        return x, y

    def mouse_press(self, event):
        pos = self.get_image_pos(event.pos())
        if pos is None:
            return

        self.is_drawing = True
        self.draw_point(pos)

    def mouse_move(self, event):
        if self.is_drawing:
            pos = self.get_image_pos(event.pos())
            if pos is not None:
                self.draw_point(pos)
                self.update_canvas()

    def mouse_release(self, event):
        self.is_drawing = False

    def draw_point(self, pos):
        x, y = pos
        r, g, b = self.brush_color.red(), self.brush_color.green(), self.brush_color.blue()
        a = self.brush_color.alpha()

        cv2.circle(self.drawing_layer, (int(x), int(y)), self.brush_size, (b, g, r, a), -1)

    def on_brush_size_changed(self, value):
        self.brush_size = value
        self.brush_size_label.setText(f"{value}px")

    def choose_color(self):
        color = QColorDialog.getColor(self.brush_color)
        if color.isValid():
            color.setAlpha(128)
            self.brush_color = color
            self.color_btn.setStyleSheet(f"background-color: {color.name()};")

    def clear_canvas(self):
        if self.drawing_layer is not None:
            h, w = self.drawing_layer.shape[:2]
            self.drawing_layer = np.zeros((h, w, 4), dtype=np.uint8)
            self.update_canvas()

    def save_mask(self):
        if self.original_image is None:
            return

        filename = self.image_files[self.current_idx]
        output_path = os.path.join(self.output_dir, filename.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png'))

        h, w = self.original_image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)

        scaled_mask = cv2.resize(self.drawing_layer[:, :, 3], (w, h))
        mask[scaled_mask > 127] = 255

        cv2.imwrite(output_path, mask)

        QMessageBox.information(self, "保存成功", f"掩膜已保存:\n{output_path}")

    def prev_image(self):
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)

    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.load_image(self.current_idx + 1)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_C:
            self.clear_canvas()
        elif event.key() == Qt.Key_S:
            self.save_mask()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    dark_style = """
    QMainWindow { background-color: #1a1a2e; }
    QLabel { color: #eee; }
    QGroupBox { color: #eee; border: 1px solid #444; border-radius: 8px; margin-top: 10px; padding-top: 10px; font-weight: bold; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
    QPushButton { background-color: #2d2d44; color: #eee; border: 1px solid #444; border-radius: 5px; padding: 8px; font-weight: bold; }
    QPushButton:hover { background-color: #3d3d5c; }
    QListWidget { background-color: #1a1a2e; color: #eee; border: 1px solid #444; border-radius: 5px; }
    QSlider::groove { border: 1px solid #444; }
    QSlider::handle { background: #3d3d5c; border: 1px solid #555; }
    """
    app.setStyleSheet(dark_style)

    window = BrushMaskAnnotationTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
