#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
管片破损掉块土建数据集掩膜标注工具
针对 病害训练集+正常状态 中的图片进行掩膜标注
"""
import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QGroupBox, QComboBox,
    QMessageBox, QListWidget, QListWidgetItem, QColorDialog, QCheckBox
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QImage, QPixmap, QColor

import warnings
warnings.filterwarnings("ignore")


class SegmentDamageMaskTool(QMainWindow):
    def __init__(self):
        super().__init__()

        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = SCRIPT_DIR

        self.data_root = os.path.join(PROJECT_ROOT, "病害训练集+正常状态")
        self.train_images = os.path.join(self.data_root, "病害训练集", "images")
        self.train_labels = os.path.join(self.data_root, "病害训练集", "labels")
        self.normal_civil = os.path.join(self.data_root, "正常状态", "土建")

        self.output_dir = os.path.join(PROJECT_ROOT, "管片破损掉块_掩膜")
        os.makedirs(self.output_dir, exist_ok=True)

        self.train_mask_dir = os.path.join(self.output_dir, "病害训练集_掩膜")
        os.makedirs(self.train_mask_dir, exist_ok=True)

        self.image_files = []
        self.label_files = []
        self.current_idx = 0
        self.original_image = None
        self.display_image = None
        self.drawing_layer = None
        self.display_scale = 1.0
        self.image_rect = QRect()

        self.is_drawing = False
        self.brush_size = 20
        self.brush_color = QColor(0, 255, 0, 128)

        self.show_bbox = True
        self.target_class = 5

        self.class_names = {
            0: "扣件歪斜",
            1: "扣件断裂",
            2: "扣件缺失",
            3: "异物入侵",
            4: "管线松脱",
            5: "管片破损掉块",
            6: "裂缝",
            7: "渗漏水"
        }

        self.load_image_list()
        self.init_ui()

        if self.image_files:
            self.load_image(0)

    def load_image_list(self):
        if os.path.exists(self.train_images):
            images = [f for f in os.listdir(self.train_images)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and f.startswith('05')]
            for img in images:
                name_without_ext = os.path.splitext(img)[0]
                self.image_files.append({
                    'path': os.path.join(self.train_images, img),
                    'name': img,
                    'type': 'train',
                    'label_name': name_without_ext + '.txt'
                })

        print(f"找到 {len(self.image_files)} 张管片破损掉块图片 (05开头)")

    def init_ui(self):
        self.setWindowTitle("管片破损掉块掩膜标注工具")
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
        self.brush_size_slider.setMaximum(100)
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

        group_view = QGroupBox("显示设置")
        view_layout = QVBoxLayout(group_view)
        self.show_bbox_checkbox = QCheckBox("显示边界框")
        self.show_bbox_checkbox.setChecked(True)
        self.show_bbox_checkbox.stateChanged.connect(self.on_show_bbox_changed)
        view_layout.addWidget(self.show_bbox_checkbox)

        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("目标类别:"))
        self.class_combo = QComboBox()
        for cls_id, cls_name in self.class_names.items():
            self.class_combo.addItem(f"{cls_id}-{cls_name}", cls_id)
        self.class_combo.setCurrentIndex(5)
        self.class_combo.currentIndexChanged.connect(self.on_class_changed)
        class_layout.addWidget(self.class_combo)
        view_layout.addLayout(class_layout)

        self.class_info_label = QLabel("当前: 5-管片破损掉块")
        self.class_info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        view_layout.addWidget(self.class_info_label)

        right_layout.addWidget(group_view)

        group_action = QGroupBox("操作")
        action_layout = QVBoxLayout(group_action)

        btn_clear = QPushButton("清除画布")
        btn_clear.clicked.connect(self.clear_canvas)
        action_layout.addWidget(btn_clear)

        btn_undo = QPushButton("撤销上次涂抹")
        btn_undo.clicked.connect(self.undo_last)
        action_layout.addWidget(btn_undo)

        btn_save = QPushButton("保存掩膜")
        btn_save.setStyleSheet("background-color: #2d6a4f; color: white; font-size: 14px; padding: 10px;")
        btn_save.clicked.connect(self.save_mask)
        action_layout.addWidget(btn_save)

        btn_batch = QPushButton("批量生成管片破损掩膜")
        btn_batch.setStyleSheet("background-color: #6b4c9a; color: white; padding: 8px;")
        btn_batch.clicked.connect(self.batch_generate_masks)
        action_layout.addWidget(btn_batch)

        nav_layout = QHBoxLayout()
        btn_prev = QPushButton("◀ 上一张")
        btn_prev.clicked.connect(self.prev_image)
        btn_next = QPushButton("下一张 ▶")
        btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        action_layout.addLayout(nav_layout)

        right_layout.addWidget(group_action)

        hint = QLabel(
            "使用说明:\n"
            "1. 选择或切换图片\n"
            "2. 调节笔刷大小和颜色\n"
            "3. 在图片上涂抹标注病害区域\n"
            "4. 保存掩膜\n\n"
            "快捷键:\n"
            "← → 切换图片\n"
            "C 清除画布\n"
            "Z 撤销\n"
            "S 保存\n"
            "B 显示/隐藏边界框"
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
            display_name = f['name']
            if f['type'] == 'normal':
                display_name = f"[正常] {f['name']}"
            item = QListWidgetItem(f"{i + 1}. {display_name}")
            if i == self.current_idx:
                item.setBackground(QColor(0, 100, 200, 50))
            self.file_list.addItem(item)

    def load_image(self, idx):
        if not self.image_files or idx < 0 or idx >= len(self.image_files):
            return

        self.current_idx = idx
        img_info = self.image_files[idx]
        filepath = img_info['path']

        self.original_image = cv2.imread(filepath)
        if self.original_image is None:
            QMessageBox.warning(self, "错误", f"无法加载图片: {img_info['name']}")
            return

        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        h, w = self.original_image.shape[:2]

        max_w = 850
        max_h = 700
        self.display_scale = min(max_w / w, max_h / h, 1.0)

        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)

        self.display_image = cv2.resize(self.original_image.copy(), (new_w, new_h))
        self.drawing_layer = np.zeros((new_h, new_w, 4), dtype=np.uint8)

        self.load_existing_mask(img_info)

        self.update_canvas()
        self.update_status()
        self.update_file_list()

    def load_existing_mask(self, img_info):
        pass

    def draw_bboxes_on_image(self, img):
        img_info = self.image_files[self.current_idx]
        if img_info['type'] != 'train' or not img_info['label_name']:
            return img

        label_path = os.path.join(self.train_labels, img_info['label_name'])
        if not os.path.exists(label_path):
            return img

        h, w = img.shape[:2]
        img_with_bbox = img.copy()

        with open(label_path, 'r') as f:
            lines = f.readlines()

        target_class = self.class_combo.currentData()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            if cls_id != target_class:
                continue

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            color = (255, 100, 100)
            cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_id}:{self.class_names.get(cls_id, 'unknown')}"
            cv2.putText(img_with_bbox, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img_with_bbox

    def update_canvas(self):
        if self.display_image is None or self.drawing_layer is None:
            return

        canvas = self.display_image.copy()

        if self.show_bbox:
            canvas = self.draw_bboxes_on_image(canvas)

        alpha_mask = self.drawing_layer[:, :, 3] / 255.0
        if alpha_mask.max() > 0:
            overlay_colors = self.drawing_layer[:, :, :3].astype(np.float32) / 255.0
            for c in range(3):
                canvas[:, :, c] = (canvas[:, :, c].astype(np.float32) * (1 - alpha_mask) + 
                                   overlay_colors[:, :, c] * alpha_mask).astype(np.uint8)

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
            img_info = self.image_files[self.current_idx]
            self.info_label.setText(
                f"{self.current_idx + 1}/{len(self.image_files)}: {img_info['name']} "
                f"[{img_info['type']}]"
            )
            self.setWindowTitle(f"管片破损掩膜标注 - {img_info['name']}")

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
        self.last_point = pos
        self.draw_point(pos)

    def mouse_move(self, event):
        if self.is_drawing:
            pos = self.get_image_pos(event.pos())
            if pos is not None:
                cv2.line(self.drawing_layer, 
                        (int(self.last_point[0]), int(self.last_point[1])),
                        (int(pos[0]), int(pos[1])),
                        (self.brush_color.blue(), self.brush_color.green(), self.brush_color.red(), self.brush_color.alpha()),
                        self.brush_size * 2)
                self.last_point = pos
                self.update_canvas()

    def mouse_release(self, event):
        self.is_drawing = False

    def draw_point(self, pos):
        x, y = pos
        r, g, b = self.brush_color.red(), self.brush_color.green(), self.brush_color.blue()
        a = self.brush_color.alpha()

        cv2.circle(self.drawing_layer, (int(x), int(y)), self.brush_size, (b, g, r, a), -1)
        self.update_canvas()

    def on_brush_size_changed(self, value):
        self.brush_size = value
        self.brush_size_label.setText(f"{value}px")

    def choose_color(self):
        color = QColorDialog.getColor(self.brush_color)
        if color.isValid():
            color.setAlpha(128)
            self.brush_color = color
            self.color_btn.setStyleSheet(f"background-color: {color.name()};")

    def on_show_bbox_changed(self, state):
        self.show_bbox = (state == Qt.Checked)
        self.update_canvas()

    def on_class_changed(self, index):
        cls_id = self.class_combo.currentData()
        self.target_class = cls_id
        self.class_info_label.setText(f"当前: {cls_id}-{self.class_names.get(cls_id, 'unknown')}")
        self.update_canvas()

    def clear_canvas(self):
        if self.drawing_layer is not None:
            h, w = self.drawing_layer.shape[:2]
            self.drawing_layer = np.zeros((h, w, 4), dtype=np.uint8)
            self.update_canvas()

    def undo_last(self):
        pass

    def save_mask(self):
        if self.original_image is None:
            return

        img_info = self.image_files[self.current_idx]
        basename = os.path.splitext(img_info['label_name'])[0] if img_info['label_name'] else os.path.splitext(img_info['name'])[0]
        output_name = f"{basename}_mask.png"
        output_path = os.path.join(self.train_mask_dir, output_name)

        h, w = self.original_image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)

        scaled_mask = cv2.resize(self.drawing_layer[:, :, 3], (w, h))
        mask[scaled_mask > 127] = 255

        cv2.imwrite(output_path, mask)

        QMessageBox.information(self, "保存成功", f"掩膜已保存:\n{output_path}")

    def batch_generate_masks(self):
        msg = QMessageBox.question(
            self, "批量生成掩膜",
            "这将基于YOLO边界框自动生成管片破损掉块(类别5)的掩膜。\n"
            "掩膜将保存到: 管片破损掉块_掩膜/病害训练集_掩膜/\n\n"
            "是否继续?",
            QMessageBox.Yes | QMessageBox.No
        )

        if msg != QMessageBox.Yes:
            return

        target_class = self.class_combo.currentData()
        count = 0

        for img_info in self.image_files:
            if img_info['type'] != 'train' or not img_info['label_name']:
                continue

            img_path = img_info['path']
            label_path = os.path.join(self.train_labels, img_info['label_name'])

            if not os.path.exists(label_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                if cls_id != target_class:
                    continue

                cx, cy, bw, bh = map(float, parts[1:5])

                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)

                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            output_name = img_info['label_name'].replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png').replace('.JPG', '_mask.png')
            output_path = os.path.join(self.train_mask_dir, output_name)
            cv2.imwrite(output_path, mask)
            count += 1

        QMessageBox.information(self, "完成", f"已生成 {count} 个掩膜文件")

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
        elif event.key() == Qt.Key_B:
            self.show_bbox = not self.show_bbox
            self.show_bbox_checkbox.setChecked(self.show_bbox)


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
    QComboBox { background-color: #2d2d44; color: #eee; border: 1px solid #444; border-radius: 5px; padding: 5px; }
    QCheckBox { color: #eee; }
    """
    app.setStyleSheet(dark_style)

    window = SegmentDamageMaskTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
