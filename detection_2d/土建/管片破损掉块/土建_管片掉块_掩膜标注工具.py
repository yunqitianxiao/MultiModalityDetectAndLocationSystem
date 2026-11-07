#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
土建_管片掉块 掩膜人工标注工具(纯手工)

数据流(脚本位于 pro/2/土建/管片破损掉块/ 下, 路径相对 pro/2):
  源图        data/土建_管片掉块/*.jpg
  SAM掩膜     data/土建_管片掉块_掩膜/   (SAM生成, 仅供你自己对照, 工具不读取)
  人工掩膜    data/土建_管片掉块_人工掩膜/{名}_mask.png  (本工具只写这里)
  人工标注名单 data/土建_管片掉块_人工标注.txt (152张已人工标注, 默认跳过)
  完成记录    data/土建_管片掉块_已复核.txt

操作:
  左键拖动   画笔涂白(把区域标为缺陷)
  右键拖动   橡皮擦黑(把区域标为背景)
  S           保存当前掩膜 + 标记已复核
  C           清空当前掩膜(认为无缺陷)
  Z           撤销一次涂抹
  +/-         调整笔刷大小
  ← →         上一张 / 下一张(仅在当前列表内切换)
  空格         标记已复核(无缺陷/确认跳过, 不改动掩膜)

列表默认只显示"待标注"的图(排除已人工标注的152张),
勾选"显示已人工标注的图"可查看全部。
列表颜色: 红=待标注  绿=已复核  灰=已人工标注(跳过)
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

import PyQt5
# 修复: 完整版 opencv-python 在 import cv2 时会把 QT_QPA_PLATFORM_PLUGIN_PATH
# 改写为 cv2/qt/plugins(其自带的 Qt 插件目录), 导致 PyQt5 加载 xcb 平台插件失败。
# 这里在 cv2 import 之后把插件目录改回 PyQt5 自带的 Qt5 插件目录。
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
    os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms"
)

SRC_DIR = "../../data/土建_管片掉块"
MASK_DIR = "../../data/土建_管片掉块_人工掩膜"
REVIEWED_FILE = "../../data/土建_管片掉块_已复核.txt"
MANUAL_FILE = "../../data/土建_管片掉块_人工标注.txt"


class TunnelSpallingMaskTool(QMainWindow):
    def __init__(self):
        super().__init__()

        project_root = os.path.dirname(os.path.abspath(__file__))
        self.src_dir = os.path.join(project_root, SRC_DIR)
        self.mask_dir = os.path.join(project_root, MASK_DIR)
        self.reviewed_path = os.path.join(project_root, REVIEWED_FILE)
        self.manual_path = os.path.join(project_root, MANUAL_FILE)
        os.makedirs(self.mask_dir, exist_ok=True)

        # 已人工标注名单(默认跳过, 不要求重新标)
        self.manual = set()
        if os.path.exists(self.manual_path):
            with open(self.manual_path, "r", encoding="utf-8") as f:
                self.manual = {line.strip() for line in f if line.strip()}

        # 已复核列表: {图名: True}
        self.reviewed = {}
        if os.path.exists(self.reviewed_path):
            with open(self.reviewed_path, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        self.reviewed[name] = True

        self.image_files = []          # [{name, path}]
        self.current_idx = 0
        self.original_image = None     # 原始分辨率 BGR
        self.display_image = None      # 显示分辨率 RGB
        self.mask_layer = None         # 显示分辨率单通道 0/255
        self.display_scale = 1.0
        self.image_rect = QRect()

        self.is_drawing = False
        self.last_point = None
        self.brush_size = 16
        self.history = []              # 撤销栈(显示分辨率 mask 副本)
        self.max_history = 50

        self.show_manual = False

        self.load_image_list()
        self.init_ui()

        if self.image_files:
            self.load_image(0)

    # ---------------- 文件列表 ----------------
    def load_image_list(self):
        files = []
        for fn in sorted(os.listdir(self.src_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                name = os.path.splitext(fn)[0]
                files.append({"name": name, "path": os.path.join(self.src_dir, fn)})
        self.image_files = files
        print(f"共 {len(files)} 张源图")

    def mask_state(self, name):
        """0=待标注 1=已复核 2=已人工标注(跳过)"""
        if name in self.manual:
            return 2
        if name in self.reviewed:
            return 1
        return 0

    def visible_files(self):
        if self.show_manual:
            return self.image_files
        # 默认排除已人工标注的图(152张), 只列待标注的
        return [f for f in self.image_files if f["name"] not in self.manual]

    def update_file_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        vis = self.visible_files()
        cur = self.current_file()
        for f in vis:
            name = f["name"]
            st = self.mask_state(name)
            tag = {0: "", 1: " [已复核]", 2: " [人工]"}[st]
            item = QListWidgetItem(f"{name}{tag}")
            if st == 2:
                item.setForeground(QColor(150, 150, 150))   # 灰=已人工标注(跳过)
            elif st == 1:
                item.setForeground(QColor(120, 220, 120))   # 绿=已复核
            else:
                item.setForeground(QColor(230, 100, 100))   # 红=待标注
            if name == cur["name"]:
                item.setBackground(QColor(0, 100, 200, 80))
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)

    def current_file(self):
        return self.image_files[self.current_idx]

    # ---------------- UI ----------------
    def init_ui(self):
        self.setWindowTitle("土建_管片掉块 掩膜标注辅助工具")
        self.setGeometry(100, 100, 1250, 900)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # 左侧画布
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

        # 右侧面板
        right = QWidget()
        right.setFixedWidth(300)
        right_layout = QVBoxLayout(right)

        g_list = QGroupBox("图片列表")
        gl = QVBoxLayout(g_list)
        self.filter_cb = QCheckBox("显示已人工标注的图")
        self.filter_cb.stateChanged.connect(self.on_filter_changed)
        gl.addWidget(self.filter_cb)
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(
            lambda item: self.load_image(self.visible_files()[self.file_list.row(item)]["name"]))
        gl.addWidget(self.file_list)
        right_layout.addWidget(g_list)

        g_brush = QGroupBox("笔刷")
        bl = QVBoxLayout(g_brush)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("大小:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(3, 100)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.valueChanged.connect(self.on_brush_changed)
        size_row.addWidget(self.brush_slider)
        self.brush_label = QLabel(f"{self.brush_size}")
        self.brush_label.setFixedWidth(30)
        size_row.addWidget(self.brush_label)
        bl.addLayout(size_row)
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
        b_undo = QPushButton("撤销 (Z)")
        b_undo.clicked.connect(self.undo)
        al.addWidget(b_undo)
        b_clear = QPushButton("清空掩膜 (C)")
        b_clear.clicked.connect(self.clear_mask)
        al.addWidget(b_clear)
        b_pass = QPushButton("直接通过/不改 (空格)")
        b_pass.clicked.connect(self.mark_reviewed)
        al.addWidget(b_pass)
        b_save = QPushButton("保存 + 标记已复核 (S)")
        b_save.setStyleSheet("background-color: #2d6a4f; color: white; font-size: 14px; padding: 10px;")
        b_save.clicked.connect(self.save_mask)
        al.addWidget(b_save)
        nav = QHBoxLayout()
        b_prev = QPushButton("◀ 上一张")
        b_prev.clicked.connect(self.prev_image)
        b_next = QPushButton("下一张 ▶")
        b_next.clicked.connect(self.next_image)
        nav.addWidget(b_prev)
        nav.addWidget(b_next)
        al.addLayout(nav)
        right_layout.addWidget(g_act)

        hint = QLabel(
            "使用说明:\n"
            "左键=涂白(标缺陷)  右键=擦除\n"
            "S 保存并标记已复核\n"
            "空格 标记已复核(无缺陷)\n"
            "Z 撤销  C 清空\n"
            "+/- 笔刷大小  ←→ 切换"
        )
        hint.setStyleSheet("color:#888; font-size:11px; background-color:#2d2d44; padding:10px; border-radius:5px;")
        right_layout.addWidget(hint)

        right_layout.addStretch()
        layout.addWidget(right, stretch=1)

        self.update_file_list()

    # ---------------- 图像加载 ----------------
    def load_image(self, name):
        """name: 图名(不含扩展名)"""
        for i, f in enumerate(self.image_files):
            if f["name"] == name:
                self.current_idx = i
                break
        info = self.current_file()
        path = info["path"]

        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法加载图片: {path}")
            return
        self.original_image = img.copy()
        h, w = img.shape[:2]

        max_w, max_h = 860, 720
        self.display_scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * self.display_scale), int(h * self.display_scale)
        self.display_image = cv2.cvtColor(cv2.resize(img, (new_w, new_h)), cv2.COLOR_BGR2RGB)

        # 纯手工标注: 不加载/不显示任何已有掩膜, 画布从空掩膜开始
        self.mask_layer = np.zeros((new_h, new_w), np.uint8)
        self.history = []

        self.update_canvas()
        self.update_status()
        self.update_file_list()

    # ---------------- 画布渲染 ----------------
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

    def update_status(self):
        info = self.current_file()
        st = self.mask_state(info["name"])
        status = {0: "待标注", 1: "已复核", 2: "已人工标注(跳过)"}[st]
        area_pct = (self.mask_layer > 127).sum() / max(1, self.mask_layer.size) * 100
        vis = self.visible_files()
        idx = next((i for i, f in enumerate(vis) if f["name"] == info["name"]), -1) + 1
        total = len(vis)
        self.info_label.setText(
            f"[{idx}/{total}] {info['name']}  状态:{status}  "
            f"掩膜占比:{area_pct:.1f}%  笔刷:{self.brush_size}px")
        self.setWindowTitle(f"土建_管片掉块掩膜标注 - {info['name']}")

    # ---------------- 鼠标绘制 ----------------
    def get_image_pos(self, pos):
        x = pos.x() - self.image_rect.left()
        y = pos.y() - self.image_rect.top()
        if x < 0 or y < 0 or x >= self.image_rect.width() or y >= self.image_rect.height():
            return None
        return x, y

    def _push_history(self):
        # 一次笔画算一步撤销: 压入当前 mask 副本(在 paint/clear 前调用)
        self.history.append(self.mask_layer.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def mouse_press(self, event):
        pos = self.get_image_pos(event.pos())
        if pos is None:
            return
        self.is_drawing = True
        self.last_point = pos
        # 记录本次笔画是画还是擦(左键涂白/右键擦黑)
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

    # ---------------- 编辑操作 ----------------
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
        info = self.current_file()
        self.reviewed[info["name"]] = True
        self.write_reviewed()
        self.update_status()
        self.update_file_list()

    def save_mask(self):
        if self.original_image is None:
            return
        info = self.current_file()
        h, w = self.original_image.shape[:2]
        mask = cv2.resize(self.mask_layer, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        out = os.path.join(self.mask_dir, f"{info['name']}_mask.png")
        cv2.imwrite(out, mask)

        # 若全黑则删除空掩膜文件(无缺陷)
        if mask.max() == 0 and os.path.exists(out):
            os.remove(out)

        self.mark_reviewed()
        print(f"已保存: {out}")

    def write_reviewed(self):
        with open(self.reviewed_path, "w", encoding="utf-8") as f:
            for name in sorted(self.reviewed):
                f.write(name + "\n")

    # ---------------- 事件 ----------------
    def _nav(self, delta):
        vis = self.visible_files()
        cur = self.current_file()["name"]
        idx = next((i for i, f in enumerate(vis) if f["name"] == cur), None)
        if idx is None:
            return
        nxt = idx + delta
        if 0 <= nxt < len(vis):
            self.load_image(vis[nxt]["name"])

    def prev_image(self):
        self._nav(-1)

    def next_image(self):
        self._nav(+1)

    def on_brush_changed(self, value):
        self.brush_size = value
        self.brush_label.setText(f"{value}")

    def on_filter_changed(self, state):
        self.show_manual = (state == Qt.Checked)
        self.update_file_list()

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
        elif k == Qt.Key_Plus or k == Qt.Key_Equal:
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
    QSlider::groove { border: 1px solid #444; }
    QSlider::handle { background: #3d3d5c; border: 1px solid #555; }
    QComboBox { background-color: #2d2d44; color: #eee; border: 1px solid #444; }
    QCheckBox { color: #eee; }
    """)

    window = TunnelSpallingMaskTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
