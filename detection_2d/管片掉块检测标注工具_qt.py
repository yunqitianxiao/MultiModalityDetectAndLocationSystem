#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
管片掉块检测标注工具（PyQt5 桌面版）

针对 data/土建_管片掉块 中 人工标注.txt 所列的正样本图片，
用鼠标框选"管片掉块(spalling)"目标，输出 YOLO 检测格式标注
（class cx cy w h，归一化坐标，单类 spalling=0）。

用法：
    python3 管片掉块检测标注工具_qt.py            # 启动桌面标注界面
    python3 管片掉块检测标注工具_qt.py --pack     # 直接把已有 labels 打包为 YOLO 数据集

数据流：
    图片   data/土建_管片掉块/<name>.jpg
    清单   data/土建_管片掉块_人工标注.txt （每行一个图名，无扩展名）
    标注   data/土建_管片掉块/labels/<name>.txt   （YOLO 格式，空文件=无病害负样本）
    打包   data/土建_管片掉块_yolo_手工/{images,labels,data.yaml}

交互：
    左键拖拽画框 · 拖动框移动 · 拉边/角缩放 · 中键拖拽平移 · 滚轮缩放 · 双击适配窗口
    Space/→ 下一张 · ←/Q 上一张 · S 保存 · Delete 删选中框 · Ctrl+Z 撤销 · Esc 取消
"""
import os
import re
import shutil
import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

# ---------------------------------------------------------------- 路径约定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
IMG_DIR = os.path.join(DATA_DIR, "土建_管片掉块")
LIST_FILE = os.path.join(DATA_DIR, "土建_管片掉块_人工标注.txt")
LABEL_DIR = os.path.join(IMG_DIR, "labels")
PACK_DIR = os.path.join(DATA_DIR, "土建_管片掉块_yolo_手工")

CLASSES = ["spalling"]
YOLO_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*$"
)


# ---------------------------------------------------------------- 数据层
def image_list():
    result = []
    if not os.path.exists(LIST_FILE):
        return result
    with open(LIST_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            base = raw.strip()
            if not base:
                continue
            name = base + ".jpg"
            if not os.path.exists(os.path.join(IMG_DIR, name)):
                continue
            lp = os.path.join(LABEL_DIR, name.replace(".jpg", ".txt"))
            box_count = 0
            if os.path.exists(lp):
                with open(lp, "r", encoding="utf-8") as lf:
                    box_count = len([ln for ln in lf.read().splitlines() if ln.strip()])
            result.append({"name": name, "done": os.path.exists(lp),
                           "annotated": box_count > 0, "boxCount": box_count})
    return result


def read_label(name):
    safe = os.path.basename(name)
    p = os.path.join(LABEL_DIR, safe.replace(".jpg", ".txt"))
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def save_label(name, text):
    safe = os.path.basename(name)
    os.makedirs(LABEL_DIR, exist_ok=True)
    p = os.path.join(LABEL_DIR, safe.replace(".jpg", ".txt"))
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def delete_label(name):
    safe = os.path.basename(name)
    p = os.path.join(LABEL_DIR, safe.replace(".jpg", ".txt"))
    if os.path.exists(p):
        os.remove(p)


def pack_yolo():
    if os.path.exists(PACK_DIR):
        shutil.rmtree(PACK_DIR)
    img_out = os.path.join(PACK_DIR, "images")
    lbl_out = os.path.join(PACK_DIR, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    total = with_lb = without_lb = 0
    for item in image_list():
        name = item["name"]
        lp = os.path.join(LABEL_DIR, name.replace(".jpg", ".txt"))
        if not os.path.exists(lp):
            continue
        shutil.copy2(os.path.join(IMG_DIR, name), os.path.join(img_out, name))
        shutil.copy2(lp, os.path.join(lbl_out, name.replace(".jpg", ".txt")))
        total += 1
        if item["boxCount"] > 0:
            with_lb += 1
        else:
            without_lb += 1

    yaml_text = (
        "# 由 管片掉块检测标注工具_qt.py 手工框标注导出(YOLO 检测框)\n"
        f"path: {PACK_DIR}\n"
        "train: images\nval: images\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    with open(os.path.join(PACK_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_text)
    return {"total": total, "withLabels": with_lb, "withoutLabels": without_lb}


# ---------------------------------------------------------------- Canvas
def clamp(v, a, b):
    return max(a, min(b, v))


class Canvas(QWidget):
    """图片显示 + 框标注画布。坐标一律为图像像素坐标，显示时按 zoom/offset 变换。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(600, 400)

        self._img = None          # QImage 原图
        self._w = 0
        self._h = 0
        self.boxes = []           # [{x,y,w,h}] 图像像素坐标
        self.sel = -1
        self._view = QPointF(0, 0)   # 图像原点在 widget 中的位置
        self._zoom = 1.0
        self.dirty = False
        self._undo = []
        self._drag = None
        self._pan = None

    # -- 视图
    def fit_in_view(self):
        if not self._img or self.width() < 10 or self.height() < 10:
            return
        z = min(self.width() / self._w, self.height() / self._h) * 0.97
        self._zoom = max(z, 0.05)
        self._view = QPointF((self.width() - self._w * self._zoom) / 2,
                             (self.height() - self._h * self._zoom) / 2)

    def set_image(self, qimg, boxes):
        self._img = qimg
        self._w = qimg.width()
        self._h = qimg.height()
        self.boxes = boxes
        self.sel = -1
        self._undo = []
        self.dirty = False
        self.fit_in_view()
        self.update()

    def _to_img(self, pos):
        return QPointF((pos.x() - self._view.x()) / self._zoom,
                       (pos.y() - self._view.y()) / self._zoom)

    # -- 绘制
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(37, 42, 49))
        if not self._img:
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setRenderHint(QPainter.Antialiasing)
        x0, y0 = self._view.x(), self._view.y()
        w, h = self._w * self._zoom, self._h * self._zoom
        p.drawImage(QRectF(x0, y0, w, h), self._img)
        p.setPen(QPen(QColor(90, 100, 110), 1))
        p.drawRect(QRectF(x0, y0, w, h))

        for i, b in enumerate(self.boxes):
            self._draw_box(p, b, i == self.sel)

        d = self._drag
        if d and d["mode"] == "draw":
            x = min(d["x0"], d["x1"]); y = min(d["y0"], d["y1"])
            ww = abs(d["x1"] - d["x0"]); hh = abs(d["y1"] - d["y0"])
            pen = QPen(QColor(37, 99, 235), max(1.0, 2.0 / self._zoom))
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawRect(QRectF(x * self._zoom + x0, y * self._zoom + y0,
                              ww * self._zoom, hh * self._zoom))

    def _draw_box(self, p, b, sel):
        x0, y0 = self._view.x(), self._view.y()
        z = self._zoom
        r = QRectF(b["x"] * z + x0, b["y"] * z + y0, b["w"] * z, b["h"] * z)
        color = QColor(220, 38, 38) if sel else QColor(34, 197, 94)
        p.setPen(QPen(color, max(1.5, 2.5 / z)))
        p.setBrush(QColor(220, 38, 38, 40) if sel else QColor(34, 197, 94, 30))
        p.drawRect(r)

        # 类标签
        f = QFont()
        f.setPointSizeF(max(6.0, 13.0 / z))
        p.setFont(f)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(CLASSES[0]) + 10.0 / z
        th = fm.height()
        ty = max(r.top() - th, 0)
        p.setBrush(color)
        p.setPen(Qt.NoPen)
        p.drawRect(QRectF(r.left(), ty, tw, th))
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(r.left(), ty, tw, th), Qt.AlignCenter, CLASSES[0])

        if sel:
            s = 6.0 / z
            p.setBrush(color)
            p.setPen(Qt.NoPen)
            for hx, hy in [(b["x"], b["y"]), (b["x"] + b["w"], b["y"]),
                           (b["x"], b["y"] + b["h"]), (b["x"] + b["w"], b["y"] + b["h"])]:
                p.drawRect(QRectF(hx * z + x0 - s / 2, hy * z + y0 - s / 2, s, s))

    # -- 命中测试
    def _hit_test(self, pt):
        z = self._zoom
        t = 6.0 / z
        for i in range(len(self.boxes) - 1, -1, -1):
            b = self.boxes[i]
            if not (b["x"] - t <= pt.x() <= b["x"] + b["w"] + t and
                    b["y"] - t <= pt.y() <= b["y"] + b["h"] + t):
                continue
            onL = abs(pt.x() - b["x"]) <= t
            onR = abs(pt.x() - (b["x"] + b["w"])) <= t
            onT = abs(pt.y() - b["y"]) <= t
            onB = abs(pt.y() - (b["y"] + b["h"])) <= t
            if onT and onL: mode = "nw"
            elif onT and onR: mode = "ne"
            elif onB and onL: mode = "sw"
            elif onB and onR: mode = "se"
            elif onT: mode = "n"
            elif onB: mode = "s"
            elif onL: mode = "w"
            elif onR: mode = "e"
            else: mode = "move"
            return {"i": i, "mode": mode}
        return None

    # -- 撤销
    def _push_undo(self):
        self._undo.append([dict(b) for b in self.boxes])
        if len(self._undo) > 80:
            self._undo.pop(0)

    def undo(self):
        if not self._undo:
            return
        self.boxes = self._undo.pop()
        self.sel = -1
        self.dirty = True
        self.update()

    # -- 鼠标
    def mousePressEvent(self, e):
        if not self._img:
            return
        if e.button() == Qt.MiddleButton:
            self._pan = (QPointF(e.pos()), QPointF(self._view))
            return
        if e.button() != Qt.LeftButton:
            return
        pt = self._to_img(QPointF(e.pos()))
        self._push_undo()
        ht = self._hit_test(pt)
        if ht:
            self.sel = ht["i"]
            b = self.boxes[ht["i"]]
            self._drag = {"i": ht["i"], "mode": ht["mode"], "ix": pt.x(), "iy": pt.y(),
                          "bx": b["x"], "by": b["y"], "bw": b["w"], "bh": b["h"]}
        else:
            self.sel = -1
            self._drag = {"mode": "draw", "x0": pt.x(), "y0": pt.y(),
                          "x1": pt.x(), "y1": pt.y()}
        self.update()

    def mouseMoveEvent(self, e):
        if self._pan:
            start, v0 = self._pan
            self._view = QPointF(v0.x() + QPointF(e.pos()).x() - start.x(),
                                 v0.y() + QPointF(e.pos()).y() - start.y())
            self.update()
            return
        if not self._drag or not self._img:
            return
        pt = self._to_img(QPointF(e.pos()))
        d = self._drag
        if d["mode"] == "draw":
            d["x1"] = clamp(pt.x(), 0, self._w)
            d["y1"] = clamp(pt.y(), 0, self._h)
            self.dirty = True
        else:
            b = self.boxes[d["i"]]
            dx = pt.x() - d["ix"]
            dy = pt.y() - d["iy"]
            L, T, R, B = d["bx"], d["by"], d["bx"] + d["bw"], d["by"] + d["bh"]
            nl, nt, nr, nb = L, T, R, B
            mode = d["mode"]
            if mode == "move":
                nl, nt, nr, nb = L + dx, T + dy, R + dx, B + dy
            else:
                if "w" in mode: nl = L + dx
                elif "e" in mode: nr = R + dx
                if "n" in mode: nt = T + dy
                elif "s" in mode: nb = B + dy
                if nr - nl < 4:
                    nl = nr - 4 if "w" in mode else nl
                    nr = nl + 4 if "e" in mode else nr
                if nb - nt < 4:
                    nt = nb - 4 if "n" in mode else nt
                    nb = nt + 4 if "s" in mode else nb
                nl = clamp(nl, 0, self._w); nr = clamp(nr, 0, self._w)
                nt = clamp(nt, 0, self._h); nb = clamp(nb, 0, self._h)
            if mode == "move":
                ww, hh = nr - nl, nb - nt
                nl = clamp(nl, 0, self._w - ww)
                nt = clamp(nt, 0, self._h - hh)
                nr, nb = nl + ww, nt + hh
            b["x"], b["y"], b["w"], b["h"] = nl, nt, nr - nl, nb - nt
            self.dirty = True
        self.update()

    def mouseReleaseEvent(self, e):
        d = self._drag
        if d and d["mode"] == "draw" and e.button() == Qt.LeftButton:
            w = abs(d["x1"] - d["x0"])
            h = abs(d["y1"] - d["y0"])
            if w >= 3 and h >= 3:          # 太小视为误点，丢弃
                self.boxes.append({
                    "x": min(d["x0"], d["x1"]),
                    "y": min(d["y0"], d["y1"]),
                    "w": w,
                    "h": h,
                })
                self.dirty = True          # 按下时已存「画框前」快照，Ctrl+Z 可直接撤销
        self._drag = None
        self._pan = None
        self.update()

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.fit_in_view()
            self.update()

    def wheelEvent(self, e):
        if not self._img:
            return
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        pos = QPointF(e.pos())
        ip = self._to_img(pos)
        self._zoom = clamp(self._zoom * factor, 0.04, 60)
        self._view = QPointF(pos.x() - ip.x() * self._zoom,
                             pos.y() - ip.y() * self._zoom)
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._img:
            self.fit_in_view()
            self.update()


# ---------------------------------------------------------------- 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("管片掉块检测标注工具（spalling）")
        self.resize(1280, 800)

        self.images = image_list()
        self.idx = -1

        # 左侧面板
        self.listw = QListWidget()
        self.listw.currentItemChanged.connect(self._on_item_changed)

        self.btn_no = QPushButton("无病害")
        self.btn_skip = QPushButton("跳过")
        self.btn_del = QPushButton("删除标注")
        self.btn_pack = QPushButton("导出YOLO")
        self.btn_no.setToolTip("当前图无病害：写空标签并下一张")
        self.btn_skip.setToolTip("跳过：不保存，下一张")
        self.btn_del.setToolTip("删除当前图全部标注")
        self.btn_pack.setToolTip("把已标注图片打包为 YOLO 数据集")

        self.lbl_progress = QLabel("")
        self.lbl_progress.setStyleSheet("font-weight:bold;")
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)

        left = QVBoxLayout()
        btn_row = QHBoxLayout()
        for b in (self.btn_no, self.btn_skip, self.btn_del, self.btn_pack):
            btn_row.addWidget(b)
        left.addLayout(btn_row)
        left.addWidget(self.listw, 1)
        left.addWidget(self.lbl_progress)
        left.addWidget(self.lbl_status)

        panel = QWidget()
        panel.setLayout(left)
        panel.setMaximumWidth(320)
        panel.setMinimumWidth(260)

        # 右侧画布
        self.canvas = Canvas()
        hint = QLabel("左键拖拽画框 · 拖动框移动 · 拉边/角缩放 · 中键平移 · 滚轮缩放 · 双击适配\n"
                      "Space/→ 下一张 · ←/Q 上一张 · S 保存 · Delete 删选中框 · Ctrl+Z 撤销")
        hint.setStyleSheet("color:#6b7280; padding:4px 8px;")

        right = QVBoxLayout()
        right.addWidget(self.canvas, 1)
        right.addWidget(hint)

        root = QWidget()
        lay = QHBoxLayout()
        lay.addWidget(panel)
        lay.addLayout(right, 1)
        root.setLayout(lay)
        self.setCentralWidget(root)

        # 信号
        self.btn_no.clicked.connect(self.mark_no_defect)
        self.btn_skip.clicked.connect(self.skip)
        self.btn_del.clicked.connect(self.delete_annotation)
        self.btn_pack.clicked.connect(self.export_yolo)

        self._populate_list()
        if self.images:
            self.listw.setCurrentRow(0)

    # -- 列表
    def _populate_list(self):
        self.listw.blockSignals(True)
        for it in self.images:
            self.listw.addItem(QListWidgetItem(self._item_text(it)))
        self.listw.blockSignals(False)
        self._update_progress()

    def _item_text(self, it):
        if not it["done"]:
            return f"○ {it['name']}  (未标)"
        if it["annotated"]:
            return f"● {it['name']}  ({it['boxCount']}框)"
        return f"● {it['name']}  (无病害)"

    def _refresh_item(self, i):
        self.listw.blockSignals(True)
        self.listw.item(i).setText(self._item_text(self.images[i]))
        self.listw.blockSignals(False)
        self._update_progress()

    def _update_progress(self):
        done = sum(1 for it in self.images if it["done"])
        self.lbl_progress.setText(f"已标 {done} / {len(self.images)}")

    def _on_item_changed(self, cur, prev):
        if cur is None:
            return
        row = self.listw.row(cur)
        if row != self.idx:
            self._goto(row)

    # -- 图片加载/切换
    def _goto(self, row):
        if row == self.idx:
            return
        if self.canvas.dirty:
            self.save(notify=False)
        self.idx = row
        self._load_image()

    def _load_image(self):
        it = self.images[self.idx]
        path = os.path.join(IMG_DIR, it["name"])
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            self.lbl_status.setText(f"无法读取图片: {it['name']}")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        boxes = self._parse_yolo(read_label(it["name"]), w, h)
        self.canvas.set_image(qimg, boxes)
        self._update_status()

    def _parse_yolo(self, text, w, h):
        out = []
        for ln in text.splitlines():
            m = YOLO_LINE_RE.match(ln)
            if not m:
                continue
            cx, cy, bw, bh = (float(m.group(i)) for i in range(2, 6))
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            out.append({"x": x, "y": y, "w": bw * w, "h": bh * h})
        return out

    def _boxes_to_yolo(self, w, h):
        lines = []
        for b in self.canvas.boxes:
            cx, cy = (b["x"] + b["w"] / 2) / w, (b["y"] + b["h"] / 2) / h
            bw, bh = b["w"] / w, b["h"] / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return "\n".join(lines)

    def _update_status(self):
        it = self.images[self.idx]
        self.lbl_status.setText(
            f"{self.idx + 1}/{len(self.images)}  {it['name']}\n"
            f"{len(self.canvas.boxes)} 个框  {'· 有未保存修改' if self.canvas.dirty else ''}"
        )

    # -- 动作
    def save(self, notify=True):
        if self.idx < 0 or not self.canvas._img:
            return
        w, h = self.canvas._w, self.canvas._h
        text = self._boxes_to_yolo(w, h)
        save_label(self.images[self.idx]["name"], text)
        it = self.images[self.idx]
        it["done"] = True
        it["annotated"] = bool(text)
        it["boxCount"] = len(self.canvas.boxes)
        self.canvas.dirty = False
        self._refresh_item(self.idx)
        self._update_status()
        if notify:
            self.statusBar().showMessage(f"已保存 {it['name']}", 2000)

    def mark_no_defect(self):
        if self.idx < 0:
            return
        self.canvas.boxes = []
        self.canvas._push_undo()
        self.save(notify=True)
        self._next()

    def skip(self):
        self.canvas.dirty = False
        self._next()

    def delete_annotation(self):
        if self.idx < 0:
            return
        delete_label(self.images[self.idx]["name"])
        self.canvas.boxes = []
        self.canvas.dirty = False
        it = self.images[self.idx]
        it["done"] = it["annotated"] = False
        it["boxCount"] = 0
        self._refresh_item(self.idx)
        self.canvas.update()
        self._update_status()
        self.statusBar().showMessage("已删除标注", 2000)

    def export_yolo(self):
        result = pack_yolo()
        QMessageBox.information(
            self, "导出完成",
            f"共 {result['total']} 张（含病害 {result['withLabels']}，负样本 {result['withoutLabels']}）\n"
            f"输出目录：\n{PACK_DIR}",
        )

    def _next(self):
        if self.idx + 1 < len(self.images):
            self.listw.setCurrentRow(self.idx + 1)
        else:
            self.statusBar().showMessage("已是最后一张", 2000)

    def _prev(self):
        if self.idx - 1 >= 0:
            self.listw.setCurrentRow(self.idx - 1)
        else:
            self.statusBar().showMessage("已是第一张", 2000)

    def keyPressEvent(self, e):
        key = e.key()
        mod = e.modifiers()
        if key == Qt.Key_Space or key == Qt.Key_Right:
            self._next()
        elif key == Qt.Key_Left or key == Qt.Key_Q:
            self._prev()
        elif key == Qt.Key_S and not (mod & Qt.ControlModifier):
            self.save()
        elif key == Qt.Key_Delete:
            if 0 <= self.canvas.sel < len(self.canvas.boxes):
                self.canvas._push_undo()
                del self.canvas.boxes[self.canvas.sel]
                self.canvas.sel = -1
                self.canvas.dirty = True
                self.canvas.update()
                self._update_status()
        elif key == Qt.Key_Escape:
            self.canvas._drag = None
        elif key == Qt.Key_Z and (mod & Qt.ControlModifier):
            self.canvas.undo()
            self._update_status()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        if self.canvas.dirty:
            r = QMessageBox.question(self, "未保存", "当前图有未保存修改，保存后退出？",
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if r == QMessageBox.Save:
                self.save(notify=False)
                e.accept()
            elif r == QMessageBox.Discard:
                e.accept()
            else:
                e.ignore()
        else:
            e.accept()


# ---------------------------------------------------------------- 入口
def main():
    if "--pack" in sys.argv:
        print(pack_yolo())
        return

    if not os.path.exists(IMG_DIR):
        print("图片目录不存在: " + IMG_DIR)
        sys.exit(1)
    if not os.path.exists(LIST_FILE):
        print("清单文件不存在: " + LIST_FILE)
        sys.exit(1)

    # opencv-python 导入时会把 QT_QPA_PLATFORM_PLUGIN_PATH 指向它自带的
    # Qt 插件目录（与 PyQt5 的 Qt ABI 不匹配，导致 xcb 无法加载）；
    # 这里强制改回 PyQt5 自带的 Qt5 插件目录（保证与 venv 内 PyQt5 ABI 匹配）。
    import PyQt5
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
        os.path.dirname(PyQt5.__file__), "Qt5", "plugins", "platforms"
    )

    app = QApplication(sys.argv)
    app.setApplicationName("管片掉块检测标注工具")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
