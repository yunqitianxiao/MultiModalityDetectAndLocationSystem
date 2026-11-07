#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VOC(Annotations XML + JPEGImages) → YOLO 检测标签
================================================
输入: data/Annotations/*.xml + data/JPEGImages/*.jpg (330 张, 3类人工 bbox 标注)
      Crack 129 框 / Leakage 225 框 / Spalling 176 框 (一张图可含多个框/多类)
输出: data/土建_VOC三病害_yolo/{images,labels}/{train,val,test} + data.yaml
      每个对象 → 一行归一化 (class cx cy w h)

类编号对齐项目统一 8 类 (病害训练集+正常状态/class.txt):
  5=guanpianposun 6=liefeng 7=shenloushui
  这样以后合并成统一 8 类数据集时编号无需重排。

划分: 随机 seed=42, 约 10% 进 test, 其余 85/15 分 train/val。
用法:  python3 VOC转YOLO框.py
"""
import os
import random
import shutil
import xml.etree.ElementTree as ET

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ANN_DIR = os.path.join(DATA, "Annotations")
IMG_DIR = os.path.join(DATA, "JPEGImages")
OUT = os.path.join(DATA, "土建_VOC三病害_yolo")

CLS_MAP = {"Spalling": 5, "Crack": 6, "Leakage": 7}
NAME_MAP = {5: "guanpianposun", 6: "liefeng", 7: "shenloushui"}
IMG_EXTS = (".jpg", ".jpeg", ".png")

random.seed(42)


def parse_xml(path):
    root = ET.parse(path).getroot()
    base = os.path.splitext(root.find("filename").text)[0]
    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)
    boxes = []
    for o in root.iter("object"):
        name = o.find("name").text
        if name not in CLS_MAP:
            continue
        bb = o.find("bndbox")
        x1 = int(float(bb.find("xmin").text))
        y1 = int(float(bb.find("ymin").text))
        x2 = int(float(bb.find("xmax").text))
        y2 = int(float(bb.find("ymax").text))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((CLS_MAP[name], x1, y1, x2, y2))
    return base, W, H, boxes


def main():
    samples = []
    for f in sorted(os.listdir(ANN_DIR)):
        if not f.endswith(".xml"):
            continue
        base, W, H, boxes = parse_xml(os.path.join(ANN_DIR, f))
        img = next((os.path.join(IMG_DIR, base + e) for e in IMG_EXTS
                    if os.path.exists(os.path.join(IMG_DIR, base + e))), None)
        if img is None or not boxes:
            continue
        samples.append((base, W, H, boxes, img))

    random.shuffle(samples)
    n_test = int(len(samples) * 0.10)
    test = samples[:n_test]
    rest = samples[n_test:]
    n_val = int(len(rest) * 0.15)
    val, train = rest[:n_val], rest[n_val:]

    def write(split, items):
        img_dir = os.path.join(OUT, "images", split)
        lbl_dir = os.path.join(OUT, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for base, W, H, boxes, img in items:
            shutil.copy(img, os.path.join(img_dir, base + ".jpg"))
            lines = []
            for cls, x1, y1, x2, y2 in boxes:
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            with open(os.path.join(lbl_dir, base + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + "\n")

    write("train", train)
    write("val", val)
    write("test", test)

    yaml = (
        "# 由 VOC转YOLO框.py 从 Annotations(xml)+JPEGImages 导出(人工 bbox 标注)\n"
        f"path: {OUT}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "\n"
        "nc: 8\n"
        "names:\n"
        "  5: guanpianposun\n"
        "  6: liefeng\n"
        "  7: shenloushui\n"
    )
    with open(os.path.join(OUT, "data.yaml"), "w") as fh:
        fh.write(yaml)

    def count(split):
        return (len(os.listdir(os.path.join(OUT, "images", split))),
                len(os.listdir(os.path.join(OUT, "labels", split))))

    print(f"解析成功 {len(samples)} 张 (含有效框)")
    print(f"  train: images/labels = {count('train')}")
    print(f"  val:   images/labels = {count('val')}")
    print(f"  test:  images/labels = {count('test')}")
    print(f"输出目录: {OUT}")


if __name__ == "__main__":
    main()
