#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
土建管片破损掉块数据集划分（修正版）
- 有掩膜的病害图片 -> 病害类 (label=1)
- 正常图片（土建_正常）-> 正常类 (label=0)
- 无掩膜的病害图片：直接排除，不再当作"正常"样本
  （原版把它们打成全 0 掩膜/正常标签，会让分割和分类模型学到"掉块=背景/正常"的错误标签）
"""
import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path


DATA_ROOT = Path(__file__).parent.parent.parent / "data"
SRC_IMAGES = DATA_ROOT / "土建_管片破损掉块"
SRC_MASKS = DATA_ROOT / "土建_管片破损掉块_掩膜"
SRC_NORMAL = DATA_ROOT / "土建_正常"

OUTPUT_ROOT = DATA_ROOT / "管片破损掉块_dataset"
SPLITS = ["train", "val", "test"]

TEST_FIXED = {"0501", "0502", "0503", "0504", "0505",
              "0506", "0507", "0508", "0509", "0510"}

random.seed(42)


def find_samples():
    """收集样本：有掩膜=病害；正常图片来自 土建_正常"""
    images = {}
    for f in os.listdir(SRC_IMAGES):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = f.rsplit('.', 1)[0]
            images[name] = f

    masks = {}
    for f in os.listdir(SRC_MASKS):
        if f.endswith('_mask.png'):
            name = f.replace('_mask.png', '')
            masks[name] = f

    # 只保留有掩膜的病害样本（无掩膜的病害图直接排除）
    disease_samples = []
    skipped = 0
    for name, img_file in images.items():
        if name in masks:
            disease_samples.append((img_file, masks[name], name))
        else:
            skipped += 1

    normal_images = [f for f in os.listdir(SRC_NORMAL)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"[数据清洗] 有掩膜病害: {len(disease_samples)} 张")
    print(f"[数据清洗] 无掩膜病害图(已排除): {skipped} 张")
    print(f"[数据清洗] 正常图片: {len(normal_images)} 张")

    return disease_samples, normal_images


def create_dirs():
    """清空并重建输出目录结构（防止旧一轮残留的污染样本混入新数据集）"""
    for split in SPLITS:
        for sub in ("images", "masks"):
            d = OUTPUT_ROOT / split / sub
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
            d.mkdir(parents=True, exist_ok=True)
    print(f"清空并重建目录: {OUTPUT_ROOT}")


def split_dataset():
    """划分数据集"""
    disease_samples, normal_images = find_samples()

    test_samples = []
    train_disease = []

    for img, mask, name in disease_samples:
        if name in TEST_FIXED:
            test_samples.append((img, mask, name))
        else:
            train_disease.append((img, mask, name))

    random.shuffle(train_disease)
    random.shuffle(normal_images)

    val_disease_size = max(1, int(len(train_disease) * 0.15))
    val_disease = train_disease[:val_disease_size]
    train_disease = train_disease[val_disease_size:]

    train_normal = normal_images[:140]
    val_normal = normal_images[140:175]
    test_normal = normal_images[175:]

    print(f"\n训练集: {len(train_disease)} 病害 + {len(train_normal)} 正常")
    print(f"验证集: {len(val_disease)} 病害 + {len(val_normal)} 正常")
    print(f"测试集: {len(test_samples)} 病害 + {len(test_normal)} 正常")

    copy_disease_split(train_disease, "train")
    copy_disease_split(val_disease, "val")
    copy_disease_split(test_samples, "test")

    copy_normal_split(train_normal, "train")
    copy_normal_split(val_normal, "val")
    copy_normal_split(test_normal, "test")


def copy_disease_split(samples, split_name):
    """复制病害样本（保持原始文件名）"""
    for img_name, mask_name, base_name in samples:
        src_img = SRC_IMAGES / img_name
        dst_img = OUTPUT_ROOT / split_name / "images" / img_name
        shutil.copy2(src_img, dst_img)

        src_mask = SRC_MASKS / mask_name
        dst_mask = OUTPUT_ROOT / split_name / "masks" / mask_name
        shutil.copy2(src_mask, dst_mask)


def copy_normal_split(samples, split_name):
    """复制正常样本，生成全 0 掩膜；文件名加 n_ 前缀，避免与病害图重名覆盖"""
    for img_name in samples:
        # 加前缀：病害图与正常图来自不同源目录，文件名可能同名（如 0_1_004_10.jpg），
        # 若都复制进同一个 images/ 目录，后复制的会覆盖先复制的。
        new_img_name = "n_" + img_name
        src_img = SRC_NORMAL / img_name
        dst_img = OUTPUT_ROOT / split_name / "images" / new_img_name
        shutil.copy2(src_img, dst_img)

        base = img_name.rsplit('.', 1)[0]
        mask_name = f"n_{base}_mask.png"
        dst_mask = OUTPUT_ROOT / split_name / "masks" / mask_name

        img = cv2.imread(str(src_img))
        if img is not None:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(str(dst_mask), mask)


def generate_yaml():
    """生成dataset.yaml"""
    yaml_content = f"""# 管片破损掉块分割数据集
path: {OUTPUT_ROOT.absolute()}
train: train/images
val: val/images
test: test/images

mask_dir: masks

names:
  0: background
  1: guanpianposun

nc: 2
"""
    yaml_path = OUTPUT_ROOT / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"\n生成配置文件: {yaml_path}")


def print_summary():
    """打印统计"""
    print("\n" + "=" * 50)
    print("数据集划分完成!")
    print("=" * 50)
    for split in SPLITS:
        n = len(list((OUTPUT_ROOT / split / "images").glob("*")))
        print(f"{split:6s}: {n:4d} images")
    print("=" * 50)
    print(f"输出目录: {OUTPUT_ROOT}")


if __name__ == "__main__":
    create_dirs()
    split_dataset()
    generate_yaml()
    print_summary()
