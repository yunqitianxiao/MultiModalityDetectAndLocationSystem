#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像相似度筛选 - 从defects中筛选Spalling相关图片
使用图像特征匹配，不需要训练模型
"""
import os
import shutil
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SPALLING_DIR = os.path.join(DATA_DIR, "土建_管片破损掉块")
DEFECTS_DIR = os.path.join(DATA_DIR, "defects")
OUTPUT_DIR = os.path.join(DATA_DIR, "defects_筛选结果")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_features(image_path):
    """提取图像特征（颜色直方图 + ORB特征）"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 颜色直方图特征
    hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])
    color_hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()
    color_hist = color_hist / (color_hist.sum() + 1e-6)

    # 灰度特征
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    texture_features = resized.flatten() / 255.0

    features = np.concatenate([color_hist, texture_features])
    return features

def compute_similarity(f1, f2):
    """计算余弦相似度"""
    dot = np.dot(f1, f2)
    norm1 = np.linalg.norm(f1)
    norm2 = np.linalg.norm(f2)
    return dot / (norm1 * norm2 + 1e-6)

def main():
    print("=" * 60)
    print("图像相似度筛选 - 从defects筛选Spalling图片")
    print("=" * 60)

    # 获取Spalling图片列表
    spalling_files = [f for f in os.listdir(SPALLING_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Spalling样本: {len(spalling_files)} 张")

    # 获取defects图片列表
    defects_files = [f for f in os.listdir(DEFECTS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Defects待筛选: {len(defects_files)} 张")

    # 提取Spalling特征
    print("\n提取Spalling特征...")
    spalling_features = []
    for i, filename in enumerate(spalling_files):
        filepath = os.path.join(SPALLING_DIR, filename)
        feat = extract_features(filepath)
        if feat is not None:
            spalling_features.append(feat)
        if (i + 1) % 30 == 0:
            print(f"  已处理 {i + 1}/{len(spalling_files)}")

    spalling_features = np.array(spalling_features)
    spalling_mean = spalling_features.mean(axis=0)
    print(f"提取了 {len(spalling_features)} 个特征")

    # 扫描defects
    print("\n扫描defects目录...")
    results = []

    for i, filename in enumerate(defects_files):
        filepath = os.path.join(DEFECTS_DIR, filename)
        feat = extract_features(filepath)

        if feat is not None:
            # 计算与Spalling平均特征的相似度
            sim = compute_similarity(feat, spalling_mean)
            results.append((filename, sim))

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(defects_files)}")

    # 按相似度排序
    results.sort(key=lambda x: -x[1])

    # 输出结果
    print("\n" + "=" * 60)
    print("筛选结果（按相似度排序）")
    print("=" * 60)

    high_sim = [r for r in results if r[1] > 0.85]
    mid_sim = [r for r in results if 0.75 < r[1] <= 0.85]
    low_sim = [r for r in results if r[1] <= 0.75]

    print(f"\n高相似度 (>0.85): {len(high_sim)} 张")
    print(f"中相似度 (0.75-0.85): {len(mid_sim)} 张")
    print(f"低相似度 (<0.75): {len(low_sim)} 张")

    # 保存结果
    result_file = os.path.join(OUTPUT_DIR, "defects_相似度筛选结果.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Defects Spalling相似度筛选结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"参考样本: {len(spalling_files)} 张Spalling图片\n\n")

        f.write("【高相似度样本】\n")
        for filename, sim in high_sim:
            f.write(f"  {sim:.4f}  {filename}\n")

        f.write("\n【中相似度样本】\n")
        for filename, sim in mid_sim:
            f.write(f"  {sim:.4f}  {filename}\n")

        f.write("\n【低相似度样本（前50个）】\n")
        for filename, sim in low_sim[:50]:
            f.write(f"  {sim:.4f}  {filename}\n")

    print(f"\n结果已保存到: {result_file}")

    # 复制高相似度图片
    high_sim_dir = os.path.join(OUTPUT_DIR, "高相似度Spalling")
    os.makedirs(high_sim_dir, exist_ok=True)

    for filename, sim in high_sim[:100]:
        src = os.path.join(DEFECTS_DIR, filename)
        dst = os.path.join(high_sim_dir, f"{sim:.4f}_{filename}")
        shutil.copy2(src, dst)

    print(f"高相似度图片已复制到: {high_sim_dir}/")

    # 显示前30个结果
    print("\n前30个最可能是Spalling的图片:")
    for i, (filename, sim) in enumerate(results[:30]):
        print(f"  {i+1}. {sim:.4f}  {filename}")

if __name__ == "__main__":
    main()
