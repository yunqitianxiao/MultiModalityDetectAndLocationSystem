#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主动学习筛选工具 - 从defects中筛选Spalling图片
使用已训练的分类模型对defects进行预测，筛选高概率样本
"""
import os
import shutil
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import PeftModel
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

MODEL_NAME = "nvidia/mit-b5"
CLS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "guanpian_classifier_results")

def load_or_train_classifier():
    """加载或训练分类器"""
    cls_model_dir = os.path.join(CLS_OUTPUT_DIR, "classification_model")
    lora_adapter_dir = os.path.join(cls_model_dir, "lora_adapter")
    processor_dir = os.path.join(cls_model_dir, "processor")

    if os.path.exists(lora_adapter_dir) and os.path.exists(processor_dir):
        print("加载已有的分类模型...")
        processor = AutoImageProcessor.from_pretrained(processor_dir)
        base_model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        return processor, model
    else:
        print("未找到训练好的模型，需要先训练...")
        print("请先运行: python 土建/管片破损掉块/Lora微调_分类模型.py")
        return None, None

def predict_image(model, processor, image_path, image_size=512):
    """预测单张图片"""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(model.device if hasattr(model, 'device') else "cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    return probs[1]  # 返回是Spalling的概率

def scan_defects_for_spalling(model, processor, top_k=50):
    """扫描defects目录，筛选可能的Spalling图片"""
    defects_files = [f for f in os.listdir(DEFECTS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    results = []

    print(f"正在扫描 {len(defects_files)} 张defects图片...")

    for i, filename in enumerate(defects_files):
        filepath = os.path.join(DEFECTS_DIR, filename)
        try:
            prob = predict_image(model, processor, filepath)
            results.append((filename, prob))

            if (i + 1) % 20 == 0:
                print(f"  已处理 {i + 1}/{len(defects_files)}")
        except Exception as e:
            print(f"  跳过 {filename}: {e}")

        if (i + 1) % 50 == 0:
            print(f"进度: {i + 1}/{len(defects_files)}")

    # 按概率排序
    results.sort(key=lambda x: -x[1])

    return results

def main():
    print("=" * 60)
    print("主动学习 - 从defects筛选Spalling图片")
    print("=" * 60)

    # 加载模型
    processor, model = load_or_train_classifier()

    if model is None:
        print("\n请先训练分类模型:")
        print("  python 土建/管片破损掉块/Lora微调_分类模型.py")
        return

    # 扫描defects
    results = scan_defects_for_spalling(model, processor)

    # 输出结果
    print("\n" + "=" * 60)
    print("预测结果（按Spalling概率排序）")
    print("=" * 60)

    high_prob = [r for r in results if r[1] > 0.7]
    mid_prob = [r for r in results if 0.5 < r[1] <= 0.7]
    low_prob = [r for r in results if r[1] <= 0.5]

    print(f"\n高概率 (>0.7): {len(high_prob)} 张")
    print(f"中概率 (0.5-0.7): {len(mid_prob)} 张")
    print(f"低概率 (<0.5): {len(low_prob)} 张")

    # 保存结果
    result_file = os.path.join(OUTPUT_DIR, "defects_筛选结果.txt")
    with open(result_file, "w") as f:
        f.write("Defects Spalling筛选结果\n")
        f.write("=" * 60 + "\n\n")
        f.write("【高概率样本】\n")
        for filename, prob in high_prob:
            f.write(f"  {prob:.4f}  {filename}\n")

        f.write("\n【中概率样本】\n")
        for filename, prob in mid_prob:
            f.write(f"  {prob:.4f}  {filename}\n")

        f.write("\n【低概率样本】\n")
        for filename, prob in low_prob[:100]:
            f.write(f"  {prob:.4f}  {filename}\n")

    print(f"\n结果已保存到: {result_file}")

    # 复制高概率图片到输出目录
    high_prob_dir = os.path.join(OUTPUT_DIR, "高概率Spalling")
    os.makedirs(high_prob_dir, exist_ok=True)

    for filename, prob in high_prob:
        src = os.path.join(DEFECTS_DIR, filename)
        dst = os.path.join(high_prob_dir, f"{prob:.4f}_{filename}")
        shutil.copy2(src, dst)

    print(f"高概率图片已复制到: {high_prob_dir}/")

    # 显示前20个结果
    print("\n前20个最可能是Spalling的图片:")
    for i, (filename, prob) in enumerate(results[:20]):
        print(f"  {i+1}. {prob:.4f}  {filename}")

if __name__ == "__main__":
    main()
