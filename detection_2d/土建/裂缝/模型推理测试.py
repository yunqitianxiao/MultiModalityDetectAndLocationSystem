#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型推理测试 - 分割 + 分类
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, AutoModelForImageClassification
from peft import PeftModel

# ============================================
# 1. 配置
# ============================================
print("=" * 60)
print("🧪 模型推理测试")
print("=" * 60)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

SEG_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "segformer_results")
CLS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "classifier_results")

SEG_MODEL_DIR = os.path.join(SEG_OUTPUT_DIR, "segmentation_model")
CLS_MODEL_DIR = os.path.join(CLS_OUTPUT_DIR, "classification_model")

TEST_IMAGE = os.path.join(PROJECT_ROOT, "data/土建_裂缝/0601.jpg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ============================================
# 2. 加载模型
# ============================================
print("\n加载模型...")

# 分类模型
cls_processor = AutoImageProcessor.from_pretrained(
    os.path.join(CLS_MODEL_DIR, "processor")
)
cls_base = AutoModelForImageClassification.from_pretrained(
    "nvidia/mit-b5",
    num_labels=2,
    ignore_mismatched_sizes=True,
)
cls_model = PeftModel.from_pretrained(cls_base, os.path.join(CLS_MODEL_DIR, "lora_adapter"))
cls_model.to(device)
cls_model.eval()

# 分割模型
seg_processor = AutoImageProcessor.from_pretrained(
    os.path.join(SEG_MODEL_DIR, "processor")
)
seg_base = AutoModelForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b5",
    num_labels=2,
    ignore_mismatched_sizes=True,
)
seg_model = PeftModel.from_pretrained(seg_base, os.path.join(SEG_MODEL_DIR, "lora_adapter"))
seg_model.to(device)
seg_model.eval()

print("✅ 模型加载成功！")

# ============================================
# 3. 推理
# ============================================
if not os.path.exists(TEST_IMAGE):
    print(f"❌ 测试图片不存在: {TEST_IMAGE}")
    print("请修改 TEST_IMAGE 路径")
    sys.exit(1)

print(f"\n测试图片: {TEST_IMAGE}")
image = Image.open(TEST_IMAGE).convert("RGB")
image_np = np.array(image)

# --- 分类推理 ---
cls_inputs = cls_processor(image, return_tensors="pt")
cls_inputs = {k: v.to(device) for k, v in cls_inputs.items()}

with torch.no_grad():
    cls_outputs = cls_model(**cls_inputs)
    cls_pred = torch.argmax(cls_outputs.logits, dim=1).item()
    cls_probs = torch.softmax(cls_outputs.logits, dim=1).cpu().numpy()[0]

cls_label = "裂缝" if cls_pred == 1 else "正常"
cls_conf = cls_probs[cls_pred]

print(f"\n🔍 分类结果: {cls_label} (置信度: {cls_conf:.4f})")

# --- 分割推理 ---
seg_inputs = seg_processor(image, return_tensors="pt")
seg_inputs = {k: v.to(device) for k, v in seg_inputs.items()}

with torch.no_grad():
    seg_outputs = seg_model(**seg_inputs)
    seg_pred = torch.argmax(seg_outputs.logits, dim=1).squeeze().cpu().numpy()

crack_pixels = (seg_pred == 1).sum()
total_pixels = seg_pred.size
crack_ratio = crack_pixels / total_pixels

print(f"🔍 分割结果: 裂缝像素占比 {crack_ratio:.4f}")

# ============================================
# 4. 可视化
# ============================================
fig, axes = plt.subplots(1, 4, figsize=(16, 5))

axes[0].imshow(image_np)
axes[0].set_title("原图")
axes[0].axis("off")

axes[1].imshow(seg_pred, cmap="jet")
axes[1].set_title("分割掩膜")
axes[1].axis("off")

# 叠加显示
overlay = image_np.copy()
overlay[seg_pred == 1] = [255, 0, 0]  # 红色高亮裂缝
axes[2].imshow(overlay)
axes[2].set_title("叠加显示")
axes[2].axis("off")

# 分类结果显示
axes[3].text(0.5, 0.5,
             f"分类: {cls_label}\n置信度: {cls_conf:.4f}\n裂缝占比: {crack_ratio:.4f}",
             ha="center", va="center", fontsize=14, transform=axes[3].transAxes)
axes[3].set_title("结果")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("inference_result.png", dpi=200)
print("\n📸 结果已保存到: inference_result.png")
plt.show()