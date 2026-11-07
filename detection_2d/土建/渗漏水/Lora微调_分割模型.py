#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 微调分割训练 - SegFormer mit-b5 (渗漏水迁移版)
渗漏水像素级分割（背景 / 渗漏水）
照 土建/裂缝/Lora微调_分割模型.py 迁移: 类别 crack→leakage, 读 leakage_split.pkl,
输出到独立目录 segformer_results_leakage。7×V100-32GB fp16 配置不变。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import sys
import pickle
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ============================================
# 1. 配置路径
# ============================================
print("=" * 70)
print("🚀 LoRA 微调 SegFormer mit-b5 渗漏水分割训练(迁移版)")
print("=" * 70)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLIT_FILE = os.path.join(DATA_DIR, "dataset_split/leakage_split.pkl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "segformer_results_leakage")  # 独立, 不覆盖裂缝结果
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "nvidia/mit-b5"

# ============================================
# 2. LoRA 训练参数 (7×V100-32GB 分布式配置, 与裂缝版一致)
# ============================================
TRAIN_CONFIG = {
    "num_train_epochs": 100,

    # Batch Size (7×V100-32GB): 640² batch10 ≈ 28GB/卡
    "per_device_train_batch_size": 10,
    "per_device_eval_batch_size": 10,
    "gradient_accumulation_steps": 1,

    "learning_rate": 1.4e-4,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,

    "warmup_steps": 0.1,  # transformers 5.15: <1 表示占总步数比例
    "lr_scheduler_type": "cosine",

    "image_size": 640,

    "use_lora": True,
    "lora_r": 1024,
    "lora_alpha": 1024,
    "lora_dropout": 0.03,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
    "lora_bias": "lora_only",
    "lora_modules_to_save": ["decode_head"],
}

print("\n⚙️  训练配置 (LoRA) :")
for key, value in TRAIN_CONFIG.items():
    print(f"  {key}: {value}")

# ============================================
# 3. 加载数据
# ============================================
if not os.path.exists(SPLIT_FILE):
    print(f"\n❌ 错误: 找不到数据划分文件 {SPLIT_FILE}")
    print("   请先运行: python3 土建/渗漏水/数据整理和划分.py")
    sys.exit(1)

with open(SPLIT_FILE, "rb") as f:
    split_data = pickle.load(f)

leakage_train = split_data["leakage_train"]
leakage_val = split_data["leakage_val"]
leakage_test = split_data["leakage_test"]
normal_train = split_data["normal_train"]
normal_val = split_data["normal_val"]
normal_test = split_data["normal_test"]

print("\n📊 数据统计:")
print(f"  渗漏水 - 训练: {len(leakage_train)}, 验证: {len(leakage_val)}, 测试: {len(leakage_test)}")
print(f"  正常   - 训练: {len(normal_train)}, 验证: {len(normal_val)}, 测试: {len(normal_test)}")


# ============================================
# 4. 数据集类
# ============================================
class LeakageSegmentationDataset(Dataset):
    def __init__(self, leakage_pairs, normal_images, processor, image_size=512):
        self.leakage_pairs = leakage_pairs
        self.normal_images = normal_images
        self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.leakage_pairs) + len(self.normal_images)

    def __getitem__(self, idx):
        if idx < len(self.leakage_pairs):
            img_path, mask_path = self.leakage_pairs[idx]
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
            mask = (mask > 127).astype(np.int64)
        else:
            normal_idx = idx - len(self.leakage_pairs)
            img_path = self.normal_images[normal_idx]
            image = Image.open(img_path).convert("RGB")
            mask = np.zeros((image.height, image.width), dtype=np.int64)

        inputs = self.processor(
            image,
            mask,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size}
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": inputs["labels"].squeeze(0),
        }


# ============================================
# 5. 加载模型并应用 LoRA
# ============================================
print("\n📦 加载模型 (LoRA)...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    size={"height": TRAIN_CONFIG["image_size"], "width": TRAIN_CONFIG["image_size"]}
)

model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,
    id2label={0: "background", 1: "leakage"},
    label2id={"background": 0, "leakage": 1},
    use_safetensors=True,
)

if TRAIN_CONFIG.get("use_lora", True):
    lora_config = LoraConfig(
        r=TRAIN_CONFIG["lora_r"],
        lora_alpha=TRAIN_CONFIG["lora_alpha"],
        target_modules=TRAIN_CONFIG["lora_target_modules"],
        lora_dropout=TRAIN_CONFIG["lora_dropout"],
        bias=TRAIN_CONFIG["lora_bias"],
        modules_to_save=TRAIN_CONFIG["lora_modules_to_save"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数量: {trainable_params / 1e6:.2f}M (全参数)")

# ============================================
# 6. 创建数据集
# ============================================
print("\n📁 创建数据集...")

train_dataset = LeakageSegmentationDataset(
    leakage_train, normal_train, processor,
    image_size=TRAIN_CONFIG["image_size"]
)
val_dataset = LeakageSegmentationDataset(
    leakage_val, normal_val, processor,
    image_size=TRAIN_CONFIG["image_size"]
)

print(f"  训练集大小: {len(train_dataset)}")
print(f"  验证集大小: {len(val_dataset)}")
print(f"  图像尺寸: {TRAIN_CONFIG['image_size']}x{TRAIN_CONFIG['image_size']}")


# ============================================
# 7. 评估函数
# ============================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    # SegFormer decode head 输出为输入 1/4 分辨率(640→160), 上采样到标签分辨率再算
    if labels.ndim == 3 and preds.shape[1:] != labels.shape[1:]:
        preds = np.stack([
            np.array(Image.fromarray(p.astype(np.uint8)).resize(
                (labels.shape[2], labels.shape[1]), Image.NEAREST))
            for p in preds
        ])

    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    tp = np.logical_and(preds_flat == 1, labels_flat == 1).sum()
    fp = np.logical_and(preds_flat == 1, labels_flat == 0).sum()
    fn = np.logical_and(preds_flat == 0, labels_flat == 1).sum()
    tn = np.logical_and(preds_flat == 0, labels_flat == 0).sum()

    iou_leakage = tp / (tp + fp + fn + 1e-6)
    iou_bg = tn / (tn + fp + fn + 1e-6)
    mIoU = (iou_leakage + iou_bg) / 2
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "leakage_IoU": iou_leakage,
        "mIoU": mIoU,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
    }


# ============================================
# 8. 训练配置
# ============================================
print("\n🔥 开始训练...")
print("=" * 70)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=TRAIN_CONFIG["num_train_epochs"],
    per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=TRAIN_CONFIG["per_device_eval_batch_size"],
    gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
    learning_rate=TRAIN_CONFIG["learning_rate"],
    weight_decay=TRAIN_CONFIG["weight_decay"],
    warmup_steps=TRAIN_CONFIG["warmup_steps"],
    lr_scheduler_type=TRAIN_CONFIG["lr_scheduler_type"],
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    # 显存吃满(640² batch10 ~28.8GB)时 load_best_model_at_end 会在 _load_best_model 阶段 OOM。
    # 改为 False, 训练结束后手动释放显存缓存再加载 best checkpoint(见下方 train() 之后)。
    load_best_model_at_end=False,
    metric_for_best_model="leakage_IoU",
    greater_is_better=True,
    logging_steps=10,
    report_to="none",
    bf16=False,  # V100(SM7.0) 无原生 bf16 → 关闭
    fp16=True,   # V100 原生 fp16
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_checkpointing=False,
    optim="adamw_torch",
    # 不用 DeepSpeed(DS 纯 fp16 引擎会让 SegFormer loss NaN), 用 torchrun DDP 原生 fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],
)

trainer.train()

# 训练结束: 释放训练期显存缓存, 再手动加载 best checkpoint(否则 load_best_model_at_end 会 OOM)。
# best_model_checkpoint 由 metric_for_best_model 自动跟踪, 与 load_best_model_at_end 无关。
import gc
gc.collect()
torch.cuda.empty_cache()
if getattr(trainer.state, "best_model_checkpoint", None):
    active_adapter = getattr(model, "active_adapter", None) or "default"
    model.load_adapter(trainer.state.best_model_checkpoint, active_adapter)
    model.set_adapter(active_adapter)
    print(f"✅ 已加载最佳 checkpoint: {trainer.state.best_model_checkpoint} (leakage_IoU={trainer.state.best_metric:.4f})")

# ============================================
# 9. 保存
# ============================================
print("\n💾 保存模型...")

seg_output_dir = os.path.join(OUTPUT_DIR, "segmentation_model")
os.makedirs(seg_output_dir, exist_ok=True)

trainer.save_model(seg_output_dir)
model.save_pretrained(os.path.join(seg_output_dir, "lora_adapter"))
processor.save_pretrained(os.path.join(seg_output_dir, "processor"))

print(f"✅ 分割模型已保存到: {seg_output_dir}")
print(f"   - 完整模型: {seg_output_dir}")
print(f"   - LoRA adapter: {seg_output_dir}/lora_adapter")
print(f"   - 处理器: {seg_output_dir}/processor")

# 测试集评估
print("\n📊 在测试集上评估...")
test_dataset = LeakageSegmentationDataset(
    leakage_test, normal_test, processor,
    image_size=TRAIN_CONFIG["image_size"]
)
test_metrics = trainer.evaluate(test_dataset)

print("\n📈 测试集结果:")
for key, value in test_metrics.items():
    if any(m in key for m in ["leakage_IoU", "F1", "Precision", "Recall"]):
        print(f"  {key}: {value:.4f}")

print("\n" + "=" * 70)
print("✅ 渗漏水分割训练完成！")
print("=" * 70)
