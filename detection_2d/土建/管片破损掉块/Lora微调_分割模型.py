#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 微调分割训练 - SegFormer mit-b5
管片破损掉块像素级分割（背景 / 管片破损掉块）
"""
import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

print("=" * 70)
print("LoRA 微调 SegFormer mit-b5 管片破损掉块分割训练")
print("=" * 70)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
DATASET_DIR = DATA_ROOT / "管片破损掉块_dataset"
OUTPUT_DIR = PROJECT_ROOT / "guanpian_segformer_results"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "nvidia/mit-b5"

# 预训练权重：裂缝分割微调完成后的 LoRA adapter 输出目录
# 由 土建/裂缝/Lora微调_分割模型.py 训练后产生（trainer.save_model 保存的是 PEFT adapter）
PRETRAINED_SEG_ADAPTER = PROJECT_ROOT / "segformer_results" / "segmentation_model"

TRAIN_CONFIG = {
    "num_train_epochs": 100,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 0.1,  # transformers 5.15: <1 表示占总步数的比例(原 warmup_ratio)
    "lr_scheduler_type": "cosine",
    "image_size": 512,
    "use_lora": True,
    "lora_r": 256,
    "lora_alpha": 256,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
    "lora_bias": "lora_only",
    "lora_modules_to_save": ["decode_head"],
}

print("\n训练配置:")
for k, v in TRAIN_CONFIG.items():
    print(f"  {k}: {v}")


def find_samples(split_dir):
    """获取分割数据集的图片和掩膜路径"""
    img_dir = split_dir / "images"
    mask_dir = split_dir / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        return []

    samples = []
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = f.rsplit('.', 1)[0]
            img_path = img_dir / f
            mask_path = mask_dir / f"{name}_mask.png"
            if mask_path.exists():
                samples.append((str(img_path), str(mask_path)))
    return samples


def split_data():
    """加载划分数据"""
    train_pairs = find_samples(DATASET_DIR / "train")
    val_pairs = find_samples(DATASET_DIR / "val")
    test_pairs = find_samples(DATASET_DIR / "test")

    print("\n数据统计:")
    print(f"  训练: {len(train_pairs)} 对")
    print(f"  验证: {len(val_pairs)} 对")
    print(f"  测试: {len(test_pairs)} 对")

    return train_pairs, val_pairs, test_pairs


class SegmentDataset(Dataset):
    def __init__(self, pairs, processor, image_size=512):
        self.pairs = pairs
        self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = (mask > 127).astype(np.int64)

        inputs = self.processor(
            image, mask,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size}
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": inputs["labels"].squeeze(0),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    valid_mask = labels_flat != 0
    if valid_mask.sum() == 0:
        return {"guanpian_IoU": 0.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0}

    preds_valid = preds_flat[valid_mask]
    labels_valid = labels_flat[valid_mask]

    tp = np.logical_and(preds_valid, labels_valid).sum()
    fp = np.logical_and(preds_valid, np.logical_not(labels_valid)).sum()
    fn = np.logical_and(np.logical_not(preds_valid), labels_valid).sum()

    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {"guanpian_IoU": iou, "F1": f1, "Precision": precision, "Recall": recall}


def main():
    train_pairs, val_pairs, test_pairs = split_data()

    print("\n加载模型...")
    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        size={"height": TRAIN_CONFIG["image_size"], "width": TRAIN_CONFIG["image_size"]}
    )

    model = AutoModelForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
        id2label={0: "background", 1: "guanpianposun"},
        label2id={"background": 0, "guanpianposun": 1},
        use_safetensors=True,
    )

    # 迁移学习：在裂缝分割微调模型基础上继续训练
    if PRETRAINED_SEG_ADAPTER.exists():
        print(f"\n[迁移学习] 从裂缝分割微调模型加载预训练权重")
        print(f"  预训练路径: {PRETRAINED_SEG_ADAPTER}")
        print("\n  加载裂缝 LoRA adapter...")
        model = PeftModel.from_pretrained(model, str(PRETRAINED_SEG_ADAPTER), is_trainable=False)
        print("  合并裂缝 LoRA 权重进基础模型（decode_head 保留裂缝微调结果）...")
        model = model.merge_and_unload()
        print("  合并完成，模型当前已在裂缝分割能力基础上")
    else:
        raise FileNotFoundError(
            f"[迁移学习前置条件不满足] 未找到裂缝分割微调模型: {PRETRAINED_SEG_ADAPTER}\n"
            f"  请先运行 土建/裂缝/Lora微调_分割模型.py 完成裂缝微调，再继续管片破损掉块迁移学习。\n"
            f"  （取消注释下面两行可改为从 {MODEL_NAME} 基础模型直接训练）"
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

    print("\n创建数据集...")
    train_dataset = SegmentDataset(train_pairs, processor, TRAIN_CONFIG["image_size"])
    val_dataset = SegmentDataset(val_pairs, processor, TRAIN_CONFIG["image_size"])
    test_dataset = SegmentDataset(test_pairs, processor, TRAIN_CONFIG["image_size"])

    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")

    print("\n开始训练...")
    print("=" * 70)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
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
        load_best_model_at_end=True,
        metric_for_best_model="guanpian_IoU",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        optim="adamw_torch",
        deepspeed=os.path.join(SCRIPT_DIR, "ds_seg.json")
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

    print("\n保存模型...")
    seg_output_dir = OUTPUT_DIR / "segmentation_model"
    trainer.save_model(str(seg_output_dir))
    model.save_pretrained(str(seg_output_dir / "lora_adapter"))
    processor.save_pretrained(str(seg_output_dir / "processor"))

    print(f"\n模型已保存: {seg_output_dir}")

    print("\n测试集评估...")
    test_metrics = trainer.evaluate(test_dataset)
    print("\n测试集结果:")
    for k, v in test_metrics.items():
        if any(m in k for m in ["guanpian_IoU", "F1", "Precision", "Recall"]):
            print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 70)
    print("分割训练完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
