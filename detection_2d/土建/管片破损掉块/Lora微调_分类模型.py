#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 微调分类训练 - SegFormer mit-b5
管片破损掉块分类（正常 / 管片破损掉块）
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

print("=" * 70)
print("LoRA 微调 SegFormer mit-b5 管片破损掉块分类训练")
print("=" * 70)

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
DATASET_DIR = DATA_ROOT / "管片破损掉块_dataset"
OUTPUT_DIR = PROJECT_ROOT / "guanpian_classifier_results"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "nvidia/mit-b5"

# 预训练权重：裂缝分类微调完成后的 LoRA adapter 输出目录
# 由 土建/裂缝/Lora微调_分类模型.py 训练后产生（trainer.save_model 保存的是 PEFT adapter）
PRETRAINED_CLS_ADAPTER = PROJECT_ROOT / "classifier_results" / "classification_model"

TRAIN_CONFIG = {
    "num_train_epochs": 100,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,
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
    "lora_modules_to_save": ["classifier"],
}

print("\n训练配置:")
for k, v in TRAIN_CONFIG.items():
    print(f"  {k}: {v}")


def find_samples(split_dir):
    """获取分割数据集的图片和掩膜路径
    返回 (img_path, mask_path_or_None, label): 病害图保留掩膜供方案A裁剪用, 正常图无掩膜"""
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
                mask = np.array(Image.open(mask_path).convert("L"))
                label = 1 if mask.max() > 0 else 0
                mp = str(mask_path) if label == 1 else None
                samples.append((str(img_path), mp, label))
    return samples


def split_data():
    """加载划分数据"""
    train_samples = find_samples(DATASET_DIR / "train")
    val_samples = find_samples(DATASET_DIR / "val")
    test_samples = find_samples(DATASET_DIR / "test")

    disease_train = [s for s in train_samples if s[2] == 1]
    normal_train = [s for s in train_samples if s[2] == 0]
    disease_val = [s for s in val_samples if s[2] == 1]
    normal_val = [s for s in val_samples if s[2] == 0]
    disease_test = [s for s in test_samples if s[2] == 1]
    normal_test = [s for s in test_samples if s[2] == 0]

    print("\n数据统计:")
    print(f"  训练: {len(train_samples)} (病害{len(disease_train)}/正常{len(normal_train)})")
    print(f"  验证: {len(val_samples)} (病害{len(disease_val)}/正常{len(normal_val)})")
    print(f"  测试: {len(test_samples)} (病害{len(disease_test)}/正常{len(normal_test)})")

    return train_samples, val_samples, test_samples


class ClassDataset(Dataset):
    """方案A: 训练时掩膜引导随机裁剪 + 随机缩放 + 随机翻转; 验证/测试整图(确定性)"""
    def __init__(self, samples, processor, image_size=512, augment=False):
        # samples: (img_path, mask_path_or_None, label)
        self.samples = samples
        self.processor = processor
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _random_resized_crop(image, min_crop=224):
        """随机缩放+裁剪(窗口尺寸随机 = 对图像随机缩放): 用于正常样本"""
        W, H = image.size
        base = min(W, H)
        crop_size = random.randint(min_crop, base) if base >= min_crop else base
        x0 = random.randint(0, W - crop_size)
        y0 = random.randint(0, H - crop_size)
        return image.crop((x0, y0, x0 + crop_size, y0 + crop_size))

    @staticmethod
    def _mask_guided_crop(image, mask_path, min_crop=224, max_crop=640):
        """掩膜引导裁剪: 窗口必须覆盖掉块, 优先以掉块像素为中心;
        窗口尺寸在 [min_crop, max_crop] 随机 ≈ 随机缩放。
        掉块原图 640x640, 裁 224~640 的小窗, 保持缺陷在原生分辨率。"""
        mask = np.array(Image.open(mask_path).convert("L"))
        mh, mw = mask.shape
        thr = max(30, int(mask.max() * 0.2))  # 自适应阈值(管片掩膜为纯白255)
        ys, xs = np.where(mask > thr)
        W, H = image.size
        xs = (xs * W / mw).astype(np.int64)   # 掩膜坐标 -> 原图坐标
        ys = (ys * H / mh).astype(np.int64)
        base = min(W, H)
        crop_size = random.randint(min_crop, min(max_crop, base))
        half = crop_size // 2
        interior = (xs >= half) & (xs < W - half) & (ys >= half) & (ys < H - half)
        if interior.any():
            xs, ys = xs[interior], ys[interior]
        k = random.randrange(len(xs))
        cx, cy = int(xs[k]), int(ys[k])
        x0 = min(max(0, cx - half), W - crop_size)
        y0 = min(max(0, cy - half), H - crop_size)
        return image.crop((x0, y0, x0 + crop_size, y0 + crop_size))

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.augment:
            if label == 1 and mask_path is not None:
                try:
                    image = self._mask_guided_crop(image, mask_path)
                except Exception:
                    image = self._random_resized_crop(image)
            else:
                image = self._random_resized_crop(image)
            if random.random() < 0.5:  # 随机水平翻转
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        inputs = self.processor(
            image,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size}
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def sliding_window_predict(model, processor, img_path, window=512, stride=384):
    """方案A 推理(分辨率处理): 在原始分辨率上滑窗分块分类, 返回所有块中'病害'概率最大值。
    掉块图 640x640 会切成 ~4 块, 保持缺陷细节不因缩放丢失; 小图(<window)直接整图。"""
    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    if min(W, H) <= window:
        inputs = processor(image, return_tensors="pt",
                           size={"height": window, "width": window})
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=1)[0, 1].item()

    xs = list(range(0, W - window + 1, stride))
    if xs[-1] != W - window:
        xs.append(W - window)
    ys = list(range(0, H - window + 1, stride))
    if ys[-1] != H - window:
        ys.append(H - window)

    best = -1.0
    for y in ys:
        for x in xs:
            tile = image.crop((x, y, x + window, y + window))
            inputs = processor(tile, return_tensors="pt",
                               size={"height": window, "width": window})
            with torch.no_grad():
                logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            if prob > best:
                best = prob
    return best


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def main():
    train_samples, val_samples, test_samples = split_data()

    print("\n加载模型...")
    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        size={"height": TRAIN_CONFIG["image_size"], "width": TRAIN_CONFIG["image_size"]}
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
        id2label={0: "normal", 1: "guanpianposun"},
        label2id={"normal": 0, "guanpianposun": 1},
        use_safetensors=True,
    )

    # 迁移学习：在裂缝分类微调模型基础上继续训练
    if PRETRAINED_CLS_ADAPTER.exists():
        print(f"\n[迁移学习] 从裂缝分类微调模型加载预训练权重")
        print(f"  预训练路径: {PRETRAINED_CLS_ADAPTER}")
        print("\n  加载裂缝 LoRA adapter...")
        model = PeftModel.from_pretrained(model, str(PRETRAINED_CLS_ADAPTER), is_trainable=False)
        print("  合并裂缝 LoRA 权重进基础模型（classifier 保留裂缝微调结果）...")
        model = model.merge_and_unload()
        print("  合并完成，模型当前已在裂缝分类能力基础上")
    else:
        raise FileNotFoundError(
            f"[迁移学习前置条件不满足] 未找到裂缝分类微调模型: {PRETRAINED_CLS_ADAPTER}\n"
            f"  请先运行 土建/裂缝/Lora微调_分类模型.py 完成裂缝微调，再继续管片破损掉块迁移学习。\n"
            f"  （取消注释下面两行可改为从 {MODEL_NAME} 基础模型直接训练）"
        )

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
    train_dataset = ClassDataset(train_samples, processor, TRAIN_CONFIG["image_size"], augment=True)
    val_dataset = ClassDataset(val_samples, processor, TRAIN_CONFIG["image_size"], augment=False)
    test_dataset = ClassDataset(test_samples, processor, TRAIN_CONFIG["image_size"], augment=False)

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
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        optim="adamw_torch",
        deepspeed=os.path.join(SCRIPT_DIR, "ds_cls.json")
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    trainer.train()

    print("\n保存模型...")
    cls_output_dir = OUTPUT_DIR / "classification_model"
    trainer.save_model(str(cls_output_dir))
    model.save_pretrained(str(cls_output_dir / "lora_adapter"))
    processor.save_pretrained(str(cls_output_dir / "processor"))

    print(f"\n模型已保存: {cls_output_dir}")

    # 测试集评估（方案A: 滑窗分块 + 逐张输出）
    print("\n测试集评估（滑窗分块）...")
    y_true, y_prob, per_image = [], [], []
    for img_path, _, label in test_samples:
        prob = sliding_window_predict(model, processor, img_path)
        y_true.append(label)
        y_prob.append(prob)
        per_image.append((os.path.basename(img_path), label, prob))

    y_pred = [1 if p > 0.5 else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_prob)

    print("\n测试集结果（滑窗）:")
    print(f"  accuracy : {acc:.4f}")
    print(f"  precision: {prec:.4f}")
    print(f"  recall   : {rec:.4f}")
    print(f"  f1       : {f1:.4f}")
    print(f"  auc      : {auc:.4f}")

    print("\n逐张结果:")
    for name, label, prob in sorted(per_image):
        pred = "病害" if prob > 0.5 else "正常"
        mark = "✓" if (label == 1) == (prob > 0.5) else "✗"
        truth = "病害" if label == 1 else "正常"
        print(f"  {name:28s} 真实={truth}  概率={prob:.3f}  预测={pred}  {mark}")

    print("\n" + "=" * 70)
    print("分类训练完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
