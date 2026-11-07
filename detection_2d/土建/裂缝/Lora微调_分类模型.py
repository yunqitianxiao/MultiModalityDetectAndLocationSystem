#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA 微调分类训练 - SegFormer mit-b5
判断图像是否存在裂缝（正常 / 裂缝）
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import sys
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ============================================
# 1. 配置
# ============================================
print("=" * 70)
print("🚀 LoRA 微调 SegFormer mit-b5 裂缝分类训练")
print("=" * 70)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLIT_FILE = os.path.join(DATA_DIR, "dataset_split/dataset_split.pkl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "classifier_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "nvidia/mit-b5"

# ============================================
# 2. LoRA 训练参数 (7×A100-PCIe-40GB 全力配置)
# ============================================
TRAIN_CONFIG = {
    # ============ 训练轮次 ============
    "num_train_epochs": 100,

    # ============ Batch Size (7×A100-PCIe-40GB 配置) ============
    # 注意: 这里写的是"每卡"batch, 不是总batch。
    # 7 卡 × 22 = 单步总 batch 154。
    # 显存: 每卡 22×1.6GB(激活) + 固定~1.3GB ≈ 36.5GB, 40GB 卡余量 ~3.5GB。
    #       batch 24 会到 ~39.7GB 太贴边易 OOM, 不建议; 若 OOM 先降到 20。
    "per_device_train_batch_size": 22,
    "per_device_eval_batch_size": 22,
    "gradient_accumulation_steps": 1,

    # ============ 优化器参数 ============
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,

    # ============ 学习率调度器 ============
    "warmup_steps": 0.1,  # transformers 5.15: <1 表示占总步数的比例(原 warmup_ratio)
    "lr_scheduler_type": "cosine",

    # ============ 分辨率 (512 更稳定) ============
    "image_size": 512,

    # ============ LoRA 参数 (200GB 可用最大 rank) ============
    "use_lora": True,
    "lora_r": 1024,
    "lora_alpha": 1024,
    "lora_dropout": 0.03,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
    "lora_bias": "lora_only",
    "lora_modules_to_save": ["classifier"],
}

print("\n⚙️  训练配置 (LoRA) :")
for key, value in TRAIN_CONFIG.items():
    print(f"  {key}: {value}")

# ============================================
# 3. 加载数据
# ============================================
if not os.path.exists(SPLIT_FILE):
    print(f"\n❌ 错误: 找不到数据划分文件 {SPLIT_FILE}")
    sys.exit(1)

with open(SPLIT_FILE, "rb") as f:
    split_data = pickle.load(f)

crack_train = split_data["crack_train"]
crack_val = split_data["crack_val"]
crack_test = split_data["crack_test"]
normal_train = split_data["normal_train"]
normal_val = split_data["normal_val"]
normal_test = split_data["normal_test"]

print("\n📊 数据统计:")
print(f"  裂缝 - 训练: {len(crack_train)}, 验证: {len(crack_val)}, 测试: {len(crack_test)}")
print(f"  正常 - 训练: {len(normal_train)}, 验证: {len(normal_val)}, 测试: {len(normal_test)}")


# ============================================
# 4. 数据集类（方案A：掩膜引导随机裁剪 + 随机缩放增广）
#    - 训练时(augment=True): 裂缝样本在掩膜周围随机裁剪窗口(窗口尺寸随机≈随机缩放),
#      保证每个正样本都真正含裂缝, 且在大图上不丢失细裂缝细节; 正常样本随机裁剪。
#    - 验证/测试(augment=False): 整图缩放, 保证结果确定性。
# ============================================
class CrackClassificationDataset(Dataset):
    def __init__(self, crack_pairs, normal_images, processor, image_size=512, augment=False):
        self.samples = []  # (img_path, mask_path_or_None, label)

        for img_path, mask_path in crack_pairs:
            self.samples.append((img_path, mask_path, 1))

        for img_path in normal_images:
            self.samples.append((img_path, None, 0))

        self.processor = processor
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _random_resized_crop(image, min_crop=224):
        """随机缩放+裁剪(窗口尺寸随机 = 对图像做随机缩放): 用于正常样本/兜底"""
        W, H = image.size
        base = min(W, H)
        crop_size = random.randint(min_crop, base) if base >= min_crop else base
        x0 = random.randint(0, W - crop_size)
        y0 = random.randint(0, H - crop_size)
        return image.crop((x0, y0, x0 + crop_size, y0 + crop_size))

    @staticmethod
    def _mask_guided_crop(image, mask_path, min_crop=224, max_crop=1024):
        """掩膜引导裁剪: 窗口必须覆盖裂缝, 优先以裂缝像素为中心;
        窗口尺寸在 [min_crop, max_crop] 随机 ≈ 随机缩放。
        大隧道图(如4096px)裁 224~1024 的小窗, 细裂缝在原生分辨率下不丢失。
        注意: 掩膜可能是小尺寸缩略图(隧道掩膜只有224x224)且灰度偏低(jpeg压缩到~110),
        因此阈值自适应, 且要把掩膜坐标按比例映射回原图坐标。"""
        mask = np.array(Image.open(mask_path).convert("L"))
        mh, mw = mask.shape
        thr = max(30, int(mask.max() * 0.2))  # 自适应阈值: 公共掩膜255 / 隧道掩膜~110
        ys, xs = np.where(mask > thr)
        W, H = image.size
        # 掩膜坐标 -> 原图坐标
        xs = (xs * W / mw).astype(np.int64)
        ys = (ys * H / mh).astype(np.int64)
        base = min(W, H)
        crop_size = random.randint(min_crop, min(max_crop, base))
        half = crop_size // 2
        # 优先选能居中放置的裂缝像素, 避免裂缝被挤到窗口边缘
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
    """方案A 推理: 在原始分辨率上滑窗分块分类, 返回所有块中'裂缝'概率最大值。
    大图(如4096px)不整体缩放, 细裂缝在每块里仍以原生分辨率出现, 不会因缩放消失。"""
    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    # processor 返回 CPU float32, 必须搬到模型所在设备并转成模型精度(bf16), 否则 dtype 不匹配报错
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    def _to_device(inputs):
        return {k: (v.to(device=device, dtype=dtype) if v.dtype.is_floating_point else v.to(device=device))
                for k, v in inputs.items()}

    if min(W, H) <= window:
        inputs = _to_device(processor(image, return_tensors="pt",
                                      size={"height": window, "width": window}))
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
            inputs = _to_device(processor(tile, return_tensors="pt",
                                          size={"height": window, "width": window}))
            with torch.no_grad():
                logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            if prob > best:
                best = prob
    return best


# ============================================
# 5. 加载模型并应用 LoRA
# ============================================
print("\n📦 加载模型并应用 LoRA...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    size={"height": TRAIN_CONFIG["image_size"], "width": TRAIN_CONFIG["image_size"]}
)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,
    id2label={0: "normal", 1: "crack"},
    label2id={"normal": 0, "crack": 1},
    use_safetensors=True,
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

# ============================================
# 6. 创建数据集
# ============================================
print("\n📁 创建数据集...")

train_dataset = CrackClassificationDataset(
    crack_train, normal_train, processor,
    image_size=TRAIN_CONFIG["image_size"], augment=True
)
val_dataset = CrackClassificationDataset(
    crack_val, normal_val, processor,
    image_size=TRAIN_CONFIG["image_size"], augment=False
)

print(f"  训练集大小: {len(train_dataset)}")
print(f"  验证集大小: {len(val_dataset)}")
print(f"  测试集大小: {len(crack_test) + len(normal_test)}")


# ============================================
# 7. 评估函数
# ============================================
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

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


# ============================================
# 8. 训练
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
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=10,
    report_to="none",
    bf16=True,  # ✅ 改成 True
    fp16=False,  # ✅ 改成 False
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    gradient_checkpointing=False,  # 提速: 关检查点。7×40GB 卡 batch22 时每卡 ~36.5GB, 关着即可; 若 OOM 改回 True
    optim="adamw_torch",
    deepspeed=os.path.join(SCRIPT_DIR, "csds.json")
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

# ============================================
# 9. 保存
# ============================================
print("\n💾 保存模型...")

cls_output_dir = os.path.join(OUTPUT_DIR, "classification_model")
os.makedirs(cls_output_dir, exist_ok=True)

# 保存完整模型（含 LoRA 权重 merged 后的状态）
trainer.save_model(cls_output_dir)
# 单独保存 LoRA adapter，便于后续推理脚本通过 PeftModel 加载
model.save_pretrained(os.path.join(cls_output_dir, "lora_adapter"))
processor.save_pretrained(os.path.join(cls_output_dir, "processor"))

print(f"✅ 分类模型已保存到: {cls_output_dir}")
print(f"   - 完整模型: {cls_output_dir}")
print(f"   - LoRA adapter: {cls_output_dir}/lora_adapter")
print(f"   - 处理器: {cls_output_dir}/processor")

# 测试集评估只在 rank0 执行, 其余 rank 训练完直接退出(避免 7 份重复滑窗推理)
if getattr(trainer.args, "process_index", 0) != 0:
    print(f"[rank {trainer.args.process_index}] 训练完成, 跳过测试集评估。")
    sys.exit(0)

# 测试集评估（方案A：大图滑窗分块 + 逐张输出）
# 注意: 测试集里的隧道域大图(如4096px)不能整体缩放(细裂缝会消失),
# 这里按 window 在原始分辨率上滑窗, 任一块判裂缝即整图判裂缝。
print("\n📊 测试集评估（滑窗分块）...")
print(f"  裂缝样本: {len(crack_test)} 张, 正常样本: {len(normal_test)} 张")

y_true, y_prob, per_image = [], [], []
for img_path, _ in crack_test:
    prob = sliding_window_predict(model, processor, img_path)
    y_true.append(1); y_prob.append(prob)
    per_image.append((os.path.basename(img_path), 1, prob))
for img_path in normal_test:
    prob = sliding_window_predict(model, processor, img_path)
    y_true.append(0); y_prob.append(prob)
    per_image.append((os.path.basename(img_path), 0, prob))

y_pred = [1 if p > 0.5 else 0 for p in y_prob]
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
auc = roc_auc_score(y_true, y_prob)

print("\n📈 测试集结果（滑窗）:")
print(f"  accuracy : {acc:.4f}")
print(f"  precision: {prec:.4f}")
print(f"  recall   : {rec:.4f}")
print(f"  f1       : {f1:.4f}")
print(f"  auc      : {auc:.4f}")

print("\n📋 隧道域裂缝图逐张结果(最关键, 6张):")
for name, label, prob in sorted(per_image):
    if name.startswith("06"):
        mark = "检出✓" if prob > 0.5 else "漏检✗"
        print(f"  {name:16s} 概率={prob:.3f}  {mark}")

print("\n📋 全部逐张结果:")
for name, label, prob in sorted(per_image):
    pred = "裂缝" if prob > 0.5 else "正常"
    mark = "✓" if (label == 1) == (prob > 0.5) else "✗"
    truth = "裂缝" if label == 1 else "正常"
    print(f"  {name:22s} 真实={truth}  概率={prob:.3f}  预测={pred}  {mark}")

print("\n" + "=" * 70)
print("✅ 分类训练完成！")
print("=" * 70)
