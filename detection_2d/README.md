# 轨道交通/隧道病害智能检测系统

基于 **SegFormer (MiT-B5) + LoRA** 的细粒度病害检测项目，支持图像分类（有无病害）与语义分割（像素级病灶定位）。训练脚本采用 **LoRA 低秩适配**，在有限显存下即可微调大模型，同时保留切换为全参数微调的开关。

---

## 项目简介

本项目面向地铁隧道、轨道线路等场景，利用 NVIDIA 的 `SegFormer` 视觉 Transformer 骨干网络，结合 LoRA 低秩适配技术完成：

- **多分类病害识别**：判断图像是否存在裂缝（正常 / 裂缝）；
- **像素级裂缝分割**：定位裂缝轮廓，输出像素级掩膜；
- **主动学习支持**：通过不确定性采样筛选高价值样本，降低标注成本。

项目采用 Hugging Face `transformers` 生态，在中国大陆环境默认使用 `hf-mirror.com` 镜像下载模型，并针对 `torch<2.6` 做了 `.bin → .safetensors` 自动转换，避免新版 transformers 的安全限制报错。

---

## 目录结构

```
.
├── 下载segFormer.py                  # 下载 SegFormer-B5 骨干（含 LoRA 封装示例）
├── 土建/
│   ├── 裂缝/
│   │   ├── 数据整理和划分.py          # 配对图像-掩膜，划分训练/验证/测试集
│   │   ├── 隧道裂缝掩膜标注工具.py    # 隧道裂缝掩膜标注（白=裂缝 / 黑=背景，全分辨率 PNG）
│   │   ├── Lora微调_分类模型.py       # 二分类（正常/裂缝）LoRA 微调训练
│   │   ├── Lora微调_分割模型.py       # 裂缝像素级分割 LoRA 微调训练（可切全参数微调）
│   │   └── 模型推理测试.py            # 分类+分割联合推理与可视化
│   └── 管片破损掉块/
│       ├── 数据划分.py                # 管片掉块数据集划分（正常/病害，无掩膜病害图排除）
│       ├── 高相似度标注工具.py        # 相似度去重辅助工具
│       ├── 主动学习筛选.py            # 主动学习不确定性采样
│       ├── Lora微调_分类模型.py       # 管片掉块二分类
│       └── Lora微调_分割模型.py       # 管片掉块分割（迁移自裂缝预训练）
├── data/
│   ├── 土建_裂缝/                     # 裂缝原图
│   ├── 土建_裂缝_手动标注的掩膜/       # 对应裂缝掩膜
│   ├── 土建_正常/                     # 正常样本
│   └── dataset_split/
│       └── dataset_split.pkl          # 数据划分结果（由 数据整理和划分.py 生成）
├── 病害训练集+正常状态/病害训练集/     # 多类病害数据集（YOLO 格式标签）
│   ├── 类别说明.txt
│   ├── class.txt
│   ├── images/
│   └── labels/
├── 病害训练集+正常状态/正常状态/       # 各场景正常样本
│   ├── 土建/
│   ├── 设备/
│   └── 轨道/
├── classifier_results/               # 分类训练输出
│   ├── classification_model/         # 完整模型
│   │   ├── lora_adapter/             # LoRA adapter
│   │   └── processor/                # 图像处理器
│   └── logs/                         # 训练日志
├── segformer_results/                # 分割训练输出
│   ├── segmentation_model/           # 完整模型
│   │   ├── lora_adapter/             # LoRA adapter
│   │   └── processor/                # 图像处理器
│   └── logs/                         # 训练日志
├── active_learning_results/          # 主动学习相关结果
│   ├── recommended_for_labeling.txt  # 推荐标注样本列表
│   └── random_samples_for_labeling/  # 推荐标注样本可视化图
├── hf_cache/                         # 本地 HuggingFace 模型缓存
└── .venv/                            # Python 虚拟环境
```

---

## 病害类别

| 编号 | 类别英文名 | 中文名 | 所属大类 |
|------|-----------|--------|---------|
| 0 | koujianwaixie | 扣件松动歪斜 | 轨道 |
| 1 | koujianduanlie | 扣件断裂 | 轨道 |
| 2 | koujianqueshi | 扣件缺失 | 轨道 |
| 3 | yiwu | 异物入侵 | 设备 |
| 4 | guanxiansongtuo | 管线支架松脱 | 设备 |
| 5 | guanpianposun | 管片破损掉块 | 土建 |
| 6 | liefeng | 裂缝 | 土建 |
| 7 | shenloushui | 渗漏水 | 土建 |
| 8 | normal | 正常 | — |

> 当前已实现完整流程的是 **土建/裂缝** 二分类与分割任务；多类病害可在此基础上扩展。

---

## 环境要求

- Python >= 3.10
- CUDA >= 12.1
- NVIDIA GPU：推荐显存 **>= 16 GB**（单卡配置下：管片分割 batch 8 ≈ 14GB/卡；裂缝分类 batch 22/卡、7 卡分布式下关检查点 ≈ 36.5GB/卡、开检查点 ≈ 5GB/卡；显存越大可将 `image_size` 与 `batch_size` 调大）
- 主要依赖：
  - `torch 2.5.1+cu121`
  - `torchvision 0.20.1+cu121`
  - `transformers >= 4.x`
  - `peft >= 0.20.0`
  - `huggingface_hub >= 1.x`
  - `safetensors`
  - `scikit-learn`
  - `Pillow`, `numpy`, `opencv-python`, `matplotlib`

### 安装依赖

```bash
# 使用项目已有虚拟环境
source .venv/bin/activate

# 或自行安装
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft huggingface_hub safetensors scikit-learn Pillow opencv-python matplotlib
```

> 若显存仍不足，可进一步调小 `TRAIN_CONFIG["image_size"]`、`per_device_train_batch_size` 或增大 `gradient_accumulation_steps`。分割脚本也可将 `TRAIN_CONFIG["use_lora"]` 设为 `False` 切换为全参数微调（需大显存）。

---

## 快速开始

### 1. 下载 SegFormer-B5 骨干模型

```bash
python 下载segFormer.py
```

脚本会：
- 通过 `hf-mirror.com` 下载 `nvidia/mit-b5`；
- 缓存到 `./hf_cache`；
- 自动将 `pytorch_model.bin` 转为 `model.safetensors`（适配 `torch<2.6`）；
- 输出 LoRA 封装后的可训练参数量。

> 若网络不通，可编辑脚本开头取消代理注释：
> ```python
> os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
> os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
> ```

### 2. 准备数据

确保 `data/` 下包含：

```
data/
├── 土建_裂缝/                 # 裂缝图像
├── 土建_裂缝_手动标注的掩膜/   # 裂缝掩膜（与图像同名）
└── 土建_正常/                 # 正常图像
```

执行数据划分：

```bash
python 土建/裂缝/数据整理和划分.py
```

生成 `data/dataset_split/dataset_split.pkl`。划分策略（当前数量：训练 411 / 验证 73 / 测试 30）：

- 隧道域裂缝图（`0601-0612`，真实目标域，含掩膜的 12 张）按 **5 训练 / 1 验证 / 6 测试** 分散；
- 公共数据集裂缝图（bitumen/concrete/masonry/runway 共 502 张）抽 24 张进测试集做稳定基准，其余按 85/15 分训练/验证；
- 正常图按 151 / 32 / 33 划分。

> 修改脚本顶部 `TEST_PUBLIC_SIZE` 可调整公共图进测试集的数量；`tunnel_train/tunnel_val/tunnel_test` 的切分长度可改隧道域分布。

> **隧道裂缝标注**：隧道域图（0601-0612）掩膜用 `土建/裂缝/隧道裂缝掩膜标注工具.py` 标注。
> 左键涂裂缝（白）、右键擦除（黑）、滚轮缩放、中键平移、`[`/`]` 调笔刷、`U` 撤销、`A`/`D` 翻页、`S` 保存。
> 保存为**全分辨率无损 PNG（纯白 255）**并自动删除旧 jpg；标注完成后重跑本划分脚本即可纳入。

### 3. 训练分类模型

```bash
# 7 卡分布式：每卡 22 张，单步总 batch 154
deepspeed --num_gpus 7 土建/裂缝/Lora微调_分类模型.py
```

- 任务：判断图像是否存在裂缝（正常 / 裂缝）；
- 策略：**LoRA 微调**（r=1024），输入分辨率 `512×512`；
- 数据增强（方案A）：训练时**掩膜引导随机裁剪 + 随机缩放**——正样本始终覆盖裂缝，大图细裂缝不因整体缩放丢失；
- 测试评估：**滑窗分块**——大图在原始分辨率上滑窗分类，任一块判裂缝即整图判裂缝，并**逐张输出隧道域图**的检出/漏检；
- 分布式：7 卡 × 22/卡（单步总 154）；`gradient_checkpointing` 默认关闭以提速（每卡约需 36.5GB，若 OOM 降到 20）；
- 输出：`classifier_results/classification_model/`；
- 指标：accuracy、precision、recall、f1、auc。

### 4. 训练分割模型

```bash
# 步骤 1：裂缝分割预训练（LoRA r=1024，7 卡分布式）
deepspeed --num_gpus 7 土建/裂缝/Lora微调_分割模型.py
```

- 任务：像素级裂缝分割（背景 / 裂缝）；
- 策略：**LoRA 微调**（默认 `use_lora=True`），输入分辨率 `512×512`；
- 输出：`segformer_results/segmentation_model/`（保存为 PEFT adapter）；
- 指标：mIoU、F1、Precision、Recall。

> `Lora微调_分割模型.py` 内部保留 `use_lora` 开关，设为 `False` 可切换为全参数微调（需大显存）。

### 4.1 管片破损掉块分割（迁移学习）

在裂缝分割预训练完成后再运行，自动加载裂缝 adapter 合并进基础模型：

```bash
# 步骤 2：管片掉块迁移训练（LoRA r=256）
python 土建/管片破损掉块/Lora微调_分割模型.py
```

- 任务：像素级管片破损掉块分割（背景 / 管片破损掉块）；
- 预训练：从 `segformer_results/segmentation_model` 加载裂缝微调权重，`merge_and_unload` 后叠加新的 LoRA；
- 数据：`data/管片破损掉块_dataset`（由 `土建/管片破损掉块/数据划分.py` 生成，无掩膜病害图已排除）；
- 输出：`guanpian_segformer_results/segmentation_model/`；
- 指标：guanpian_IoU、F1、Precision、Recall。

### 4.2 管片破损掉块分类

```bash
python 土建/管片破损掉块/Lora微调_分类模型.py
```

- 任务：判断图像是否有管片破损掉块（正常 / 病害）；
- 迁移：从裂缝分类 adapter（`classifier_results/classification_model`）加载合并后继续训练；
- 数据增强（方案A）：训练时**掩膜引导随机裁剪 + 随机缩放 + 随机翻转**；
- 测试评估：**滑窗分块** + 逐张输出；
- 指标：accuracy、precision、recall、f1、auc。

### 5. 推理测试

```bash
python 土建/裂缝/模型推理测试.py
```

修改脚本中的 `TEST_IMAGE` 路径。训练脚本会同时保存完整模型与 `lora_adapter`，`模型推理测试.py` 默认通过 `PeftModel.from_pretrained(..., "lora_adapter")` 加载，路径与 LoRA 训练输出一致。运行后输出 `inference_result.png`，包含：
- 原图；
- 分割掩膜；
- 裂缝叠加高亮图；
- 分类结果与置信度。

---

## 模型配置

### LoRA 微调参数（当前配置）

| 参数 | 分类（裂缝） | 分割（裂缝预训练） | 分割（管片迁移） |
|------|-------------|------------------|----------------|
| 骨干网络 | `nvidia/mit-b5` | `nvidia/mit-b5` | `nvidia/mit-b5`（加载裂缝 adapter 后迁移） |
| 微调方式 | LoRA | LoRA | LoRA |
| LoRA rank `r` / alpha | **1024** / **1024** | **1024** / **1024** | **256** / **256** |
| target_modules | `q_proj`,`v_proj`,`k_proj`,`out_proj` | 同左 | 同左 |
| modules_to_save | `classifier` | `decode_head` | `decode_head` |
| 输入分辨率 | 512×512 | 512×512 | 512×512 |
| 学习率 / 调度 | 3e-4, cosine+warmup10% | 3e-4, cosine+warmup10% | 3e-4, cosine+warmup10% |
| Batch / 卡 | **22**（7卡总154） | **22**（7卡总154） | **8** × 累积4 |
| 梯度累积 | 1 | 1 | 4 |
| Epochs / 早停 | 100 / 10 | 100 / 15 | 100 / 15 |
| 混合精度 | bf16 | bf16 | bf16 |
| 评估指标 | accuracy/precision/recall/f1/auc | IoU/F1/Precision/Recall | guanpian_IoU/F1/Precision/Recall |

> **显存提醒**：分类每卡 22 张 + 关检查点 ≈ **36.5GB/卡**（40GB 卡可跑，若 OOM 降到 20）；裂缝分割每卡 22 张 ≈ **36.5GB/卡**；管片分割每卡 8 张 ≈ **14GB/卡**。

### 全参数微调参数（备用）

若显存充足，可将 `Lora微调_分割模型.py` 中的 `use_lora` 设为 `False`，并参考以下配置：

| 参数 | 分类任务 | 分割任务 |
|------|---------|---------|
| 微调方式 | 全参数微调 | 全参数微调 |
| 输入分辨率 | 1024×1024 | 1024×1024 |
| 学习率 | 1e-4 | 1e-4 |
| Batch Size | 64 | 32 |
| 混合精度 | bf16 | bf16 |
| Epochs | 40 | 60 |

> 注意：SegFormer 的注意力投影层实际名称为 `q_proj` / `v_proj`，本项目的 LoRA 脚本已正确配置。若从其他来源复制 LoRA 配置时遇到 `No modules were targeted`，请检查是否为 `query` / `value` 旧名称。

---

## 关键技术说明

### 1. 中国大陆 HuggingFace 下载

所有脚本默认设置：

```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 2. torch<2.6 权重加载问题

新版 `transformers` 因安全漏洞（CVE-2025-32434）禁止在 `torch<2.6` 上加载 `.bin` 权重。`下载segFormer.py` 提供自动转换：

```python
pytorch_model.bin  --(weights_only=True)-->  model.safetensors
```

转换一次性完成，后续直接读取 `model.safetensors`。

### 3. LoRA 低秩微调

当前训练脚本默认采用 **LoRA 低秩微调**：

- 仅训练注意力投影层（`q_proj`、`v_proj`）与任务头（`classifier` / `decode_head`）；
- 冻结大部分骨干参数，显著降低显存占用；
- 可训练参数由 LoRA rank 决定：r=1024 时约 9400 万、r=256 时约 2700 万（其余骨干参数冻结）；
- 输入分辨率默认 512×512，在消费级 GPU 上即可训练。

若显存充足并希望追求更高精度，可在 `Lora微调_分割模型.py` 中设置：

```python
TRAIN_CONFIG["use_lora"] = False
```

切换为全参数微调，此时所有参数参与训练，可适当提高分辨率与 Batch Size。

---

## 扩展方向

1. **多类病害分割**：将 `num_labels` 从 2 改为 9，使用 `病害训练集+正常状态/` 下的 8 类病害 + 正常类；
2. **主动学习闭环**：利用 `active_learning_results/` 推荐未标注样本，迭代扩充训练集；
3. **模型部署**：将 LoRA adapter 与 base 模型合并后导出为 ONNX/TensorRT，用于边缘设备推理；
4. **数据增强**：增加随机裁剪、翻转、光照变化、Mosaic 等策略提升鲁棒性。

---

## 常见问题

**Q1: 下载模型时报错 `Repository Not Found`？**  
A: 检查能否访问 `https://hf-mirror.com/nvidia/mit-b5`，或配置代理后重试。

**Q2: 加载模型时报 `Due to a serious vulnerability issue in torch.load`？**  
A: 运行 `python 下载segFormer.py` 完成 `.bin → .safetensors` 转换即可。

**Q3: LoRA 报错 `No modules were targeted for adaptation`？**  
A: 将 `target_modules` 从 `["query", "value"]` 改为 `["q_proj", "v_proj"]`。

**Q4: 显存不足？**  
A: 
1. 减小 `TRAIN_CONFIG["image_size"]`（如从 512 改为 384）；
2. 减小 `per_device_train_batch_size`；
3. 增大 `gradient_accumulation_steps` 保持等效 batch；
4. 将 `fp16=True` 改为 `bf16=True` 仅当 GPU 支持 bf16 且显存仍够；
5. 或换用更小的骨干网络，如 `mit-b0/b1/b2`。

**Q5: 推理脚本找不到模型或报 `lora_adapter` 不存在？**  
A: 确认已运行 LoRA 训练脚本，模型保存路径应为 `classifier_results/classification_model/` 与 `segformer_results/segmentation_model/`，且子目录包含 `lora_adapter` 与 `processor`。`模型推理测试.py` 默认通过 `PeftModel.from_pretrained(base_model, ".../lora_adapter")` 加载，路径已与此对齐。

---

## 作者

ylf

---

## 许可证

本项目仅供学习与科研使用。模型权重遵循各自原始许可证（NVIDIA SegFormer 使用 Apache-2.0）。
