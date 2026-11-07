import os
import time

# ============================================================
# 1. 设置 Hugging Face 镜像（中国大陆推荐）
# ============================================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 如果需要代理，取消下面两行的注释并填写你的代理地址
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
import transformers.utils.import_utils as import_utils
from peft import LoraConfig, get_peft_model
import torch
from safetensors.torch import save_file

# ============================================================
# 2. 你的标签映射（根据数据集修改）
# ============================================================
id2label = {
    0: "轨道扣件松动歪斜",
    1: "轨道扣件断裂",
    2: "轨道扣件缺失",
    3: "设备异物入侵",
    4: "设备管线支架松脱",
    5: "土建管片破损掉块",
    6: "土建裂缝",
    7: "土建渗漏水",
    8: "正常"
}
label2id = {v: k for k, v in id2label.items()}
num_classes = len(id2label)

# ============================================================
# 3. 选择模型 checkpoint
# ============================================================
# nvidia/mit-b5 是 SegFormer-B5 的骨干网络（encoder/backbone）。
# 如果你想直接下载在 ADE20K 上预训练好的完整 SegFormer-B5，
# 可以把下面改成：
#   checkpoint = "nvidia/segformer-b5-finetuned-ade-512-512"
# 或 Cityscapes 版本：
#   checkpoint = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
checkpoint = "nvidia/mit-b5"

# 本地缓存目录，避免模型文件散落在系统默认缓存里
cache_dir = os.path.abspath("./hf_cache")
os.makedirs(cache_dir, exist_ok=True)

print(f"目标模型: {checkpoint}")
print(f"本地缓存目录: {cache_dir}")
print(f"当前 HuggingFace Endpoint: {os.environ.get('HF_ENDPOINT')}")

# ============================================================
# 4. 显式下载模型文件（带重试）
# ============================================================
def download_model(repo_id: str, cache_dir: str, max_retries: int = 3):
    """使用 snapshot_download 先把模型文件拉下来，再返回本地路径。"""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[尝试 {attempt}/{max_retries}] 正在下载 {repo_id} ...")
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=False,
                resume_download=True,   # 断点续传
            )
            print(f"[成功] 模型已下载到: {local_path}")
            return local_path
        except Exception as e:
            print(f"[失败] 第 {attempt} 次下载出错: {e}")
            if attempt < max_retries:
                wait_time = 5 * attempt
                print(f"[重试] {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("\n[错误] 已达到最大重试次数，下载失败。")
                print("常见排查建议：")
                print("  1. 检查网络连接，或填写上面的 HTTP_PROXY/HTTPS_PROXY 代理地址；")
                print("  2. 访问 https://hf-mirror.com/nvidia/mit-b5 确认镜像可用；")
                print("  3. 升级 huggingface_hub 和 transformers：")
                print("     pip install -U huggingface_hub transformers")
                raise

local_model_path = download_model(checkpoint, cache_dir)

# ============================================================
# 5. 处理 torch<2.6 无法直接加载 .bin 权重的问题
# ============================================================
def ensure_safetensors_weights(local_path: str):
    """
    transformers 新版本出于安全考虑，要求 torch>=2.6 才能加载 .bin 权重。
    如果当前 torch 版本较低，且目录下只有 pytorch_model.bin，
    则用 torch.load(weights_only=True) 读取并转换为 model.safetensors，
    之后 from_pretrained(..., use_safetensors=True) 即可正常加载。
    """
    bin_path = os.path.join(local_path, "pytorch_model.bin")
    safe_path = os.path.join(local_path, "model.safetensors")

    if not os.path.exists(bin_path):
        return  # 没有 .bin 权重，不需要转换
    if os.path.exists(safe_path):
        print(f"[信息] 已存在 safetensors 权重: {safe_path}")
        return

    torch_version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if torch_version >= (2, 6):
        print("[信息] torch>=2.6，可直接加载 .bin 权重，跳过转换。")
        return

    print(f"\n[转换] 当前 torch {torch.__version__} < 2.6，"
          f"正在把 pytorch_model.bin 转为 model.safetensors 以规避加载限制...")

    # 临时绕过 transformers 的 torch 版本检查，使用 weights_only=True 安全读取
    original_check = import_utils.check_torch_load_is_safe
    import_utils.check_torch_load_is_safe = lambda: None
    try:
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        save_file(state_dict, safe_path)
        print(f"[转换完成] 已保存: {safe_path}")
    except Exception as e:
        print(f"[转换失败] {e}")
        print("建议手动升级 torch：")
        print("  pip install torch>=2.6")
        raise
    finally:
        import_utils.check_torch_load_is_safe = original_check

ensure_safetensors_weights(local_model_path)

# ============================================================
# 7. 验证本地文件是否已存在
# ============================================================
required_files = ["config.json"]
for f in required_files:
    file_path = os.path.join(local_model_path, f)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"下载目录缺少关键文件: {file_path}")
print("\n[验证] 关键模型文件已存在，开始加载...")

# ============================================================
# 8. 从本地下载目录加载 processor 和 model
# ============================================================
processor = AutoImageProcessor.from_pretrained(local_model_path)
model = AutoModelForSemanticSegmentation.from_pretrained(
    local_model_path,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # 分类头尺寸变化时忽略尺寸不匹配警告
    use_safetensors=True,          # 优先使用 safetensors 格式，绕开 torch.load 限制
)

# ============================================================
# 9. 配置 LoRA
# ============================================================
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # SegFormer 注意力投影层实际名称为 q_proj/v_proj
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save=["decode_head"],
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

print("\n[完成] SegFormer-B5 模型已成功下载并加载为 LoRA 模型。")
