import os
import random
import pickle
from sklearn.model_selection import train_test_split

# ============================================
# 1. 配置路径 (渗漏水: 549 图/549 掩膜同名, 正常负样本 土建_正常)
# ============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR = os.path.join(PROJECT_ROOT, "data")

NORMAL_DIR = os.path.join(BASE_DIR, "土建_正常")
LEAKAGE_DIR = os.path.join(BASE_DIR, "土建_渗漏水")
MASK_DIR = os.path.join(BASE_DIR, "土建_渗漏水_手动标注的掩膜")

SPLIT_DIR = os.path.join(BASE_DIR, "dataset_split")
os.makedirs(SPLIT_DIR, exist_ok=True)

# ============================================
# 2. 收集图片和掩膜
# ============================================
def get_image_paths(directory):
    paths = []
    for f in os.listdir(directory):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            paths.append(os.path.join(directory, f))
    return paths

normal_images = get_image_paths(NORMAL_DIR)
leakage_images = get_image_paths(LEAKAGE_DIR)
mask_images = get_image_paths(MASK_DIR)

print(f"正常图片: {len(normal_images)} 张")
print(f"渗漏水图片: {len(leakage_images)} 张")
print(f"掩膜图片: {len(mask_images)} 张")

# 配对渗漏水图片和掩膜 (同名, 源图 .jpg / 掩膜 .png)
mask_index = {os.path.splitext(os.path.basename(m))[0]: m for m in mask_images}

leakage_pairs = []  # (image_path, mask_path)
for img_path in leakage_images:
    basename = os.path.splitext(os.path.basename(img_path))[0]
    if basename in mask_index:
        leakage_pairs.append((img_path, mask_index[basename]))
    else:
        print(f"  [跳过] 无掩膜: {os.path.basename(img_path)}")

print(f"配对的渗漏水-掩膜: {len(leakage_pairs)} 对")

# ============================================
# 3. 划分: 随机 10% 测试 + 其余 85/15 训练/验证 (与 土建三病害掩膜转YOLO框.py 比例一致)
#    正常图: train 70% / val 15% / test 15% (仅作背景负样本)
# ============================================
random.seed(42)
random.shuffle(leakage_pairs)
n_test = max(1, round(len(leakage_pairs) * 0.10))
test = leakage_pairs[:n_test]
rest = leakage_pairs[n_test:]
n_val = max(1, round(len(rest) * 0.15))
val = rest[:n_val]
train = rest[n_val:]

normal_train, normal_temp = train_test_split(normal_images, test_size=0.3, random_state=42)
normal_val, normal_test = train_test_split(normal_temp, test_size=0.5, random_state=42)

print("\n划分结果:")
print(f"  渗漏水 - 训练: {len(train)}, 验证: {len(val)}, 测试: {len(test)}  (共 {len(leakage_pairs)})")
print(f"  正常   - 训练: {len(normal_train)}, 验证: {len(normal_val)}, 测试: {len(normal_test)}")

# ============================================
# 4. 保存划分结果 (供 Lora微调_分割模型.py 读取)
# ============================================
split_data = {
    'leakage_train': train,
    'leakage_val': val,
    'leakage_test': test,
    'normal_train': normal_train,
    'normal_val': normal_val,
    'normal_test': normal_test,
}

with open(os.path.join(SPLIT_DIR, 'leakage_split.pkl'), 'wb') as f:
    pickle.dump(split_data, f)

print(f"\n数据划分已保存到: {SPLIT_DIR}/leakage_split.pkl")
