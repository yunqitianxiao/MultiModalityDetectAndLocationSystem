import os
import random
from sklearn.model_selection import train_test_split

# ============================================
# 1. 配置路径
# ============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BASE_DIR = os.path.join(PROJECT_ROOT, "data")

NORMAL_DIR = os.path.join(BASE_DIR, "土建_正常")
CRACK_DIR = os.path.join(BASE_DIR, "土建_裂缝")
MASK_DIR = os.path.join(BASE_DIR, "土建_裂缝_手动标注的掩膜")

# 输出目录（用于存放划分后的数据）
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
crack_images = get_image_paths(CRACK_DIR)
mask_images = get_image_paths(MASK_DIR)

print(f"正常图片: {len(normal_images)} 张")
print(f"裂缝图片: {len(crack_images)} 张")
print(f"掩膜图片: {len(mask_images)} 张")

# 配对裂缝图片和掩膜（精确匹配，支持 同名 / {base}_mask / mask_{base} 三种命名）
# 注意：不能用"basename in m_basename"子串匹配，否则 0602 会错误匹配到 06021_mask 这类掩膜
mask_index = {}
for m in mask_images:
    m_basename = os.path.splitext(os.path.basename(m))[0]
    mask_index[m_basename] = m

crack_pairs = []  # (image_path, mask_path)
for img_path in crack_images:
    basename = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = None
    for candidate in (basename, basename + "_mask", "mask_" + basename):
        if candidate in mask_index:
            mask_path = mask_index[candidate]
            break
    if mask_path is not None:
        crack_pairs.append((img_path, mask_path))
    else:
        print(f"  [跳过] 无掩膜: {os.path.basename(img_path)}")

print(f"配对的裂缝-掩膜: {len(crack_pairs)} 对")

# ============================================
# 3. 划分数据集：隧道域 11 张按 5/3/3 分散，公共裂缝图补给训练/验证
#    - 隧道域图(0602-0612, 含掩膜的11张)掩膜质量差(224x224、灰度~110, 分割脚本 >127 会判为背景),
#      不宜全放测试集当基准, 改为 训练5/验证3/测试3
#    - 公共数据集裂缝图抽 TEST_PUBLIC_SIZE 张进测试集做稳定基准，其余给训练/验证
#    - 想要纯公共测试集：把 TEST_PUBLIC_SIZE 改成 0
# ============================================
random.seed(42)

TUNNEL_PREFIX = "06"     # 隧道域图片文件名前缀(0601-0612)
TEST_PUBLIC_SIZE = 24    # 测试集保留的公共裂缝图数量

def is_tunnel(pair):
    return os.path.basename(pair[0]).startswith(TUNNEL_PREFIX)

tunnel_crack = [p for p in crack_pairs if is_tunnel(p)]
public_crack = [p for p in crack_pairs if not is_tunnel(p)]

# 隧道域 12 张: 5 训练 / 1 验证 / 6 测试
# (验证集只留 1 张隧道域, 其余 2 张挪进测试集, 让真实域召回评估样本更多更稳)
random.shuffle(tunnel_crack)
tunnel_train = tunnel_crack[:5]
tunnel_val = tunnel_crack[5:6]
tunnel_test = tunnel_crack[6:]

# 公共裂缝图：少量进测试集，其余按 85/15 分训练/验证
random.shuffle(public_crack)
public_test = public_crack[:TEST_PUBLIC_SIZE]
public_rest = public_crack[TEST_PUBLIC_SIZE:]
public_train, public_val = train_test_split(public_rest, test_size=0.15, random_state=42)

crack_train = tunnel_train + public_train
crack_val = tunnel_val + public_val
crack_test = tunnel_test + public_test

# 正常数据划分（只用于训练集，作为背景类）
normal_train, normal_temp = train_test_split(normal_images, test_size=0.3, random_state=42)
normal_val, normal_test = train_test_split(normal_temp, test_size=0.5, random_state=42)

print("\n划分结果:")
print(f"  裂缝 - 训练: {len(crack_train)}, 验证: {len(crack_val)}, 测试: {len(crack_test)}")
print(f"    隧道域: 训练 {len(tunnel_train)} 张 {[os.path.basename(p[0]) for p in tunnel_train]}")
print(f"            验证 {len(tunnel_val)} 张 {[os.path.basename(p[0]) for p in tunnel_val]}")
print(f"            测试 {len(tunnel_test)} 张 {[os.path.basename(p[0]) for p in tunnel_test]}")
print(f"    公共裂缝: 训练 {len(public_train)}, 验证 {len(public_val)}, 测试 {len(public_test)}")
print(f"  正常 - 训练: {len(normal_train)}, 验证: {len(normal_val)}, 测试: {len(normal_test)}")

# ============================================
# 4. 保存划分结果（用于后续训练）
# ============================================
import pickle

split_data = {
    'crack_train': crack_train,
    'crack_val': crack_val,
    'crack_test': crack_test,
    'normal_train': normal_train,
    'normal_val': normal_val,
    'normal_test': normal_test,
}

with open(os.path.join(SPLIT_DIR, 'dataset_split.pkl'), 'wb') as f:
    pickle.dump(split_data, f)

print(f"\n数据划分已保存到: {SPLIT_DIR}/dataset_split.pkl")