"""
融合子系统端到端链路测试 (真实样本 output_v7/000000)
================================================================
接口先行阶段的验证目标: **前向 shape 正确 + 反向梯度有限**, 不训练。
将来换真模型权重后跑同样的测试, 保证契约 shape 没被破坏。

覆盖 (对照冻结契约):
  1. ImageEncoder -> P3(1,256,80,80)/P4(1,256,40,40)/P5(1,256,20,20)/tokens(1,8400,256)
     全有限; 冻结主干无梯度、投影层有梯度; 两次 forward 确定性一致。
  2. PointCloudEncoder -> feat(1,256,45,80)/tokens(1,3600,256); depth 归一化∈[0,1]。
  3. pose JSON -> 12 维 /pose_state(四元数->欧拉三角) -> PoseEncoder -> (1,128)。
  4. FusionPipeline 端到端 -> tokens(1,12000,256)/image_tokens(1,8400,256)/
     point_tokens(1,3600,256)/image_scales 三尺度/pose_embed(1,128)。
  5. backward: tokens.pow(2).mean().backward() -> 所有 requires_grad 参数梯度有限。
  6. GPU 显存占用可接受(前向+反向, 无卡跳过)。

运行: `.venv/bin/python fusion/test_fusion.py`
"""

import math
import os
import sys

import numpy as np
import torch

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from fusion.config import (  # noqa: E402
    D_MODEL, N_IMG, N_IMG_SPLIT, N_PC, NECK_HW, NECK_ORDER, N_TOKENS, OUTPUT_V7,
    POSE_DIM, POSE_EMB, YOLO_SIZE,
)
from fusion.fusion import FusionPipeline  # noqa: E402


def _sample_paths():
    # 默认 output_v7; 换数据集用环境变量覆盖(如 FUSION_DATA=data_gen/output_data1)
    root = os.environ.get('FUSION_DATA', OUTPUT_V7)
    return {
        'img': os.path.join(root, 'images', '000000.jpg'),
        'attr': os.path.join(root, 'attrmaps', '000000.npz'),
        'pose': os.path.join(root, 'poses', '000000.json'),
    }


def load_sample():
    import cv2
    import json
    p = _sample_paths()
    img_bgr = cv2.imread(p['img'])                     # (720,1280,3) BGR uint8
    assert img_bgr is not None, f"读不到图像: {p['img']}"
    a = np.load(p['attr'])
    attrmap = np.stack([a['depth'], a['reflec'], a['height']]).astype(np.float32)  # (3,720,1280)
    pose = json.load(open(p['pose']))
    return [img_bgr], attrmap, pose


def pose_json_to_state(p):
    """poses/{i}.json(odom) -> 12 维 /pose_state, 与 pose_state_tunnel.csv 字段序一致:
    x,y,z,sin_roll,cos_roll,sin_pitch,cos_pitch,sin_yaw,cos_yaw,vx,vy,wz。
    JSON 无速度字段 -> vx/vy/wz=0(隧道巡检近似静止)。"""
    x, y, z = p['x'], p['y'], p['z']
    qx, qy, qz, qw = p['qx'], p['qy'], p['qz'], p['qw']
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.array(
        [x, y, z, math.sin(roll), math.cos(roll), math.sin(pitch),
         math.cos(pitch), math.sin(yaw), math.cos(yaw), 0.0, 0.0, 0.0],
        dtype=np.float32)


_PASS = 0
_FAIL = []


def check(name, cond, detail=''):
    global _PASS
    if cond:
        _PASS += 1
        print(f"  [PASS] {name} {detail}")
    else:
        _FAIL.append(name)
        print(f"  [FAIL] {name} {detail}")


def main():
    print("=" * 72)
    print("融合子系统端到端测试 —— 真实样本 output_v7/000000")
    print("=" * 72)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device = {device}\n")

    # ---- 0. 装载真实样本 ----
    img_bgr_list, attrmap, pose = load_sample()
    pose_state = pose_json_to_state(pose)
    print(f"样本: 图像 {list(img_bgr_list[0].shape)} BGR | attrmap {tuple(attrmap.shape)} "
          f"| pose_state {tuple(pose_state.shape)}")
    print(f"pose_state[0:3] = {pose_state[0:3]}  (x,y,z)")

    # ---- 1. ImageEncoder ----
    print("\n[1] ImageEncoder (YOLO11l 主干 + 投影)")
    from fusion.image_encoder import ImageEncoder
    enc_img = ImageEncoder(device=device)
    x = ImageEncoder.preprocess(img_bgr_list, device=device)
    check("preprocess 尺寸", tuple(x.shape) == (1, 3, YOLO_SIZE, YOLO_SIZE),
          f"-> {tuple(x.shape)}")
    out = enc_img(x)
    for name in NECK_ORDER:
        check(f"空间图 {name}",
              tuple(out[name].shape) == (1, D_MODEL, *NECK_HW[name]),
              f"-> {tuple(out[name].shape)}")
    check("tokens (1,8400,256)", tuple(out['tokens'].shape) == (1, N_IMG, D_MODEL),
          f"-> {tuple(out['tokens'].shape)}")
    check("特征全有限", bool(torch.isfinite(out['tokens']).all()), "")
    img_grad = [p.grad for p in enc_img.parameters() if p.requires_grad]
    back_grad = [p.grad for p in enc_img.model.parameters()]
    check("主干冻结无梯度", all(g is None for g in back_grad), "")
    check("投影层可训练", any(p.requires_grad for p in enc_img.scale_proj.parameters()), "")
    out2 = enc_img(x)
    check("两次 forward 确定性一致",
          bool(torch.equal(out['tokens'], out2['tokens'])), "")

    # ---- 2. PointCloudEncoder ----
    print("\n[2] PointCloudEncoder (attrmap CNN)")
    from fusion.pointcloud_encoder import PointCloudEncoder
    enc_pc = PointCloudEncoder(device=device)
    pc_out = enc_pc(attrmap)
    check("feat (1,256,45,80)", tuple(pc_out['feat'].shape) == (1, D_MODEL, 45, 80),
          f"-> {tuple(pc_out['feat'].shape)}")
    check("tokens (1,3600,256)", tuple(pc_out['tokens'].shape) == (1, N_PC, D_MODEL),
          f"-> {tuple(pc_out['tokens'].shape)}")
    dn = enc_pc._normalize(torch.from_numpy(attrmap).unsqueeze(0)).cpu().numpy()
    check("depth 归一化∈[0,1]", bool((dn[0, 0] >= 0).all() and (dn[0, 0] <= 1).all()),
          f"depth max={dn[0,0].max():.3f}")
    check("点云特征全有限", bool(torch.isfinite(pc_out['tokens']).all()), "")

    # ---- 3. PoseEncoder ----
    print("\n[3] PoseEncoder (12 -> 128)")
    from data_gen.pose_encoder import PoseEncoder
    pose_enc = PoseEncoder()
    pe = pose_enc(pose_state)                          # (128,)
    check("pose_state 12 维", pose_state.shape == (POSE_DIM,), "")
    check("pose_embed (128,)", pe.shape == (POSE_EMB,), f"-> {tuple(pe.shape)}")
    check("pose_embed 全有限", bool(torch.isfinite(pe).all()), "")

    # ---- 4. FusionPipeline 端到端 ----
    print("\n[4] FusionPipeline 端到端 (图像+点云+位姿 -> 交叉注意力)")
    pipe = FusionPipeline(device=device)
    fused = pipe(img_bgr_list, attrmap, pose_state)
    check("tokens (1,12000,256)", tuple(fused.tokens.shape) == (1, N_TOKENS, D_MODEL),
          f"-> {tuple(fused.tokens.shape)}")
    check("image_tokens (1,8400,256)", tuple(fused.image_tokens.shape) == (1, N_IMG, D_MODEL),
          f"-> {tuple(fused.image_tokens.shape)}")
    check("point_tokens (1,3600,256)", tuple(fused.point_tokens.shape) == (1, N_PC, D_MODEL),
          f"-> {tuple(fused.point_tokens.shape)}")
    for name in NECK_ORDER:
        check(f"image_scales[{name}]", tuple(fused.image_scales[name].shape) == (1, D_MODEL, *NECK_HW[name]),
              f"-> {tuple(fused.image_scales[name].shape)}")
    check("pose_embed (1,128)", tuple(fused.pose_embed.shape) == (1, POSE_EMB),
          f"-> {tuple(fused.pose_embed.shape)}")
    check("融合输出全有限", bool(torch.isfinite(fused.tokens).all()), "")

    # ---- 5. 反向梯度有限 ----
    print("\n[5] 反向梯度")
    loss = fused.tokens.pow(2).mean()
    loss.backward()
    bad = 0
    for name, p in pipe.named_parameters():
        if p.requires_grad:
            if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                bad += 1
                print(f"  !! {name}: grad={'None' if p.grad is None else '非有限'}")
    check("全部可训参数梯度有限", bad == 0, f"(可训参数 {sum(1 for p in pipe.parameters() if p.requires_grad)} 个)")
    # 梯度确实流向各编码器(不只融合): 抽验投影层与点云 CNN
    gr = {k: v for k, v in pipe.named_parameters()}
    proj_grads = [p.grad is not None for k, p in gr.items() if 'scale_proj' in k]
    pc_grads = [p.grad is not None for k, p in gr.items() if 'point_encoder.net' in k]
    check("图像投影层有梯度", all(proj_grads), f"({sum(proj_grads)}/{len(proj_grads)})")
    check("点云 CNN 有梯度", all(pc_grads), f"({sum(pc_grads)}/{len(pc_grads)})")

    # ---- 6. 显存 ----
    if torch.cuda.is_available():
        mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        check("显存峰值 < 4GB (前向+反向)", mb < 4 * 1024, f"({mb:.0f} MiB)")
    else:
        print("  [SKIP] 显存检查(无 GPU)")

    print("\n" + "=" * 72)
    if _FAIL:
        print(f"共 {_PASS + len(_FAIL)} 项, {len(_FAIL)} 项失败: {_FAIL}")
        sys.exit(1)
    print(f"全部 {_PASS} 项通过 ✔  融合契约未被破坏")
    print("=" * 72)


if __name__ == '__main__':
    main()
