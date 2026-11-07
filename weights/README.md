# 权重目录约定

权重文件体积较大，不纳入 Git。按以下路径放置后可复现实验：

```text
weights/
├── yolo11l.pt                 # 2D 检测分支主干/微调权重
├── assist_crack.pt            # 图像 + 点云融合权重
├── assist_crack_qfl.pt        # Range-QFL 变体
└── pose_mlp_tunnel.pt         # 12 维位姿编码器
```

部分脚本也支持环境变量覆盖：

- `YOLO11L_CKPT`
- `POSE_CKPT`
- `IMG_CKPT`
- `IR_ROOT`
