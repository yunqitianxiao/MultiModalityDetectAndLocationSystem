# 数据目录约定

仓库不包含实际数据集。将配对数据放在项目根目录的 `data/` 下：

```text
data/
└── output_crack_big/
    ├── images/       # 1280x720 RGB，文件名如 000000.jpg
    ├── labels/       # YOLO 格式，单类 crack
    ├── attrmaps/     # 000000.npz，depth / reflec / height
    └── poses/        # 000000.json，车辆位姿
```

数据生成脚本位于 `inspection_robot/data_gen/`。常用入口：

```bash
python inspection_robot/data_gen/build_tujian_3cls_from_sim.py --help
python inspection_robot/data_gen/build_track_3cls_from_sim.py --help
python inspection_robot/data_gen/build_device_2cls_from_sim.py --help
python inspection_robot/data_gen/extract_single_class.py --help
```

如果沿用旧目录结构，可通过 `IR_ROOT` 指向 `inspection_robot/`，或用各脚本的 `--data` 参数显式传入数据目录。
