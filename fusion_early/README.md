# 项目 3 — 隧道缺陷检测:三分支 → 跨模态融合 → 路径决策闭环

本目录是**隧道病害巡检「视觉感知 → 三维定位 → 路径决策」的主线**，承接上游仿真
（`inspection_robot`）生成的图像/点云数据，往下打通检测、特征融合、决策与离线闭环。
（这份 README 反映当前管线的最新状态 —— 已超出早期「只建数据集」的阶段。）

## 0. 上游依赖: `inspection_robot`
所有原始感知数据都来自 `~/workspace/inspection_robot`（ROS2 Humble + Gazebo Harmonic 两轮差速小车仿真）。
pro/3 **不自造数据**，只消费或复用其产物：

| inspection_robot 模块 | 对 pro/3 的作用 |
|---|---|
| `data_gen/`（`split_bing_*.py` + `build_*_from_sim.py` + 各类 `test_stems.txt`） | 生成三分支 sim 检测数据集、`pcdata` 图像池与属性图/位姿/标定 |
| `data_gen/output_*` → pro/3 `pcdata/` | 每帧 RGB 图 + `attrmap`（4 通道: 深度/反射/高度/遮罩）+ `poses` + `calibration.json`（K/T_cam_base） |
| `localization/img_px_to_world.py`（`TunnelGeometry`, `pixel_to_camera`, `camera_to_base`） | 像素框 → 里程/环号/时钟方位（定位链），以及车体系**距离/方位** |
| `localization/decision_decoder.py`（`DecisionDecoder`, `Target`） | 目标列表 → `FREE / APPROACH / INSPECT` 状态 + `(v, yaw)` 指令 |
| `localization/tunnel_params.py`（`TUNNEL`, `AVOID_*`, `DEC_*`） | 隧道几何 + 避障/决策阈值（差速车无法横移） |

> `fusion_decision_dump.py` 在服务器上直接 `import img_px_to_world`（纯 numpy，已 scp 到远端）；
> `run_fusion_closed_loop.py` 直接 `from localization.decision_decoder import DecisionDecoder`。

---

## 1. 三分支 3 类检测数据集

基于真实病害图（`~/workspace/pro/2 (1)/病害训练集+正常状态/病害训练集`）经 simulation 管线生成，
目标为**纯缺陷检测**。三个分支：土建 / 轨道 / 设备，各一套 train/val(仿真) + test(病集留测真图)。

### 1.1 类别映射（病集全局 → 分支局部）
| 分支 | 局部 id | 类别 | 病集全局 | 仿真全局 |
|------|---------|------|---------|----------|
| 土建 | 0 | 裂缝 | 6 | 0 |
| 土建 | 1 | 渗漏水 | 7 | 1 |
| 土建 | 2 | 管片掉块 | 5 | 6 |
| 轨道 | 0 | 断裂 | 1 | 2 |
| 轨道 | 1 | 丢失 | 2 | 3 |
| 轨道 | 2 | 松动歪斜 | 0 | 4 |
| 设备 | 0 | 异物入侵 | 3 | 5 |
| 设备 | 1 | 管线支架松脱 | 4 | 7 |

### 1.2 数据集划分（防泄漏设计）
- **train/val**：来自 Gazebo 仿真，4 seed(0-3) 合并，只保留含对应类框的帧，`val-frac 0.15`。
- **test**：病集**留测真图**，分层抽，**不参与生成**（裁贴图时用 `--skip-stems` 跳过，
  共 23 个留测 stem：土建10 + 轨道5 + 设备5）—— 训练贴图与测试真图**零重叠**，否则 recall 虚高。

```
轨道_3cls_sim/  0断裂/1丢失/2松动歪斜
土建_3cls_sim/  0裂缝/1渗漏水/2管片掉块
设备_2cls_sim/  0异物入侵/1管线支架松脱
  各含 images/{train,val}/*.jpg   labels/{train,val}/*.txt
       images/test/*.jpg           labels/test/*.txt   (病集留测真图, 原始stem)
       data.yaml                   # 指向本目录绝对路径
```

### 1.3 统计
| 数据集 | split | 图数 | 类框(0/1/2) |
|--------|-------|------|------------|
| 轨道_3cls_sim | train | 534 | 断裂9929 / 丢失10378 / 松动9833 |
| 轨道_3cls_sim | val   | 94  | 1753 / 1829 / 1781 |
| 轨道_3cls_sim | test  | 5   | 3 / 1 / 2 |
| 土建_3cls_sim | train | 534 | 裂缝3199 / 渗漏3928 / 掉块4008 |
| 土建_3cls_sim | val   | 94  | 579 / 694 / 765 |
| 土建_3cls_sim | test  | 10  | 11 / 4 / 8 |
| 设备_2cls_sim | train | 534 | 异物4366 / 支架3220 |
| 设备_2cls_sim | val   | 94  | 759 / 583 |
| 设备_2cls_sim | test  | 5   | 4 / 1 |

---

## 2. 管线四段

```
① 三分支检测     train_轨道/土建/设备_*.py → best.pt
② 摘编码器       extract_yolo11_encoder.py → encoders/yolo11l_*_backbone|neck_encoder.pt
③ 跨模态融合     fusion.py (Self-Attn → Cross-Attn 图像↔点云 → 门控+FFN, 每尺度)
                    点云编码器: pcl_enc.py + train_pcl_detect.py (检测式预训, 4通道属性图)
                    训练融合:   train_fusion.py / train_fusion_ddp.py → runs/fusion_ddp/
④ 决策+闭环     fusion_decision_dump.py → detections_big_s3.json (每帧 cls/conf/距离/方位)
                    → probe_decode.py / run_fusion_closed_loop.py (DecisionDecoder → cmd_vel)
```

### ① 分支检测
`train_轨道_3cls.py` / `train_土建_3cls.py` / `train_设备_2cls.py` —— 无参即跑。
**关键**：用 `--imgsz 1280 --test-imgsz 1280`（仿真 1280×720，病集百万像素）。
早前只改 `--test-imgsz 1280` 差点火候 —— 训练仍是 640，评测 1280，属尺度错配：
**单改训练 imgsz 到 1280** 后 轨道 0.53→0.97、土建 0.02→0.32、设备 0.88。

### ② 摘编码器
三分支各自 best 权重，`extract_yolo11_encoder.py` 摘出 backbone/neck 编码器存 `encoders/`，
与 `det_encoders_runtime.py` 对齐输入约定（单帧 RGB → `{P3,P4,P5}` 多尺度特征）。

### ③ 跨模态融合（`fusion.py`）
四路输入 → 每个尺度一个 block：
- **① Self-Attn**（Q=K=V=图像 token，模态内/跨分支上下文）
- **② Cross-Attn**（Q=图像，K/V=点云，按外参空间对齐）—— `pcl_feats=None` 时**跳过**（图像占位先跑）
- **③ 门控(SE) + FFN + LayerNorm**

输出 F2–F5 特征金字塔 → 送「检测 / 路径决策」双头。token 超 `max_tokens` 自动降采样防 OOM。
- 点云支：`pcl_enc.py`（`forward_maps()` 吃 4 通道 attrmap）+ `train_pcl_detect.py` **检测式预训**
  （复用 `v8DetectionLoss`，TAL+DFL；自建数据，不用 COCO RGB，因 4 通道属性图 ≠ 3 通道 + 模态迁移差）。
- 训练：`train_fusion.py`（CPU/单卡）/ `train_fusion_ddp.py`（DDP）→ `runs/fusion_ddp/`。
- ⚠️ checkpoint 冻结坑：best 权重载入后**全层 `requires_grad=False`**（含 Detect 头），
  融合微调必须**显式解冻 Detect 头**；`v8DetectionLoss` 的 bbox 要 xywh 归一化到 [0,1]。

### ④ 决策 + 闭环
- `fusion_decision_dump.py`：对 `output_big_s3` 逐帧跑融合权重 `fusion_土建_1280_neg.pt`，
  解码框 → attrmap depth 取框内中值 → `pixel_to_camera→camera_to_base` → 车体系**距离/方位** →
  `detections_big_s3.json`（每帧 `{cls, conf, distance, azimuth_deg, bbox, grid_pts}`）。
  标定取自 `output_rail_s1`（同隧道/同传感器）；**无需点云**。
- `probe_decode.py`：纯离线喂 `decision_decoder.DecisionDecoder`，不依赖 GPU/fork。
- `run_fusion_closed_loop.py`：把 dump 与**世界系锚定**，用模拟小车积分 `cmd_vel` + 每步按最近录制帧
  重新相对化距离/方位 → `DecisionDecoder` → `(v, yaw)` → 运动学前进。**①→⑧ 第一次串成闭环**。
  `DecisionDecoder` 状态机：`FREE`（巡航）/ `APPROACH`（减速斜坡+轻转向居中，`DEC_V_CREEP` 爬行下限
  **防止 d0→STOP 时死锁**）/ `INSPECT`（`d0≤AVOID_STOP_DIST` 停车检查，带 `AVOID_RESUME_HYST` 迟滞）。
  **检测到异物也是「检查」而非「绕行避让」**（无 STEER 分支 = 天然不避让）。
- 复用阈值：`tunnel_params.py` 的 `AVOID_BODY_FRONT/AVOID_V_CRUISE/AVOID_STOP_DIST/AZ_ABEAM_DEG/DEC_*`。

---

## 3. 评测脚本与当前结论

| 脚本 | 作用 |
|---|---|
| `eval_3cls_sim.py` / `test_3cls_sim.py` | 分支在 sim / 病集上的检测评测 |
| `eval_fusion_recall.py` / `baseline_fusion_recall.py` / `report_fusion_vs_baseline.py` | 融合 vs 单支召回对比 |
| `search_conf.py` | 阈值/conf 搜索 |

**当前已知结论**
- 融合召回：土建@1280 正确 R=0.253 / 短差 0.747，根因是**阈值伪影**（阈值把很多短差标成漏检）。
- 真实帧 DecisionDecoder 输出**全 APPROACH、无 INSPECT** —— 逼近真实巡检行为仍未闭环。
- 本地闭环为离线 numpy（不含 GPU），可本地复现。

---

## 4. 目录索引
```
轨道/土建/设备_{3|2}cls_{sim,comp,real[,aug]}   # 分支数据集
pcdata/              # sim 图像池 + attrmap/poses/calibration (来自 inspection_robot/data_gen)
encoders/            # 分支摘出的 backbone/neck 编码器
runs/                # 评测/融合/点云预训/闭环 产物
weights/             # 训练权重
架构图/               # 整体架构图 img.png
```

## 5. 运行环境
- 本地：`inspection_robot/.venv`（torch + cu130 + CUDA），`python -u <script>.py`。
- 服务器/远端：`yolo13l_env`（GPU + weights + pcdata），跑 ④ 的 dump 与融合训练。
- 纯 numpy 部分（`img_px_to_world`、`decision_decoder`）无需 GPU，可本地直接跑。
