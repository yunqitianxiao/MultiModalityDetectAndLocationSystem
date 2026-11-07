# 空间语义映射与精准定位（定位链）

> 赛题 CS-202613（中铁十一局 · 地铁多领域病害综合智能巡检系统）"核心算法（二）2"的落地实现：
> **把 2D 检测框变成隧道里的物理坐标——类型 + 里程 / 环号 / 时钟方位 + 3D坐标 + 尺寸。**
> 全程是**标定/几何驱动**，不学任何参数，因此**不需要成对的图片-点云训练集**。

---

## 一、怎么回事（原理）

### 1. 要解决的问题
检测器（YOLO）只告诉你"像素 (u,v,w,h) 处有渗漏水"。赛题要你回答：**它在隧道里哪个位置、什么类型**。中间隔着一整条坐标变换链。

### 2. 坐标变换链（本模块全部内容）
```
像素 (u,v) + 深度d            ← 深度 d 来自雷达点云（把雷达点投进图像，取检测框内中值）
   │  相机内参 K（针孔逆投影）
   ▼
相机系 X_cam                  ← 标准针孔模型：z 沿光轴，即深度
   │  外参 T_cam_base（相机光心→车体，来自联合标定/URDF）
   ▼
车体系 X_base
   │  位姿 T_world_base（IMU+里程计 / FAST-LIO2 提供）
   ▼
世界系 X_world
   │  隧道几何（轴线+环宽+起点里程）
   ▼
里程(m) / 环号 / 时钟方位 / 3D坐标 / 径向距离
```
反向映射（"3D属性图→2D"）同样在这套几何里：
**雷达点云 → 投影到图像 → 深度图 / 反射率图 / 高度图**，可作为暗光下的 2D 检测辅助通道，也可做 BEV 高度图用于限界/避障。

### 3. 三个关键约定
- **时钟方位**：面向行车方向（里程递增方向）看隧道截面，**12点=正上方，3点=右侧**，顺时针一圈 12 格（30°/格）。公式：`angle = atan2(r·e_right, r·e_up)`。
- **环号**：`环号 = floor((里程−起点里程) / 环宽) + 1`，1 起始。
- **深度来源**：检测框内的雷达点取**中值**（病害是一个面区域，单点不稳）；框内没点时自动在框中心附近找最近点兜底。

---

## 二、怎么用（三步）

### 第 0 步：装依赖
```bash
pip install numpy            # 核心只依赖 numpy
# scipy / matplotlib 可选：matplotlib 用于出图
```

### 第 1 步：改参数（只有这一处跟现场相关）
`传感器参数.py`：
- `CAMERA`：分辨率、视场角、安装位姿（当前值与 `inspection_robot/inspection_robot.urdf.xacro` 一致）
- `TUNNEL`：**隧道轴线起点 / 方向 / 起点里程 / 环宽** —— 换隧道必须改这三个

### 第 2 步：跑通闭环验证（不需要 ROS、不需要真数据）
```bash
python3 仿真验证_定位链.py
```
- 控制台打印病害记录 JSON（里程/环号/时钟方位/3D/尺寸/置信度）
- 末尾 5 项 PASS/FAIL 断言，全过说明链数学正确
- 生成 `docs/定位链demo输出.png`（相机视角 + 3D隧道 + 属性图，可直接进报告）

### 第 3 步：接到真车/仿真
把 `ros_定位链节点.py` 的 YOLO 模型换成你的三领域模型，`source setup.sh` 后：
```bash
ros2 launch inspection_robot inspection_robot.launch.py
python3 ros_定位链节点.py --model /path/to/best.pt
```
节点订阅 `/camera /lidar/points /odom`，每帧打印病害记录 JSON。

### 代码里怎么调用（核心只有一行）
```python
from img_px_to_world import DetectionLocalizer, TunnelGeometry
from 传感器参数 import K, T_CAM_BASE, TUNNEL, IMG_SIZE

loc = DetectionLocalizer(K=K, T_cam_base=T_CAM_BASE,
                         tunnel=TunnelGeometry(**TUNNEL), img_size=IMG_SIZE)
record = loc.locate(detections, lidar_points_base, T_world_base)   # -> 病害记录 dict
```
`detections = [{"class_name": ..., "bbox": [x1,y1,x2,y2], "confidence": ...}, ...]`

---

## 三、文件清单

| 文件 | 作用 |
|---|---|
| `img_px_to_world.py` | 核心库：内参/外参/位姿/逆投影/点云深度/隧道几何/属性图/定位器 |
| `传感器参数.py` | 与 xacro 对齐的相机-雷达-IMU 参数 + 隧道几何（唯一要改的地方） |
| `仿真验证_定位链.py` | 合成隧道闭环 demo，断言 + 出图 |
| `ros_定位链节点.py` | ROS2 集成模板（订阅三路传感器→检测→定位→JSON） |
| `docs/架构图.png` | 系统架构图（graphviz，报告用） |
| `docs/时序图.png` | 定位链运行时序图（graphviz，报告用） |
| `docs/甘特图.png` | 4 周任务排期（matplotlib，报告用） |
| `docs/定位链demo输出.png` | 闭环验证四联图（实证材料） |

---

## 四、已知简化与改进项（报告里可写，体现深度）
1. **直隧道模型**：弯隧道/缓和曲线需把轴线换成 polyline/曲率，改 `TunnelGeometry` 的 `world_to_tunnel` 即可，接口不变。
2. **尺寸近似正对平面**：`bbox_size_meters` 用针孔公式 `w=Δu/fx·d`，斜视时偏小；精确做法是把框四角逆投影、在点云上拟合病害平面再量尺寸（已提供框内点云锚点，可直接扩展）。
3. **深度用中值**：框内点云可升级为 RANSAC 平面拟合取平面深度，抗噪更强。
4. **时间同步**：真车上相机/雷达/里程计各帧时间戳需插值对齐（仿真里无此问题，天然同步）。
5. **弱纹理暗光**：把反射率/深度图作为检测模型额外输入通道（多通道 YOLO），是本方案的融合增强路线。
