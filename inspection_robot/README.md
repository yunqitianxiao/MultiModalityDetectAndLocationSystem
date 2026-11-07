# inspection_robot — 隧道病害巡检两轮差速小车仿真

基于 **ROS 2 Humble + Gazebo Harmonic** 的两轮差速巡检小车仿真包。用于隧道病害巡检仿真，后续接入：

1. **FAST-LIO2 SLAM** —— 依赖 16 线激光雷达点云（`/lidar/points`）+ IMU（`/imu`）+ 里程计（`/odom`）
2. **图像-点云多模态病害检测** —— 依赖 RGB 相机（`/camera/image_raw`）+ 点云

**架构**：`base_link` 为根连杆，2 驱动轮（continuous，被 `DiffDrive` 驱动）+ 2 被动万向轮（fixed）；车体固定安装 RGB 相机 / IMU / 16 线激光雷达（全部 fixed 关节）。无 GPS。

---

## 1. 目录结构

```
inspection_robot/
├── package.xml
├── CMakeLists.txt
├── urdf/
│   └── inspection_robot.urdf.xacro   # 机器人模型(参数化, 传感器位姿集中可调)
├── worlds/
│   └── inspection_tunnel.world       # 隧道走廊世界(墙+顶+特征物)
├── config/
│   ├── ros_gz_bridge.yaml            # gz<->ROS 话题桥接表
│   └── inspection_robot.rviz         # RViz 可视化配置
├── scripts/
│   └── odom_tf_broadcaster.py        # odom->base_link 动态 TF 广播
├── launch/
│   └── inspection_robot.launch.py    # 一键启动
├── docs/
│   └── 导航与序列技术教程.md          # 技术入门索引: ROS2导航/标定/位姿嵌入, 每条教程对应项目代码
└── README.md
```

---

## 2. 环境要求与安装

### 2.1 系统与 ROS2

- Ubuntu 22.04 + **ROS 2 Humble**
- **Gazebo Harmonic**（gz-sim 8.x）

### 2.2 安装 Gazebo Harmonic

按官方文档：<https://gazebosim.org/docs/harmonic/install_ubuntu>

```bash
sudo apt-get update
sudo apt-get install lsb-release wget gnupg
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo-ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list
sudo apt-get update
sudo apt-get install gz-harmonic
```

验证：`gz sim --version`。

### 2.3 安装 ros_gz（Humble + Harmonic 组合）

> ⚠️ **关键**：ROS2 Humble 官方二进制里的 `ros-humble-ros-gz` 是为 **Gazebo Fortress** 编译的。
> 要和 **Harmonic** 搭配，需要**从源码编译 ros_gz**（这是 Humble+Harmonic 社区最常用的做法）。

```bash
sudo apt-get install ros-humble-ament-cmake ros-humble-rclcpp ros-humble-std-msgs \
  ros-humble-geometry-msgs ros-humble-sensor-msgs ros-humble-nav-msgs \
  ros-humble-tf2-ros ros-humble-robot-state-publisher ros-humble-xacro \
  ros-humble-rviz2 python3-vcstool

mkdir -p ~/ros_gz_ws/src && cd ~/ros_gz_ws
vcs import src --input https://raw.githubusercontent.com/gazebosim/ros_gz/humble/ros_gz_humble.repos
# 若 humble 分支依赖为 Fortress, 可改用 master 分支(支持 Garden/Harmonic):
# vcs import src --input https://raw.githubusercontent.com/gazebosim/ros_gz/master/ros_gz.repos
rosdep update && rosdep install --from-paths src -iry --ignore-src
colcon build --symlink-install
echo "source ~/ros_gz_ws/install/setup.bash" >> ~/.bashrc
```

> 若 `rosdep` 报缺少包，确认已安装 `ros-humble-ros-gz` 相关依赖或用 `git clone` 手动拉取
> `ros_gz_bridge`、`ros_gz_sim` 等仓库到 src 下。

> **✅ 本机已配置完成（2026-08-17）**：ros_gz 已从源码（`gazebosim/ros_gz` 单仓库 humble 分支）
> 编译到 `/home/ylf/workspace/ros_gz_ws/install`，用 `GZ_VERSION=harmonic` 构建，链接 gz-msgs10/gz-transport13。
> 使用 `source setup.sh` 即可一次性加载 Humble + ros_gz + 本项目三个环境。
> 注意：ros_gz 单仓库名称为 `gazebosim/ros_gz`（旧 `ros_gz_bridge` 等独立仓库已并入）；编译 ros_gz_interfaces 时若报
> `canonicalize_version() got an unexpected keyword argument 'strip_trailing_zero'`，是 packaging 版本过旧（<22），
> 执行 `python3 -m pip install --user "packaging==23.2"` 后重编即可。

### 2.5 本机验证状态（2026-08-17）

已验证可发布话题（ROS 侧实测频率）：**相机 30Hz、camera_info 30Hz、IMU 200Hz、16线点云 9Hz**，TF 树完整。

> ⚠️ **本机内存限制**：开发机仅 14GB 内存且被 IDE/浏览器占满、swap 打满时，gz 物理引擎与 DiffDrive 里程计会停滞（连 gz 官方示例世界都停）。
> 相机/雷达/IMU 传感器不受影响。**请在内存充足的机器（或服务器）上运行以获得完整 odom/物理行为**。

### 2.4 本包依赖

```bash
sudo apt-get install ros-humble-robot-state-publisher ros-humble-xacro \
  ros-humble-rviz2 ros-humble-teleop-twist-keyboard
```

---

## 3. 构建

```bash
cd inspection_robot
colcon build --symlink-install
source install/setup.bash
```

---

## 4. 运行

### 4.1 一键启动

```bash
source install/setup.bash
ros2 launch inspection_robot inspection_robot.launch.py
```

会依次启动：robot_state_publisher → gz sim(隧道世界) → spawn 小车 → ros_gz_bridge → odom TF → RViz。

常用变体：

```bash
# 无 gz GUI（服务器 / 远程）
ros2 launch inspection_robot inspection_robot.launch.py headless:=true
# 不启动 RViz
ros2 launch inspection_robot inspection_robot.launch.py use_rviz:=false
# 自定义世界
ros2 launch inspection_robot inspection_robot.launch.py world:=/path/to/my.world
```

### 4.2 遥控小车

```bash
# 键盘遥控(推荐)
ros2 run teleop_twist_keyboard teleop_twist_keyboard

# 或手动发布 Twist
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.3}, angular: {z: 0.0}}' --rate 10
```

### 4.3 验证话题（应全部有数据）

```bash
ros2 topic list
ros2 topic echo /imu --once
ros2 topic echo /camera/camera_info --once
ros2 topic echo /odom --once
# 图像
ros2 run rqt_image_view rqt_image_view /camera/image_raw
# 点云(需 RViz)
```

---

## 5. 话题一览

| ROS2 话题 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/cmd_vel` | geometry_msgs/Twist | ROS→GZ | 差速驱动遥控 |
| `/odom` | nav_msgs/Odometry | GZ→ROS | 里程计 |
| `/camera/image_raw` | sensor_msgs/Image | GZ→ROS | RGB 图像 1280×720@30Hz |
| `/camera/camera_info` | sensor_msgs/CameraInfo | GZ→ROS | 相机内参 |
| `/imu` | sensor_msgs/Imu | GZ→ROS | IMU 200Hz |
| `/lidar/points` | sensor_msgs/PointCloud2 | GZ→ROS | 16线点云 10Hz |

---

## 6. TF 树（完整）

```
odom
 └── base_link                    (动态, odom_tf_broadcaster)
      ├── left_wheel              (continuous)
      ├── right_wheel             (continuous)
      ├── caster_front_left       (fixed)
      ├── caster_rear_right       (fixed)
      ├── camera_link             (fixed)
      ├── imu_link                (fixed)
      └── lidar_link              (fixed)
```

验证：`ros2 run tf2_tools view_frames` 或 `rqt_tf_tree`。

---

## 7. 传感器位姿与联合标定

所有传感器安装位姿集中在 `urdf/inspection_robot.urdf.xacro` **顶部**的 `xacro:property`：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `camera_x/y/z`、`camera_roll/pitch/yaw` | 0.22 / 0 / 0.10 / 0 / 0 / 0 | 相机相对 base_link 位姿 |
| `imu_x/y/z` | 0 / 0 / 0.02 | IMU 相对 base_link |
| `lidar_x/y/z` | 0 / 0 / 0.30 | 雷达相对 base_link |

- **相机光心** = `camera_link` 原点，+x 朝前；
- 修改这些参数即修改**外参真值**。做**相机-雷达-IMU 联合标定**时：
  1. 在仿真中摆出标定板/角点特征物；
  2. 采集 `/camera/image_raw`、`/lidar/points`、`/imu`；
  3. 用 `easy_handeye`（或自研求解器）估计外参；
  4. 把估计结果回填到 xacro 属性，与仿真真值对比验证。

---

## 8. 后续集成规划

- **FAST-LIO2**：点云 `/lidar/points` + `/imu` 输入，里程计 `/odom` 可作为先验/校验；
- **多模态病害检测**：`/camera/image_raw` 与 `/lidar/points` 时间戳对齐 + 外参变换（camera↔lidar 已留参数）；
- **隧道病害巡检**：可替换 `inspection_tunnel.world` 中的特征物为管片病害贴图，供分割/分类模型仿真验证。

---

## 9. 开发调试步骤

### 9.1 分步启动（排查用）

```bash
# 1) 只起 gz，看世界与小车是否正常
gz sim worlds/inspection_tunnel.world
#    (可选) 手动 spawn：
ros2 run ros_gz_sim create -topic /robot_description -entity inspection_robot -z 0.20 -x 0 -y 0

# 2) 单独验证静态 TF
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p robot_description:="$(xacro urdf/inspection_robot.urdf.xacro)"

# 3) 单独验证桥接
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=config/ros_gz_bridge.yaml

# 4) 单独验证 odom TF
ros2 run inspection_robot odom_tf_broadcaster.py
```

### 9.2 常见问题排查

| 现象 | 检查步骤 |
|---|---|
| gz 世界/小车不显示 | 1) `gz sim` 单独启动确认世界文件；2) 看 gz 输出有无插件加载错误；3) 确认 `create` 命令执行成功（终端有 spawn 日志） |
| 话题没有数据 | 1) `gz topic -l` 看 gz 侧是否有话题；2) 确认桥接 yaml 的 `gz_topic_name` 与 URDF 中 `<topic>` 一致（都以 `/` 开头绝对话题）；3) `ros2 topic echo` 看桥接是否收到 |
| 只有点云没有图像 | 检查相机 `<render-engine>`；GPU 不支持 ogre2 时加 `--render-engine ogre` 或改用 `GpuLidarSensor` |
| odom 有但小车在 RViz 不动 | 1) `ros2 topic echo /odom` 确认位姿在变；2) `rqt_tf_tree` 确认 `odom->base_link` 存在；3) RViz Fixed Frame 设为 `odom` |
| Twist 发出去小车不动 | 1) `gz topic -e -t /cmd_vel` 看 gz 侧是否收到；2) 确认 DiffDrive 插件 `left_joint/right_joint` 名字与 URDF 一致；3) 轮子 z 高度要能接触地面 |
| 启动时 spawn 报错 | 确认 robot_state_publisher 已启动（`/robot_description` 有内容）；等 gz 世界加载完再 spawn（launch 已内置 3s 延迟） |
| `create` 找不到 | 确认 ros_gz 已源码编译并 source；命令应为 `ros2 run ros_gz_sim create` |
| xacro 报错 | 安装 `ros-humble-xacro`；单独 `xacro urdf/inspection_robot.urdf.xacro` 验证输出 |

### 9.3 建议的集成调试顺序

1. 先单起 `gz sim` 确认世界+小车+传感器都正常（gz 侧话题有数据）；
2. 再起 bridge，确认 ROS 侧话题有数据；
3. 再起 odom TF，确认 RViz 中小车能跟随 odom 移动；
4. 最后接 FAST-LIO2 / 病害检测算法。

---

## 作者

ylf

## 许可证

Apache-2.0（模型/世界文件基于 Gazebo 生态惯例）。
