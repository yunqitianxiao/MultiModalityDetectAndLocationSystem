#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
巡检小车仿真一键启动
  - robot_state_publisher  : 发布 /robot_description + 静态TF(base_link->相机/IMU/雷达)
  - joint_state_publisher  : 发布 /joint_states, 驱动 6 个连续关节(驱动轮/脚轮轴)的 TF
  - gz sim (Harmonic)      : 加载隧道世界(机器人模型已内嵌, 由 generate_world.py 生成)
  - ros_gz_bridge          : 桥接 /cmd_vel /odom /camera /imu /lidar 话题
  - odom_tf_broadcaster    : 发布 odom->base_link 动态TF
  - rviz2                  : 可视化(可选)
  - 裂缝检测推理(infer:=true): tunnel_crack_mm 三模态融合权重实时检测, 发布 /crack/annotated
                               默认 dry-run(只推理画框、不碰 /cmd_vel), 以便与 teleop 遥控并存

配对数据录制模式:
  record:=true 会加载病害世界(defect_tunnel.world), 自动带起 record_paired_data
  节点驱车往返并同步落盘(图像/点云/里程/自动标注真值框), 录完自动关闭 launch。
  配套参数: out_dir / vx / t_forward 传给录制节点。

用法:
  ros2 launch inspection_robot inspection_robot.launch.py
  ros2 launch inspection_robot inspection_robot.launch.py headless:=true   # 无 gz GUI
  ros2 launch inspection_robot inspection_robot.launch.py use_rviz:=false  # 不启动 RViz
  ros2 launch inspection_robot inspection_robot.launch.py record:=true headless:=true use_rviz:=false
  ros2 launch inspection_robot inspection_robot.launch.py record:=true out_dir:=/data/gen1
  ros2 launch inspection_robot inspection_robot.launch.py infer:=true                       # 开检测(只画框, 手动遥控)
  ros2 launch inspection_robot inspection_robot.launch.py infer:=true infer_dry_run:=false  # 全自动闭环(检测->决策->控车)
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, LogInfo, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.substitutions import LaunchConfiguration, Command, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('inspection_robot')

    # ---------------- 可配置参数 ----------------
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    headless = LaunchConfiguration('headless', default='false')
    record   = LaunchConfiguration('record',   default='false')
    world_arg = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    out_dir  = LaunchConfiguration('out_dir',  default=os.path.join(os.path.expanduser('~'), 'workspace', 'inspection_robot', 'data_gen', 'output'))
    vx       = LaunchConfiguration('vx',       default='0.4')
    t_forward = LaunchConfiguration('t_forward', default='40.0')

    xacro_file = os.path.join(pkg_share, 'urdf', 'inspection_robot.urdf.xacro')
    rviz_file  = os.path.join(pkg_share, 'config', 'inspection_robot.rviz')
    bridge_yaml = os.path.join(pkg_share, 'config', 'ros_gz_bridge.yaml')
    normal_world = os.path.join(pkg_share, 'worlds', 'inspection_tunnel.world')
    defect_world = os.path.join(pkg_share, 'worlds', 'defect_tunnel.world')

    # world 默认值由 record 决定: 录制必须用病害世界(否则没有真值目标)
    world_default = PythonExpression([
        "'", defect_world, "' if '", record, "' == 'true' else '", normal_world, "'",
    ])
    # record 模式默认不开 RViz(纯数据采集, 省资源)
    rviz_default = PythonExpression([
        "'true' if '", record, "' == 'false' and '", use_rviz, "' == 'true' else 'false'",
    ])

    # ---------------- 声明 launch 参数 ----------------
    declare_use_rviz = DeclareLaunchArgument('use_rviz', default_value='true', description='是否启动 RViz 可视化')
    declare_headless = DeclareLaunchArgument('headless', default_value='false', description='gz sim 无 GUI 模式(加 -s)')
    declare_record   = DeclareLaunchArgument('record', default_value='false', description='配对数据录制模式(自动带录制节点)')
    declare_world    = DeclareLaunchArgument('world', default_value=world_default,
                                             description='世界文件(默认: record=true 用病害世界, 否则普通隧道)')
    declare_use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true',
                                                 description='仿真时钟同步(桥接 /clock, 录制/控制节点用 sim 时间戳)')
    declare_out_dir  = DeclareLaunchArgument('out_dir', default_value=out_dir)
    declare_vx       = DeclareLaunchArgument('vx', default_value=vx)
    declare_t_forward = DeclareLaunchArgument('t_forward', default_value=t_forward)

    # ---------------- 1. robot_state_publisher ----------------
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': ParameterValue(
            Command(['xacro ', xacro_file]), value_type=str)}],
    )

    # ---------------- joint_state_publisher ----------------
    # 两个驱动轮 + 两组脚轮轴是 continuous 关节, 其 TF 只能由 /joint_states 驱动;
    # gz 世界文件与桥接配置都不提供 /joint_states, 缺了本节点 RViz 里整车会少掉所有轮子。
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ---------------- 2. Gazebo Harmonic ----------------
    # 贴图目录进 GZ_SIM_RESOURCE_PATH: 病害盒 PBR albedo_map 用 file:// 绝对路径
    # (generate_defect_world.py 已按本机重写)。再加资源路径兜底, 防止个别环境
    # 解析绝对路径失败 -> 贴图找不到 -> 白斑。
    import pathlib
    _textures_dir = str(pathlib.Path(__file__).resolve().parent.parent / 'data_gen' / 'textures')
    _gz_env = {'GZ_SIM_RESOURCE_PATH': _textures_dir}
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '--render-engine', 'ogre2', '-v', '3', world_arg],
        output='screen',
        additional_env=_gz_env,
        condition=UnlessCondition(headless),
    )
    gz_sim_headless = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-s', '--render-engine', 'ogre2', '-v', '3', world_arg],
        output='screen',
        additional_env=_gz_env,
        condition=IfCondition(headless),
    )

    # ---------------- 3. ros_gz_bridge ----------------
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        parameters=[{'config_file': bridge_yaml}],
    )

    # ---------------- 4. odom -> base_link TF ----------------
    odom_tf = Node(
        package='inspection_robot',
        executable='odom_tf_broadcaster.py',
        name='odom_tf_broadcaster',
        output='screen',
        parameters=[{'child_frame_id': 'base_link'}, {'use_sim_time': use_sim_time}],
    )

    # ---------------- 5. 配对数据录制节点 (record:=true 时) ----------------
    # 只在桥启动后再 include, 避免 gz 未就绪时 /cmd_vel 无处可去
    recorder = Node(
        package='inspection_robot',
        executable='record_paired_data.py',
        name='record_paired_data',
        output='screen',
        parameters=[{'out_dir': out_dir}, {'vx': vx}, {'t_forward': t_forward},
                    {'use_sim_time': use_sim_time}],
        condition=IfCondition(record),
    )

    # ---------------- 6. RViz (record 模式默认关闭) ----------------
    # 本机是 Optimus 双显卡(AMD iGPU + RTX 4060), rviz2 走硬渲染时 3D 视图全黑;
    # 该变量只能作用于 rviz 自己 —— 全局设它会连 gz 的传感器渲染一起拖垮(仿真直接卡死)。
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_file],
        additional_env={'LIBGL_ALWAYS_SOFTWARE': '1'},
        condition=IfCondition(rviz_default),
    )

    # ---------------- 7. 裂缝检测推理节点 (infer:=true 时) ----------------
    # 该脚本在 tunnel_crack_mm/ 下, 不是 ROS 包, 且必须用 .venv 解释器(rclpy 与
    # torch/CUDA 同环境) => 用 ExecuteProcess 直接起, 不走 Node(package=...)。
    # 默认 --dry-run: 只推理并发布 /crack/annotated, 不碰 /cmd_vel, 这样能和
    # teleop 遥控并存(两者都发 /cmd_vel 会互相打架)。要全自动闭环用 infer_dry_run:=false。
    import pathlib as _pl
    _repo = str(_pl.Path(__file__).resolve().parent.parent)      # 解析 symlink 回源目录
    infer = LaunchConfiguration('infer', default='false')
    infer_dry_run = LaunchConfiguration('infer_dry_run', default='true')
    # 0.006 不是拍脑袋: 该融合权重的分数尺度整体很低(实测最高仅 ~0.02), 模型的
    # 排序是对的但绝对数值小, 用常规的 0.25 会把框全部滤光、日志显示"检测=0"
    # 却毫无报错。runs/pr_eval_crack/pr_results.json 实测最优工作点
    # opt_thresh=0.0057 (P=0.96, R=0.94, mAP@0.5=0.9051), 故取 0.006。
    infer_conf = LaunchConfiguration('infer_conf', default='0.006')
    # --dry-run 并不等于"不控车": 它仍以控制频率往 cmd_topic 发 0 速度, 与遥控并开
    # 会把遥控下发的速度不断清零。所以 dry_run 模式要额外把指令改道到无人订阅的
    # 影子话题(靠脚本的 --cmd-topic 参数), 才真正谈得上与 teleop 并存。
    infer_cmd_topic = PythonExpression([
        "'/crack/cmd_vel_shadow' if '", infer_dry_run, "' == 'true' else '/cmd_vel'"])
    infer_python = LaunchConfiguration(
        'infer_python', default=os.path.join(_repo, '.venv', 'bin', 'python'))
    infer_script = LaunchConfiguration(
        'infer_script',
        default=str((_pl.Path(_repo).parent / 'tunnel_crack_mm' /
                     'tunnel_crack_infer_node.py').resolve()))
    infer_ckpt = LaunchConfiguration(
        'infer_ckpt', default=os.path.join(_repo, 'runs', 'assist_crack', 'last.pt'))

    infer_node = ExecuteProcess(
        cmd=[infer_python, '-u', infer_script, '--ckpt', infer_ckpt, '--rate', '5',
             '--conf-thr', infer_conf,
             '--cmd-topic', infer_cmd_topic,
             PythonExpression(["'--dry-run' if '", infer_dry_run, "' == 'true' else '--no-dry-run'"])],
        output='screen',
        condition=IfCondition(infer),
    )

    declare_infer = DeclareLaunchArgument(
        'infer', default_value='false',
        description='启动裂缝检测推理节点(需 tunnel_crack_mm 的融合权重)')
    declare_infer_dry_run = DeclareLaunchArgument(
        'infer_dry_run', default_value='true',
        description='true=只推理画框不控车(可与 teleop 并存); false=/cmd_vel 全自动闭环')
    declare_infer_conf = DeclareLaunchArgument(
        'infer_conf', default_value='0.006',
        description='检测置信度阈值; 该融合权重分数尺度整体偏低, 出厂工作点是 0.006 而非常规 0.25')
    declare_infer_python = DeclareLaunchArgument('infer_python', default_value=infer_python)
    declare_infer_script = DeclareLaunchArgument('infer_script', default_value=infer_script)
    declare_infer_ckpt = DeclareLaunchArgument('infer_ckpt', default_value=infer_ckpt)

    # 桥起来后再带起录制节点; 录制结束(节点自退出)后关闭整个 launch
    # (record=false 时 recorder 不启动, OnProcessExit 不会触发, launch 正常常驻)
    start_recorder = RegisterEventHandler(
        OnProcessStart(target_action=bridge, on_start=[recorder]))
    stop_on_record_end = RegisterEventHandler(
        OnProcessExit(target_action=recorder, on_exit=[Shutdown()]))

    return LaunchDescription([
        declare_use_rviz,
        declare_headless,
        declare_record,
        declare_world,
        declare_use_sim_time,
        declare_out_dir,
        declare_vx,
        declare_t_forward,
        declare_infer,
        declare_infer_dry_run,
        declare_infer_conf,
        declare_infer_python,
        declare_infer_script,
        declare_infer_ckpt,
        robot_state_publisher,
        joint_state_publisher,
        gz_sim,
        gz_sim_headless,
        bridge,
        odom_tf,
        rviz,
        infer_node,
        start_recorder,
        stop_on_record_end,
    ])
