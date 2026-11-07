#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nav2 巡航 bringup —— 在巡检小车仿真上做自主导航规划/巡航
================================================================
本 launch 只含 Nav2 规划/控制/行为栈 + 占用地图 + 恒定 map->odom, **不含 AMCL**。
原因: 隧道世界完全确定、机器人出生在世界原点、gz DiffDrive 里程与世界对齐, 故
map==odom, 用静态恒等变换即可定位, 也避开"均匀长隧道 AMCL 沿 x 方向易退化"的坑。

前置(先跑主 launch 把机器人/里程/激光/时钟带起来):
    ros2 launch inspection_robot inspection_robot.launch.py (use_rviz:=false headless:=true)
再(本 launch):
    ros2 launch inspection_robot nav2_bringup.launch.py

前提: 已安装 Nav2 包(见 config/nav2/nav2_params.yaml 顶部注释)。
     sudo apt install ros-humble-nav2-bringup ros-humble-nav2-dwb-controller \
                      ros-humble-nav2-regulated-pure-pursuit-controller \
                      ros-humble-nav2-navfn-planner ros-humble-nav2-behaviors \
                      ros-humble-nav2-velocity-smoother ros-humble-nav2-map-server
     (nav2_bringup 会连带大部分依赖; 上面的显式列出确保 RPP/velocity_smoother 都齐)

巡航用法(另开终端, 给路点):
    ros2 run nav2_simple_commander basic_navigator --help

预期 TF 树:  map -> odom -> base_link -> lidar_link
            (map->odom: 本 launch static transform; odom->base_link: odom_tf_broadcaster;
             base_link->lidar_link: robot_state_publisher 在 主 launch 里)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('inspection_robot')

    # ---------------- 可配置参数 ----------------
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    params_file = LaunchConfiguration(
        'params',
        default=os.path.join(pkg_share, 'config', 'nav2', 'nav2_params.yaml'),
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true', description='仿真时钟(桥接 /clock)')
    declare_params = DeclareLaunchArgument(
        'params', default_value=params_file, description='Nav2 参数文件')

    # ---------------- 恒定 map -> odom (恒等, 见模块 docstring) ----------------
    # 机器人/odom 都按世界原点对齐, 故 map 与 odom 重合; 发布一个固定的 map->odom。
    # parent=map, child=odom; 位姿全 0。
    static_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_static',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    # ---------------- Nav2 lifecycle 服务器 (每个只读自己命名空间的参数) ----------------
    # 都是 lifecycle 节点, 由下方 lifecycle_manager 统一 configure->activate。
    nav_nodes = [
        # 1) 占用地图: 先激活, 使 costmap static_layer 能取 /map
        Node(package='nav2_map_server', executable='map_server', name='map_server',
             output='screen', parameters=[params_file]),
        # 2) 全局规划器
        Node(package='nav2_planner', executable='planner_server', name='planner_server',
             output='screen', parameters=[params_file]),
        # 3) 局部控制器
        Node(package='nav2_controller', executable='controller_server', name='controller_server',
             output='screen', parameters=[params_file]),
        # 4) 行为节点(自旋/退避/等待)
        Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server',
             output='screen', parameters=[params_file]),
        # 5) BT 导航(行为树)节点
        Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
             output='screen', parameters=[params_file]),
        # 6) 路径平滑
        Node(package='nav2_smoother', executable='smoother_server', name='smoother_server',
             output='screen', parameters=[params_file]),
        # 7) 路点巡航
        Node(package='nav2_waypoint_follower', executable='waypoint_follower',
             name='waypoint_follower', output='screen', parameters=[params_file]),
    ]

    # ---------------- 速度平滑器 (普通节点, 非 lifecycle, 把控制器 cmd_vel 限速后发 /cmd_vel) ----------------
    velocity_smoother = Node(
        package='nav2_velocity_smoother', executable='velocity_smoother',
        name='velocity_smoother', output='screen', parameters=[params_file],
    )

    # ---------------- lifecycle 统一激活 ----------------
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': [
                'map_server', 'planner_server', 'controller_server',
                'behavior_server', 'bt_navigator', 'smoother_server',
                'waypoint_follower',
            ],
        }],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_params,
        static_map_odom,
        *nav_nodes,
        velocity_smoother,
        lifecycle_manager,
    ])
