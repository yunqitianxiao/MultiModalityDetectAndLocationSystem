cmake_minimum_required(VERSION 3.8)
project(inspection_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# 安装数据目录: URDF / 世界 / 桥接配置 / RViz配置 / launch
install(DIRECTORY urdf worlds config launch
  DESTINATION share/${PROJECT_NAME}
)

# 安装可执行 Python 脚本(odom TF 广播 / 位姿状态向量 / 配对数据录制 / 位姿编码器)
install(PROGRAMS scripts/odom_tf_broadcaster.py scripts/pose_state_node.py scripts/record_paired_data.py scripts/pose_encoder_node.py scripts/train_pose_encoder.py
  DESTINATION lib/${PROJECT_NAME}
)

# 安装共享 Python 模块(供上述脚本 import: data_gen.common / pose_mlp 等)
install(FILES data_gen/common.py data_gen/defect_placement.py data_gen/prepare_textures.py data_gen/pose_mlp.py
  DESTINATION lib/${PROJECT_NAME}/data_gen
)

ament_package()
