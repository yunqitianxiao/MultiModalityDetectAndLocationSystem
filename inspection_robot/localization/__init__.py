"""
localization 包 —— 空间语义映射定位链 + 几何避障检测
================================================================
- img_px_to_world : 检测框 -> 里程/环号/时钟方位/3D坐标(像素→相机→车体→世界→隧道)
- tunnel_params   : 隧道几何 + 传感器标定(与 output_v7 对齐)
- det_source      : 检测框来源接口(GTBoxSource 现在 / FusionDetSource 将来)
- obstacle        : 几何避障检测(BEV 高度图 + 连通域 -> {障碍物距离/方位})
- verify_chain    : 端到端验证脚本(定位断言 + 避障断言 + 可视化)
"""
from localization.img_px_to_world import (  # noqa: F401
    DetectionLocalizer, TunnelGeometry, pose_to_T, project_to_image,
    lidar_to_bev_height, lidar_to_image_attributes, record_to_json,
)
from localization.tunnel_params import (  # noqa: F401
    build_tunnel, load_calibration, RAIL_TOP, INTRUSION_Z_MAX,
)
from localization.det_source import (  # noqa: F401
    DetectionSource, GTBoxSource, FusionDetSource,
)
from localization.obstacle import (  # noqa: F401
    Obstacle, find_obstacles, bev_height_map,
)
