#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三模态融合(单类裂缝) 实时推理 + 决策节点
================================================================
订阅 /camera/image_raw + /lidar/points + /odom
  -> 在线构建点云属性图(depth/reflec/height)
  -> PointCloudAssistedYOLO(阶段3权重) 推理
  -> 解码检测框 -> 框内深度中值 -> 车体系 距离/方位
  -> DecisionDecoder -> 发布 /cmd_vel

预处理严格对齐 fusion/pointcloud_assist_detector.load_sample(), 否则实时与离线不符:
  - 图像: LetterBox(new_shape=(640,640), auto=False, stride=32) + BGR, /255, CHW
  - attrmap: 只取 depth/reflec/height **三通道**(第4通道 mask 不喂模型),
             reflec 按本帧 1/99 分位归一化, 三通道各自 fill_nearest 补密
  - pose_state: 12 维 [x,y,z,sin/cos_roll,sin/cos_pitch,sin/cos_yaw,vx,vy,wz],
             **vx/vy/wz 恒填 0** —— 训练时 poses/*.json 无速度字段, 填真实速度会造成
             训练/推理分布不一致

INSPECT 放行: DecisionDecoder 只有停车锁存、没有"已检查"概念, 不外部标记就会永远停在
同一缺陷前。这里与离线闭环同法: 每个检测锚定到世界系, 停够 --hold-s 秒后把该世界点记入
visited, 之后半径 --visited-r 内的目标不再喂给决策器 -> 放行到下一处。

用法(需要同时具备 rclpy 与 torch/CUDA 的 Python 环境):
  source /opt/ros/humble/setup.bash
  python -u tunnel_crack_infer_node.py \
      --ckpt ../weights/assist_crack.pt --rate 5
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

_IR = os.environ.get(
    'IR_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'inspection_robot')),
)
if _IR not in sys.path:
    sys.path.insert(0, _IR)

import rclpy                                             # noqa: E402
from rclpy.node import Node                              # noqa: E402
from sensor_msgs.msg import Image, PointCloud2           # noqa: E402
from nav_msgs.msg import Odometry                        # noqa: E402
from geometry_msgs.msg import Twist                      # noqa: E402
from numpy.lib.stride_tricks import as_strided           # noqa: E402

from data_gen.common import (                            # noqa: E402
    K, T_CAM_BASE, T_LIDAR_BASE, IMG_SIZE, fill_nearest,
    lidar_to_image_attributes, odom_to_T_world_base,
)
from fusion.config import YOLO_SIZE                      # noqa: E402
from fusion.pointcloud_assist_detector import PointCloudAssistedYOLO  # noqa: E402
from localization.decision_decoder import DecisionDecoder, Target     # noqa: E402
from localization.img_px_to_world import camera_to_base, pixel_to_camera  # noqa: E402
from localization.tunnel_params import AVOID_V_CRUISE, AZ_ABEAM_DEG   # noqa: E402

H, W = IMG_SIZE[1], IMG_SIZE[0]


# ---------------------------------------------------------------- 纯函数
def letterbox_geo(h, w, new_size=YOLO_SIZE):
    """复刻 ultralytics LetterBox(auto=False) 几何: (r, dw, dh)。
    反变换 x_native = (x_lb - dw) / r。"""
    r = min(new_size / h, new_size / w)
    nw, nh = round(w * r), round(h * r)
    return r, (new_size - nw) / 2.0, (new_size - nh) / 2.0


def preprocess_image(img_bgr):
    """BGR uint8 HxWx3 -> letterbox 640 的 BGR float32 (3,640,640)。
    与 load_sample 的 LetterBox + transpose + /255 逐字一致。"""
    from ultralytics.data.augment import LetterBox
    lb = LetterBox(new_shape=(YOLO_SIZE, YOLO_SIZE), auto=False, stride=32)
    img640 = lb(image=img_bgr)
    return np.transpose(img640.astype(np.float32) / 255.0, (2, 0, 1))


def pose_json_to_state(p):
    """poses JSON -> 12 维, 字段序与 fusion.pointcloud_assist_detector 一致。
    JSON 无速度 -> vx/vy/wz 恒 0(推理必须与训练同分布)。"""
    x, y, z = p['x'], p['y'], p['z']
    qx, qy, qz, qw = p['qx'], p['qy'], p['qz'], p['qw']
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.array([x, y, z, math.sin(roll), math.cos(roll), math.sin(pitch),
                     math.cos(pitch), math.sin(yaw), math.cos(yaw),
                     0.0, 0.0, 0.0], dtype=np.float32)


def yaw_from_pose(p):
    return math.atan2(2 * (p['qw'] * p['qz'] + p['qx'] * p['qy']),
                      1 - 2 * (p['qy'] * p['qy'] + p['qz'] * p['qz']))


def world_z_of_cloud(cloud, pose):
    """车体系点云 -> 世界系 z (height 通道用)。同 scripts/build_attrmaps.py。"""
    T_wb = odom_to_T_world_base([pose['x'], pose['y'], pose['z']],
                                [pose['qx'], pose['qy'], pose['qz'], pose['qw']])
    Xw = (T_wb @ np.concatenate([cloud[:, :3], np.ones((len(cloud), 1))], axis=1).T).T
    return Xw[:, 2]


def build_attrmap(cloud, pose):
    """车体系点云 (N,4) -> (3,720,1280) float32, 与 build_attrmaps.py 同法但只取 3 通道。
    点云全空/补密后仍有 NaN 时返回 None(调用方发 0 速度)。"""
    if cloud is None or len(cloud) == 0:
        return None
    zworld = world_z_of_cloud(cloud, pose)
    depth, reflec, height = lidar_to_image_attributes(
        cloud[:, :3], T_CAM_BASE, K, IMG_SIZE,
        reflectivity=cloud[:, 3], z_world=zworld)
    valid_r = reflec[np.isfinite(reflec)]
    if len(valid_r):
        lo, hi = np.percentile(valid_r, 1), np.percentile(valid_r, 99)
        if hi > lo:
            reflec = np.clip((reflec - lo) / (hi - lo), 0.0, 1.0)
    maps = [fill_nearest(m) for m in (depth, reflec, height)]
    if not all(np.isfinite(m).all() for m in maps):
        return None                       # 补密后仍空洞(该帧无有效投影)
    return np.stack(maps).astype(np.float32)


def nms(xyxy, conf, cls, iou_thr=0.45):
    """按 conf 降序抑制同 class 且 IoU>thr 的框。"""
    order = np.argsort(-conf)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        same = cls[rest] == cls[i]
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0]); yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2]); yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ai = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        ar = (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        iou = inter / (ai + ar - inter + 1e-9)
        order = rest[~(same & (iou > iou_thr))]
    return keep


def box_location(dmap, bbox):
    """原生像素框 -> 框内有效深度中值(相机系 m)。无有效深度时退回框中心单像素。"""
    x1, y1, x2, y2 = bbox
    xa = int(max(0.0, min(W - 1, x1))); xb = int(max(xa + 1, min(W, x2)))
    ya = int(max(0.0, min(H - 1, y1))); yb = int(max(ya + 1, min(H, y2)))
    reg = dmap[ya:yb, xa:xb]
    valid = reg[reg > 0.01]
    if valid.size:
        return float(np.median(valid)), int(valid.size)
    cx = int(max(0.0, min(W - 1, x1 + (x2 - x1) / 2)))
    cy = int(max(0.0, min(H - 1, y1 + (y2 - y1) / 2)))
    return float(dmap[cy, cx]), 0


# ---------------------------------------------------------------- 节点
class CrackInferNode(Node):
    def __init__(self, args):
        super().__init__('tunnel_crack_infer')
        # 必须在建 timer/订阅之前设: 仿真下 /clock 的 sec 是几十, 墙钟是 1.79e9,
        # 用墙钟去比 stamp 会让"数据新鲜度"恒判超时 -> 永久停车(车一动不动)。
        if args.sim_time:
            from rclpy.parameter import Parameter
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.args = args
        self.dev = 'cuda' if (torch.cuda.is_available() and args.device != 'cpu') else 'cpu'
        self.r, self.dw, self.dh = letterbox_geo(H, W)

        ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        self.nc = int(ck.get('nc', 1))
        self.use_point = bool(ck.get('use_point', True)) and not args.no_point
        # 微调过的图像主干(裂缝 best.pt, nc=1)的 cv3 分类层与 COCO(nc=80)通道不同,
        # 必须用同一主干重建, 否则 image_encoder.*.cv3.* 形状不匹配(ckpt.args 里存了它)
        img_ckpt = (ck.get('args') or {}).get('img_ckpt') or None
        self.model = PointCloudAssistedYOLO(use_point=self.use_point, nc=self.nc,
                                            device=self.dev, ckpt=img_ckpt)
        self.model.load_state_dict(ck['model'])
        self.model.eval()
        torch.set_grad_enabled(False)
        self.get_logger().info(
            f'模型已载入 {args.ckpt} (nc={self.nc}, use_point={self.use_point}, device={self.dev})')

        # face_kp=0: 关掉"转向把目标摆到画面中心"。该语义是给正前方障碍用的(朝它开),
        # 而隧道裂缝贴在 ±1.25m 的墙面上, 3.5m 外只有 ~20° 偏轴、本来就在视场内;
        # 朝它转 = 把车往墙上开, 且 yaw 是角速度指令、没有回正项 -> heading 单向累积
        # (+4° 一路锁到 -17°), 车最终把相机怼进右墙、画面退化、检测瞬间归零后 FREE 直冲出去。
        # 直隧道里"沿轴向直行 + 到距停车拍照"才是对的。
        self.dec = DecisionDecoder(conf_thr={i: args.conf_thr for i in range(self.nc)},
                                   class_names=args.class_names, face_kp=0.0,
                                   stop_dist=args.stop_dist, slow_dist=args.slow_dist)
        self.class_names = args.class_names

        # 缓冲
        self.latest_img = None          # (stamp, BGR)
        self.latest_pc = None           # (stamp, np.array Nx4 车体系)
        self.latest_odom = None         # (stamp, dict)
        self._last_t = 0.0
        self._last_status = -1e9
        self._last_warn = {}            # 告警节流(墙钟, 与 sim 时间无关)

        # INSPECT 放行状态
        self.visited = []               # list[np(2)] 已检查的世界锚点
        self.inspecting_anchor = None
        self.hold_until = 0.0
        self._last_state = None

        self.create_subscription(Image, args.image_topic, self.cb_img, 5)
        self.create_subscription(PointCloud2, args.lidar_topic, self.cb_pc, 5)
        self.create_subscription(Odometry, args.odom_topic, self.cb_odom, 5)
        self.pub_cmd = self.create_publisher(Twist, args.cmd_topic, 5)
        self.pub_viz = None if args.no_viz else self.create_publisher(Image, args.viz_topic, 2)
        self.create_timer(1.0 / max(0.5, args.rate), self.tick)
        self.get_logger().info(
            f'控制 {args.rate}Hz | conf_thr={args.conf_thr} | hold={args.hold_s}s '
            f'| visited_r={args.visited_r}m | dry_run={args.dry_run}')

    # ---------- 回调 ----------
    def cb_img(self, msg: Image):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if arr.shape[2] == 3:
            arr = arr[:, :, ::-1]                    # RGB8 -> BGR
        self.latest_img = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                           np.ascontiguousarray(arr))

    def cb_pc(self, msg: PointCloud2):
        cloud = self._pc_to_base(msg)
        self.latest_pc = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, cloud)

    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.latest_odom = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                            {'x': p.x, 'y': p.y, 'z': p.z,
                             'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w})

    def _pc_to_base(self, msg):
        """PointCloud2 -> (N,4) 车体系 [x,y,z,intensity]。
        与 scripts/record_paired_data.py 的 _pc_to_base 同一实现(手动按 offset 解析,
        因 gz gpu_lidar 的 intensity(FLOAT32)+ring(UINT16) 混型会让 read_points_numpy 炸)。"""
        n = msg.width * msg.height
        ps = msg.point_step
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        off = {f.name: f.offset for f in msg.fields}
        rows = as_strided(buf, shape=(n, ps), strides=(ps, 1))

        def _f32(name):
            o = off.get(name)
            if o is None:
                return None
            return np.ascontiguousarray(rows[:, o:o + 4]).view(np.float32).reshape(-1)

        x, y, z = _f32('x'), _f32('y'), _f32('z')
        if x is None or y is None or z is None:
            return None
        inten = _f32('intensity')
        if inten is None:
            inten = np.ones(n, dtype=np.float32)
        xyz = np.stack([x, y, z], axis=1)
        ok = np.isfinite(xyz).all(axis=1)
        xyz, inten = xyz[ok], inten[ok]
        X4 = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1).T
        base = (T_LIDAR_BASE @ X4).T[:, :3]
        return np.concatenate([base, inten.reshape(-1, 1)], axis=1).astype(np.float32)

    # ---------- 决策辅助 ----------
    def _anchor(self, dist, az_deg, pose):
        """车体系(distance, azimuth) -> 世界系 (x,y)。azimuth 左正右负, 前向沿 yaw。"""
        yaw = yaw_from_pose(pose)
        a = math.radians(az_deg)
        fwd, left = dist * math.cos(a), dist * math.sin(a)
        return np.array([pose['x'] + fwd * math.cos(yaw) - left * math.sin(yaw),
                         pose['y'] + fwd * math.sin(yaw) + left * math.cos(yaw)])

    def _is_visited(self, wpos):
        for v in self.visited:
            if float(np.hypot(wpos[0] - v[0], wpos[1] - v[1])) < self.args.visited_r:
                return True
        return False

    # ---------- 主循环 ----------
    def _warn_throttled(self, key, msg, period=2.0):
        """按 key 节流告警(不依赖 logger 的 throttle 扩展)。"""
        if self._last_warn.get(key, -1e9) + period <= time.monotonic():
            self._last_warn[key] = time.monotonic()
            self.get_logger().warn(msg)

    def tick(self):
        if not (self.latest_img and self.latest_pc and self.latest_odom):
            return
        # 数据新鲜度: 消息 stamp 与 ROS 时钟同基准(仿真下为 sim 时间), 别用墙钟比
        now = self.get_clock().now().nanoseconds * 1e-9
        img_stamp, img_bgr = self.latest_img
        if abs(now - img_stamp) > self.args.timeout:
            self._warn_throttled('stale', '数据超时, 停车')
            self._stop()
            return
        _, cloud = self.latest_pc
        _, pose = self.latest_odom

        # 同一帧不重复推理
        if img_stamp <= self._last_t:
            return
        self._last_t = img_stamp

        # --- 属性图 ---
        attrmap = build_attrmap(cloud, pose)
        if attrmap is None:
            self._warn_throttled('attrmap', '属性图构建失败(点云空), 停车')
            self._stop()
            return

        # --- 推理 ---
        inp = torch.from_numpy(preprocess_image(img_bgr))[None].to(self.dev)
        am = torch.from_numpy(attrmap)[None].to(self.dev)
        ps = torch.from_numpy(pose_json_to_state(pose))[None].to(self.dev)
        try:
            out = self.model(inp, am, ps)
        except Exception as e:                              # noqa: BLE001
            self.get_logger().error(f'推理异常: {e}')
            self._stop()
            return
        y = out[0] if isinstance(out, (tuple, list)) else out
        y = y[0].float().cpu()

        # --- 解码框 ---
        xywh = y[:4].t().numpy()
        conf_all = y[4:]
        cc, cid = conf_all.max(0)
        cc, cid = cc.numpy(), cid.numpy()
        m = cc >= self.args.conf_thr
        dets = []
        if m.any():
            cc2, cid2, xywh2 = cc[m], cid[m], xywh[m]
            xc, yc, w, h = xywh2[:, 0], xywh2[:, 1], xywh2[:, 2], xywh2[:, 3]
            xyxy = np.stack([(xc - w / 2 - self.dw) / self.r, (yc - h / 2 - self.dh) / self.r,
                             (xc + w / 2 - self.dw) / self.r, (yc + h / 2 - self.dh) / self.r], 1)
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, W)
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, H)
            keep = nms(xyxy, cc2, cid2, self.args.iou_thr)
            dmap = attrmap[0]
            for j in keep:
                bbox = [float(v) for v in xyxy[j]]
                depth, npts = box_location(dmap, bbox)
                if not math.isfinite(depth) or depth < 0.05 or depth > 200:
                    continue
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                Xb = camera_to_base(pixel_to_camera(cx, cy, depth, K), T_CAM_BASE)
                dist = float(np.linalg.norm(Xb))
                az = float(np.degrees(np.arctan2(Xb[1], Xb[0])))
                dets.append({'cls': int(cid2[j]), 'conf': float(cc2[j]),
                             'distance': dist, 'azimuth_deg': az,
                             'bbox': bbox, 'grid_pts': npts})

        # --- 世界锚定 + 剔除已检查 ---
        pairs = []          # (wpos, det)
        for d in dets:
            wpos = self._anchor(d['distance'], d['azimuth_deg'], pose)
            if self._is_visited(wpos):
                continue
            pairs.append((wpos, d))
        targets = [Target(cls=d['cls'], conf=d['conf'], distance=d['distance'],
                          azimuth_deg=d['azimuth_deg'],
                          class_name=self.class_names[d['cls']] if d['cls'] < len(self.class_names) else '')
                   for _, d in pairs]
        dec = self.dec.decide(targets, v_now=0.0, nav_yaw=yaw_from_pose(pose))

        # --- INSPECT 停够时长 -> 标记该锚点已检查并放行 ---
        front = dec.front
        if dec.state == 'INSPECT' and front is not None:
            if self.inspecting_anchor is None:
                best, bd = None, 1e9
                for wpos, d in pairs:
                    dd = abs(d['distance'] - front.distance) + abs(d['azimuth_deg'] - front.azimuth_deg)
                    if dd < bd:
                        bd, best = dd, wpos
                self.inspecting_anchor = best
                self.hold_until = now + self.args.hold_s
            if self.inspecting_anchor is not None and now >= self.hold_until:
                self.visited.append(self.inspecting_anchor)
                self.get_logger().info(
                    f'已完成检查 #{len(self.visited)} @world({self.inspecting_anchor[0]:.2f},'
                    f'{self.inspecting_anchor[1]:.2f}) -> 放行')
                self.inspecting_anchor = None
                self.dec.reset()
        else:
            self.inspecting_anchor = None

        # --- 发布指令 ---
        if dec.state != self._last_state:
            self.get_logger().info(
                f'[{dec.state}] v={dec.target_speed:.3f} wz={dec.target_yaw_rate:+.3f} | '
                f'{dec.reason} | 检测{len(dets)} 待检查{len(pairs)} 已查{len(self.visited)}')
            self._last_state = dec.state
        if self.args.dry_run:
            self._publish(0.0, 0.0)
        else:
            self._publish(dec.target_speed, dec.target_yaw_rate)

        # 周期状态行: 只在状态"变化"时打日志会把"一个检测都没有"藏起来, 巡检过程中
        # 恰恰要靠这条判断模型到底有没有在出框。
        if now - self._last_status >= self.args.status_s:
            self._last_status = now
            self.get_logger().info(
                f'[{dec.state}] v={dec.target_speed:.2f} 检测={len(dets)} 待查={len(pairs)} '
                f'已查={len(self.visited)} @({pose["x"]:.1f},{pose["y"]:.1f}) '
                f'yaw={math.degrees(yaw_from_pose(pose)):+.0f}°')

        if self.pub_viz is not None:
            self._publish_viz(img_bgr, dets, dec)

    # ---------- 发布 ----------
    def _publish(self, v, wz):
        if not rclpy.ok():
            return                      # 收尾(SIGTERM/ctrl-c)时 context 已失效, 再 publish 会抛
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(wz)
        self.pub_cmd.publish(t)

    def _stop(self):
        self._publish(0.0, 0.0)

    def _publish_viz(self, img_bgr, dets, dec):
        try:
            import cv2
        except ImportError:
            self.pub_viz = None
            return
        vis = img_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d['bbox']]
            col = (0, 0, 255) if dec.front is not None and abs(d['distance'] - dec.front.distance) < 1e-6 \
                else (0, 200, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"{d['distance']:.1f}m {d['azimuth_deg']:+.0f} {d['conf']:.2f}",
                        (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(vis, f"{dec.state} v={dec.target_speed:.2f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = vis.shape[0], vis.shape[1]
        msg.encoding = 'bgr8'
        msg.step = vis.shape[1] * 3
        msg.data = vis.tobytes()
        self.pub_viz.publish(msg)


def parse_args():
    ap = argparse.ArgumentParser(description='三模态裂缝检测实时推理/决策节点')
    ap.add_argument('--ckpt', required=True, help='fusion/train_assist.py 的 last.pt')
    ap.add_argument('--conf-thr', type=float, default=0.25)
    # 裂缝贴在拱顶/墙面上部, 车在地面最近仅能到 ~3.2m, 沿用避障的 0.5m 阈值会使 INSPECT
    # 永不触发。按"可拍摄距离": 真净距 <=3.5m 停车记录、<6m 开始减速(与离线闭环一致)。
    ap.add_argument('--stop-dist', type=float, default=3.5, help='前向目标真净距 <= 此值 -> 停车检查')
    ap.add_argument('--slow-dist', type=float, default=6.0, help='真净距 < 此值 -> 减速斜坡')
    ap.add_argument('--iou-thr', type=float, default=0.45)
    ap.add_argument('--rate', type=float, default=5.0, help='控制/推理频率 Hz')
    ap.add_argument('--hold-s', type=float, default=4.0, help='INSPECT 停车检查时长(s)')
    ap.add_argument('--status-s', type=float, default=2.0, help='周期状态日志间隔(s)')
    ap.add_argument('--visited-r', type=float, default=0.4, help='已检查锚点的放行半径(m)')
    ap.add_argument('--timeout', type=float, default=1.0, help='数据新鲜度阈值(s), 超时停车')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu'])
    # 仿真节点必须走 sim 时间(见 __init__ 注释); 真机部署时用 --no-sim-time
    ap.add_argument('--sim-time', action=argparse.BooleanOptionalAction, default=True,
                    help='使用 /clock 仿真时间(默认开); 真机用 --no-sim-time')
    ap.add_argument('--no-point', action='store_true', help='载入 RGB-only 消融权重')
    # BooleanOptionalAction 而非 store_true: launch 在 infer_dry_run:=false 时生成
    # "--no-dry-run", store_true 认不出这个参数会直接 argparse 退出(cmd 里的死路)。
    # default=False 与原 store_true 的缺省一致 -> 不传时仍是全自动控车, 行为不变。
    ap.add_argument('--dry-run', action=argparse.BooleanOptionalAction, default=False,
                    help='只推理不发布运动指令(调试); 省略/--no-dry-run=全自动控车')
    ap.add_argument('--no-viz', action='store_true')
    # 指向别处可让本节点"跑着但不控车": 注意 --dry-run 并非不发指令, 它仍以控制
    # 频率往 cmd_topic 发 0 速度, 与遥控同时开会把遥遥控的指令不断清零。
    ap.add_argument('--cmd-topic', default='/cmd_vel')
    ap.add_argument('--image-topic', default='/camera/image_raw')
    ap.add_argument('--lidar-topic', default='/lidar/points')
    ap.add_argument('--odom-topic', default='/odom')
    ap.add_argument('--viz-topic', default='/crack/annotated')
    ap.add_argument('--class-names', nargs='*', default=['crack'])
    return ap.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = CrackInferNode(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node._stop()
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
