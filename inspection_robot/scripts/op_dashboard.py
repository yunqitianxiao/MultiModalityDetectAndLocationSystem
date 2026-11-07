#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
巡检小车远程操作台 (无依赖 web 方案)
====================================
给"远程操纵仿真小车"用: 通过 AutoDL 自定义服务端口 (6006) 提供:
  - /stream.mjpg   相机 MJPEG 流
  - /lidar.mjpg    雷达俯视 MJPEG 流 (机器人居中朝上, 隧道轮廓)
  - /              遥操作页面 (WASD/按钮 + 状态)
  - /cmd?vx=..&wz=..  发 cmd_vel (HTTP)
  - /status            里程/速度 JSON 轮询

纯 Python 标准库 + rclpy + cv2, 无 websocket/框架依赖。
安全: 2s 无新遥操作指令自动停车 (看门狗)。

用法: source setup.sh && python3 op_dashboard.py --port 6006
"""
import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs_py import point_cloud2

# ------------------------------------------------------------------ ROS 节点
class OpsNode(Node):
    def __init__(self):
        super().__init__('op_dashboard')
        # 共享状态
        self._lock = threading.Lock()
        self._jpeg_cam = None
        self._jpeg_lidar = None
        self._img_err_shown = False
        self._odom = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'vx': 0.0, 'wz': 0.0}
        self._last_cmd_time = time.time()
        self._stopped = False

        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self._on_img, 1)
        self.sub_lid = self.create_subscription(PointCloud2, '/lidar/points', self._on_lidar, 5)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self._on_odom, 5)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 5)

        # 看门狗: 2s 无新指令自动停车
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()

    # ---------------- 回调 ----------------
    def _on_img(self, msg):
        try:
            n = msg.height * msg.width
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            if msg.encoding in ('rgb8', 'bgr8'):
                img = arr.reshape((msg.height, msg.width, 3))
                if msg.encoding == 'rgb8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding in ('mono8', '8UC1'):
                img = arr.reshape((msg.height, msg.width))
            else:
                return
            h, w = img.shape[:2]
            if w > 720:
                img = cv2.resize(img, (720, int(h * 720.0 / w)))
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with self._lock:
                    self._jpeg_cam = buf.tobytes()
        except Exception as exc:
            if not self._img_err_shown:
                self._img_err_shown = True
                print(f'[op_dashboard] camera err: {exc!r}', flush=True)

    def _on_lidar(self, msg):
        try:
            pts = [(float(p[0]), float(p[1])) for p in
                   point_cloud2.read_points(msg, field_names=('x', 'y'), skip_nans=True)]
            step = max(1, len(pts) // 3000)
            self._jpeg_lidar = self._render_lidar(pts[::step])
        except Exception:
            pass

    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self._lock:
            self._odom = {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'yaw': yaw,
                'vx': msg.twist.twist.linear.x,
                'wz': msg.twist.twist.angular.z,
            }

    # ---------------- 指令 ----------------
    def set_cmd(self, vx, wz):
        t = Twist()
        t.linear.x = float(vx)
        t.angular.z = float(wz)
        self.pub_cmd.publish(t)
        with self._lock:
            self._last_cmd_time = time.time()
            self._stopped = False

    def stop_now(self):
        self.set_cmd(0.0, 0.0)
        with self._lock:
            self._stopped = True

    def _watchdog_loop(self):
        while True:
            time.sleep(0.5)
            with self._lock:
                stale = time.time() - self._last_cmd_time > 2.0 and not self._stopped
            if stale:
                t = Twist()
                self.pub_cmd.publish(t)
                with self._lock:
                    self._stopped = True
                    self._last_cmd_time = time.time()

    # ---------------- 渲染 ----------------
    @staticmethod
    def _render_lidar(pts):
        S = 480
        scale = 22.0  # px / m
        img = np.zeros((S, S, 3), dtype=np.uint8)
        cx = cy = S // 2
        for r in (2, 5, 10, 15):
            cv2.circle(img, (cx, cy), int(r * scale), (45, 45, 45), 1)
        for x, y in pts:
            sx = cx + y * scale
            sy = cy - x * scale
            if 0 <= sx < S and 0 <= sy < S:
                cv2.circle(img, (int(sx), int(sy)), 1, (0, 255, 255), -1)
        # 机器人箭头(绿色, 朝上=前进)
        cv2.arrowedLine(img, (cx, cy), (cx, cy - 34), (0, 220, 0), 3, tipLength=0.35)
        cv2.circle(img, (cx, cy), 8, (0, 220, 0), 2)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buf.tobytes() if ok else None

    # ---------------- 对外读取 ----------------
    def jpeg_cam(self):
        with self._lock:
            return self._jpeg_cam

    def jpeg_lidar(self):
        with self._lock:
            return self._jpeg_lidar

    def odom(self):
        with self._lock:
            return dict(self._odom)


# ------------------------------------------------------------------ HTTP
NODE = None  # 启动后绑定

INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>巡检小车 · 远程操作台</title>
<style>
  body{margin:0;background:#101014;color:#e8e8ee;font-family:"Segoe UI",system-ui,sans-serif}
  .wrap{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:minmax(0,1fr) auto auto;gap:8px;
        height:100vh;padding:10px;box-sizing:border-box}
  .card{background:#1a1a22;border:1px solid #2c2c3a;border-radius:10px;padding:8px;display:flex;flex-direction:column;min-height:0}
  .card h2{margin:2px 0 6px;font-size:13px;color:#9aa;font-weight:600}
  .card img{width:100%;object-fit:contain;background:#000;border-radius:6px;flex:1;min-height:0}
  .status{grid-column:1/3;display:flex;gap:22px;align-items:center;background:#15151d;border:1px solid #2c2c3a;
          border-radius:10px;padding:8px 14px;font-size:13px;flex-wrap:wrap}
  .status b{color:#fff}
  .controls{grid-column:1/3;display:flex;gap:26px;align-items:center;justify-content:center;padding:6px}
  .dpad{display:grid;grid-template-columns:repeat(3,58px);grid-template-rows:repeat(3,58px);gap:4px}
  .dpad button{font-size:20px;border:none;border-radius:8px;background:#26263a;color:#fff;cursor:pointer}
  .dpad button:active{background:#3a3a55}
  .stop{width:100px;height:100px;border-radius:50%;background:#b22222;color:#fff;font-size:16px;border:3px solid #e44;cursor:pointer}
  .stop:active{background:#d33}
  .speed input{width:130px;vertical-align:middle}
  label{font-size:12px;color:#889}
  #conn{font-weight:600}
</style></head><body>
<div class="wrap">
  <div class="card"><h2>📷 相机画面</h2><img src="/stream.mjpg" alt="camera"></div>
  <div class="card"><h2>📡 激光雷达俯视 (机器人居中, 朝上)</h2><img src="/lidar.mjpg" alt="lidar"></div>
  <div class="status">
    <span>里程 X <b id="ox">0.00</b> Y <b id="oy">0.00</b> 航向 <b id="oyaw">0.0°</b></span>
    <span>速度 线 <b id="svx">0.00</b> 角 <b id="swz">0.00</b></span>
    <span class="speed">最大线速 <input id="vmax" type="range" min="0.1" max="1.0" step="0.05" value="0.5"><b id="vmaxv">0.50</b> m/s</span>
    <span id="conn" style="color:#c66">连接中…</span>
  </div>
  <div class="controls">
    <div class="dpad">
      <span></span><button id="bw">▲</button><span></span>
      <button id="ba">◀</button><button id="bs">▼</button><button id="bd">▶</button>
    </div>
    <button class="stop" id="bstop">⏹ 急停</button>
  </div>
</div>
<script>
const held=new Set();
function send(){
  let vx=0,wz=0;
  const v=parseFloat(document.getElementById('vmax').value);
  if(held.has('w'))vx+=v;
  if(held.has('s'))vx-=v;
  if(held.has('a'))wz+=v*1.5;
  if(held.has('d'))wz-=v*1.5;
  fetch('/cmd?vx='+vx.toFixed(3)+'&wz='+wz.toFixed(3));
}
// 心跳: 按住键时每400ms续发指令, 防止2s看门狗在长按期间误停
// (之前只在按键事件发一次, 长按>2s必被看门狗自动停车)
setInterval(()=>{ if(held.size>0) send(); }, 400);
const press=k=>{held.add(k);send();};
const release=k=>{held.delete(k);send();};
// 页面失焦(切走/锁屏)清空按键, 防止键位状态残留导致持续行驶
window.addEventListener('blur',()=>{ if(held.size){held.clear();send();} });
document.addEventListener('keydown',e=>{const k=e.key.toLowerCase();if('wasd'.includes(k)&&!e.repeat)press(k);});
document.addEventListener('keyup',e=>{const k=e.key.toLowerCase();if('wasd'.includes(k))release(k);});
['w','a','s','d'].forEach(k=>{
  const b=document.getElementById('b'+k);
  b.addEventListener('pointerdown',()=>press(k));
  b.addEventListener('pointerup',()=>release(k));
  b.addEventListener('pointerleave',()=>release(k));
  b.addEventListener('pointercancel',()=>release(k));
});
document.getElementById('bstop').addEventListener('click',()=>{held.clear();fetch('/cmd?vx=0&wz=0');});
document.getElementById('vmax').addEventListener('input',e=>{document.getElementById('vmaxv').textContent=(+e.target.value).toFixed(2);});
async function loop(){
  try{
    const s=await(await fetch('/status')).json();
    document.getElementById('ox').textContent=s.x.toFixed(2);
    document.getElementById('oy').textContent=s.y.toFixed(2);
    document.getElementById('oyaw').textContent=(s.yaw*180/Math.PI).toFixed(1);
    document.getElementById('svx').textContent=s.vx.toFixed(2);
    document.getElementById('swz').textContent=s.wz.toFixed(2);
    document.getElementById('conn').textContent='● 在线';
    document.getElementById('conn').style.color='#3c6';
  }catch(e){
    document.getElementById('conn').textContent='○ 离线';
    document.getElementById('conn').style.color='#c66';
  }
  setTimeout(loop,500);
}
loop();
</script></body></html>"""


def _mjpeg_head():
    return (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
            + b'1234\r\n\r\n')  # 占位, 实际逐帧写长度


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send_mjpg(self, getter):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()
        try:
            while True:
                jpg = getter()
                if jpg:
                    frame = (b'--frame\r\nContent-Type: image/jpeg\r\n'
                             + ('Content-Length: %d\r\n\r\n' % len(jpg)).encode()
                             + jpg + b'\r\n')
                    self.wfile.write(frame)
                    self.wfile.flush()
                time.sleep(0.06)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            body = INDEX_HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == '/stream.mjpg':
            self._send_mjpg(NODE.jpeg_cam)
        elif u.path == '/lidar.mjpg':
            self._send_mjpg(NODE.jpeg_lidar)
        elif u.path == '/status':
            s = NODE.odom()
            body = json.dumps(s).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == '/cmd':
            q = parse_qs(u.query)
            vx = float(q.get('vx', ['0'])[0])
            wz = float(q.get('wz', ['0'])[0])
            NODE.set_cmd(vx, wz)
            self.send_response(200)
            self.send_header('Content-Length', '2')
            self.end_headers()
            self.wfile.write(b'ok')
        else:
            self.send_response(404)
            self.send_header('Content-Length', '0')
            self.end_headers()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=6006)
    args = ap.parse_args()

    global NODE
    rclpy.init()
    NODE = OpsNode()
    threading.Thread(target=rclpy.spin, args=(NODE,), daemon=True).start()

    srv = ThreadingHTTPServer(('0.0.0.0', args.port), Handler)
    print(f'[op_dashboard] listening on :{args.port}', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
