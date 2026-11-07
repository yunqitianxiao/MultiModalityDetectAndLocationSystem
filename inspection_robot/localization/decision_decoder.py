# -*- coding: utf-8 -*-
"""
路径决策解码器 —— 融合检测目标 -> 小车行驶指令
================================================================
把定位链产出的目标列表(每项含 距离/方位/类别/置信度)变成小车该怎么做:
接近 -> 减速 -> 停车检查。类感知触发(各类独立 conf 阈值), 但【动作统一为
"检查"而非"避让"】—— 检测到异物同样是靠上去看, 不走"绕行/躲开"。

状态机(首个匹配生效), front = 首个 conf>=DEC_CONF_THR[cls] 且 |az|<AZ_ABEAM_DEG
的目标(按 distance 升序); d0 = front.distance - body_front(真净距):
  FREE    无前方目标            -> V=AVOID_V_CRUISE, yaw=0
  INSPECT d0<=stop_dist         -> V=0, yaw=0                          (停车检查)
  APPROACH else                 -> V=min(CRUISE, 斜坡), yaw=轻转向居中    (接近检查)
迟滞: 进入 INSPECT 后需 d0 > stop_dist+AVOID_RESUME_HYST 才恢复。
stop_dist/slow_dist/body_front 可用构造参数覆盖(默认 = 避障常量), 供"病害在
拱顶/墙面、车只能停在数米外拍摄"的场景(见 __init__ 注释)按可拍摄距离设阈值。
轻转向: |az|>AZ_DEAD_CENTER_DEG 时 yaw=clip(DEC_FACE_KP*az, ±AVOID_YAW_MAX)。

纯 numpy, 无 ROS/torch, 可单测。与几何避障共用 tunnel_params 的 AVOID_*/AZ_* 阈值。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from localization.tunnel_params import (  # noqa: E402
    AVOID_BODY_FRONT, AVOID_RESUME_HYST, AVOID_SLOW_DIST, AVOID_STOP_DIST,
    AVOID_V_CRUISE, AVOID_YAW_MAX, AZ_ABEAM_DEG, AZ_DEAD_CENTER_DEG,
    DEC_CONF_THR, DEC_FACE_KP, DEC_V_CREEP,
)


@dataclass
class Target:
    """路径决策的目标: 定位链产出的一个检测 + 它在车体系下的位置。"""
    cls: int            # 类 id(各支局部)
    conf: float         # 检测置信度
    distance: float     # 到目标的欧氏距离(m, 车体系)
    azimuth_deg: float  # 方位角(°), 左正右负, 与 find_obstacles 同向
    class_name: str = ''  # 可选人类可读名称


@dataclass
class Decision:
    """解码结果: 单一状态 + 目标指令。"""
    state: str          # FREE / APPROACH / INSPECT
    target_speed: float  # 前进速度(m/s)
    target_yaw_rate: float  # 转向角速度(rad/s), 左正右负
    reason: str
    front: Target | None   # 当前驱动决策的前方目标(None 表示无)


class DecisionDecoder:
    """融合检测目标 -> 接近-检查行驶指令(差速车: 前进+原地转, 无法横移)。"""

    def __init__(self, conf_thr: dict | None = None, face_kp: float | None = None,
                 class_names: list[str] | None = None,
                 stop_dist: float | None = None, slow_dist: float | None = None,
                 body_front: float | None = None):
        self.conf_thr = conf_thr if conf_thr is not None else DEC_CONF_THR
        self.face_kp = face_kp if face_kp is not None else DEC_FACE_KP
        self.class_names = class_names or []
        # 检查距离三件套可覆盖, 默认沿用避障常量(正前方障碍 -> 车头前 0.5m 刹停)。
        # 病害贴在拱顶/墙面上部时(如隧道裂缝), 车在地面最近也只能到数米外, 需按
        # "可拍摄距离"给更大的 stop_dist/slow_dist; 否则恒为 APPROACH, INSPECT 永不触发。
        self.body_front = AVOID_BODY_FRONT if body_front is None else body_front
        self.stop_dist = AVOID_STOP_DIST if stop_dist is None else stop_dist
        self.slow_dist = AVOID_SLOW_DIST if slow_dist is None else slow_dist
        assert self.slow_dist > self.stop_dist, "slow_dist 必须大于 stop_dist"
        self._stop_latched = False       # 停车锁存(迟滞, 防抖)

    def _thr(self, cls: int) -> float:
        return self.conf_thr.get(cls, 0.25)

    def _pick_front(self, targets: list[Target]) -> Target | None:
        """首个 conf>=阈 且 |az|<AZ_ABEAM(前方而非侧掠) 的目标, 按距离升序。"""
        cands = [t for t in targets
                 if t.conf >= self._thr(t.cls) and abs(t.azimuth_deg) < AZ_ABEAM_DEG]
        if not cands:
            return None
        cands.sort(key=lambda t: t.distance)
        return cands[0]

    def decide(self, targets: list[Target], v_now: float = 0.0,
               nav_yaw: float = 0.0) -> Decision:
        """解码。v_now/nav_yaw 供实时闭环用; 离线验证缺省即可(当前未用于决策)。"""
        front = self._pick_front(targets)
        if front is None:
            self._stop_latched = False
            return Decision('FREE', AVOID_V_CRUISE, 0.0, '无前方目标', None)

        name = front.class_name or str(front.cls)
        d0 = front.distance - self.body_front    # 真净距
        az = front.azimuth_deg

        # 迟滞: 停稳后需 d0 > STOP+HIST 才恢复, 避免在阈值附近来回抖
        if self._stop_latched and d0 > self.stop_dist + AVOID_RESUME_HYST:
            self._stop_latched = False
        if d0 <= self.stop_dist or self._stop_latched:
            self._stop_latched = True
            return Decision('INSPECT', 0.0, 0.0,
                            f'停车检查({name} d0={d0:.2f}m az={az:.0f}°)', front)

        # APPROACH: 减速斜坡 + 轻转向让目标居中
        if d0 < self.slow_dist:
            speed = AVOID_V_CRUISE * (d0 - self.stop_dist) / (self.slow_dist - self.stop_dist)
        else:
            speed = AVOID_V_CRUISE
        # 最低爬行: 斜坡在 d0->STOP 时归 0, 纯运动学会渐进逼近却跨不进 STOP(死锁)。
        # 给 DEC_V_CREEP 下限, 车才顶到触发点锁存 INSPECT(真实车有最低起爬速度)。
        speed = float(min(AVOID_V_CRUISE, max(DEC_V_CREEP, speed)))
        yaw = 0.0
        if abs(az) > AZ_DEAD_CENTER_DEG:
            yaw = float(np.clip(self.face_kp * az, -AVOID_YAW_MAX, AVOID_YAW_MAX))
        return Decision('APPROACH', speed, yaw,
                        f'接近检查({name} d0={d0:.2f}m az={az:.0f}°)', front)

    def reset(self) -> None:
        self._stop_latched = False


# ================================================================
# L1 合成单测(离经不起真数据时也能验证状态机正确性)
# ================================================================
def _t(cls=0, conf=0.9, d=1.0, az=0.0, name=''):
    return Target(cls, conf, d, az, name)


def _selftest() -> int:
    dec = DecisionDecoder()
    fails = []
    # 1. 无目标 -> FREE/巡航
    d = dec.decide([])
    fails += [('无目标→FREE', d.state == 'FREE'),
              ('无目标→巡航速度', d.target_speed == AVOID_V_CRUISE),
              ('无目标→yaw0', d.target_yaw_rate == 0.0)]
    # 2. 远目标(3.0m, 正前) -> APPROACH/全速
    d = dec.decide([_t(d=3.0, az=0)])
    fails += [('远目标→APPROACH', d.state == 'APPROACH'),
              ('远目标→全速(>SLOW)', d.target_speed == AVOID_V_CRUISE),
              ('正前→不转', d.target_yaw_rate == 0.0)]
    # 3. 中距离(1.0m, az20) -> APPROACH/减速斜坡+轻转向
    d = dec.decide([_t(d=1.0, az=20)])
    exp = AVOID_V_CRUISE * (1.0 - AVOID_BODY_FRONT - AVOID_STOP_DIST) / (AVOID_SLOW_DIST - AVOID_STOP_DIST)
    fails += [('中距→APPROACH', d.state == 'APPROACH'),
              ('中距→斜坡减速', abs(d.target_speed - exp) < 1e-6),
              ('偏左→左转朝目标(正yaw, 居中而非避让)', d.target_yaw_rate > 0.0)]
    # 4. 近目标(0.6m <= STOP) -> INSPECT/停
    d = dec.decide([_t(d=0.6, az=0)])
    fails += [('近距→INSPECT', d.state == 'INSPECT'),
              ('近距→停', d.target_speed == 0.0)]
    # 5. 迟滞: 停后 d0 回升到 0.95(STOP+HIST=0.80 以上) 才恢复
    d = dec.decide([_t(d=0.95, az=0)])      # d0=0.725<0.80 仍锁存
    fails += [('迟滞沿→仍INSPECT', d.state == 'INSPECT')]
    d = dec.decide([_t(d=1.05, az=0)])      # d0=0.825>0.80 恢复
    fails += [('迟滞沿→恢复APPROACH', d.state == 'APPROACH')]
    # 6. 侧掠(|az|≈85 > 80) -> 不计入前方 -> FREE
    d = dec.decide([_t(d=1.0, az=85)])
    fails += [('侧掠→FREE', d.state == 'FREE')]
    # 7. 低置信(低于阈值)非触发
    d = dec.decide([_t(d=1.0, az=0, conf=0.1)])
    fails += [('低conf→FREE', d.state == 'FREE')]
    # 8. 异物类(d分类任意), 近距 -> INSPECT(检查而非避让; 无STEER分支即天然不避让)
    d = dec.decide([_t(cls=0, d=0.6, az=0, name='equip_intrusion')])
    fails += [('异物→INSPECT而非避让', d.state == 'INSPECT')]

    bad = [name for name, ok in fails if not ok]
    print('===== DecisionDecoder L1 单测 =====')
    for name, ok in fails:
        print(('  [PASS] ' if ok else '  [FAIL] ') + name)
    print(f'>>> {len(fails) - len(bad)}/{len(fails)} 通过' + (' 全部通过' if not bad else f' 失败: {bad}'))
    return 0 if not bad else 1


if __name__ == '__main__':
    raise SystemExit(_selftest())
