# -*- coding: utf-8 -*-
"""4周任务排期甘特图(2026-08-17 ~ 2026-09-15 提交)  -> docs/甘特图.png
依赖: matplotlib (含中文字体 Noto Sans CJK SC)。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from datetime import date

_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT):
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

# (任务, 开始, 结束, 阶段)
TASKS = [
    ("三领域 YOLO 训练（远程 7×V100）",   date(2026, 8, 17), date(2026, 8, 30), "精度分"),
    ("定位链核心代码（本模块）",          date(2026, 8, 17), date(2026, 8, 23), "定位"),
    ("报告：系统架构章节",                date(2026, 8, 17), date(2026, 8, 23), "报告"),
    ("Gazebo 隧道跑通 + 配对数据导出",    date(2026, 8, 24), date(2026, 8, 30), "仿真"),
    ("定位链仿真闭环验证（真值外参）",     date(2026, 8, 24), date(2026, 9, 6),  "定位"),
    ("点云属性图增强检测实验",            date(2026, 8, 31), date(2026, 9, 6),  "仿真"),
    ("FAST-LIO2 里程定位 + TEB 避障",    date(2026, 8, 31), date(2026, 9, 6),  "仿真"),
    ("报告：融合策略 + 空间映射章节",      date(2026, 8, 31), date(2026, 9, 6),  "报告"),
    ("8000 字报告完善 + 实证材料",        date(2026, 9, 7),  date(2026, 9, 13), "报告"),
    ("代码打包 + 演示视频 + 点云效果图",  date(2026, 9, 7),  date(2026, 9, 13), "报告"),
]

COLOR = {"精度分": "#4c8bf5", "定位": "#34c08a", "仿真": "#9d6fe0", "报告": "#f0a24a"}

fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
for i, (name, s, e, phase) in enumerate(TASKS):
    ax.barh(i, (e - s).days + 1, left=mdates.date2num(s),
            color=COLOR[phase], edgecolor="white", height=0.62, label=phase if i == 0 else "")
    ax.text(mdates.date2num(e) + 0.4, i, phase, va="center", fontsize=10, color="#666")

# 周分隔线
for d in [date(2026, 8, 24), date(2026, 8, 31), date(2026, 9, 7), date(2026, 9, 14)]:
    ax.axvline(mdates.date2num(d), color="#bbb", lw=1, ls="--")

ax.set_yticks(range(len(TASKS)))
ax.set_yticklabels([t[0] for t in TASKS], fontsize=11)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.set_xlim(mdates.date2num(date(2026, 8, 16)), mdates.date2num(date(2026, 9, 16)))
ax.set_ylim(-0.6, len(TASKS) - 0.4)
ax.set_xlabel("日期（2026年8月17日 → 9月15日提交）", fontsize=12)
ax.set_title("地铁多领域病害综合智能巡检 · 4周任务排期（甘特图）", fontsize=15, pad=12)
# 提交里程碑
ax.axvline(mdates.date2num(date(2026, 9, 15)), color="#d33", lw=2)
ax.text(mdates.date2num(date(2026, 9, 15)), len(TASKS) - 0.3, "9/15 提交",
        color="#d33", fontsize=12, ha="center", fontweight="bold")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR.values()]
ax.legend(handles, COLOR.keys(), ncol=4, loc="lower right", fontsize=10, frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.savefig(os.path.join(os.path.dirname(__file__), "甘特图.png"), dpi=130)
print("已生成: docs/甘特图.png")
