# 实测结果索引

| 文件 | 内容 |
|---|---|
| `fusion_ab_compare.json` | 图像单支、融合、无点云分支在同一评测集上的 AP 对比 |
| `range_qfl_eval.json` | 原版 v8 loss 与 Range-QFL 的留出集/随机集评测、分桶召回和置信度统计 |
| `closed_loop_summary.json` | 离线闭环回放的步数、里程、检查步与巡航占比 |
| `closed_loop_crack.png` | 离线闭环轨迹可视化 |
| `early_fusion/` | 早期融合阶段的分领域评测、置信度搜索与闭环对照结果 |
| `yolo11l_results.csv` | 2D 检测训练曲线 |
| `yolo11l_train_args.yaml` | 2D 检测训练参数快照 |

说明：三领域 2D 检测的 P@0.93 / R@0.98 来自配套检测分支；本目录保留的是多模态融合、Range-QFL 与闭环定位相关结果。
