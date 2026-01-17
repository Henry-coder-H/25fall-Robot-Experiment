# 深度强化学习在无人机飞行控制上的应用（FrozenLake 系列）

本仓库汇总了多种强化学习算法在 Gymnasium FrozenLake 环境（4×4 / 8×8，确定/随机）上的实现与实验结果，覆盖表格型 Q-learning、SARSA、DQN 及其变体（Nature DQN、Double DQN）、CSPMP，以及仿真到实机的示例。

## 背景概览（简述）
- 强化学习要素：状态 S、动作 A、即时奖励 R、策略 π(a|s)、价值函数 vπ/qπ、折扣 γ、探索率 ε、转移 P(s'|s,a)。
- Q-learning：无模型价值迭代，维护 Q 表，ε-greedy 平衡探索/利用。
- DQN：用神经网络近似 Q，配合经验回放与 target 网络；Nature/Double DQN 进一步减小目标估计偏差。
- 任务：在有/无打滑环境中学习从起点到终点并避开坑洞的策略，可扩展奖励塑形、随机地图等。

## 仓库结构与链接
- [DQN](DQN/README.md)：4×4 DQN（标准奖励）与 8×8 自定义奖励 DQN。
- [Nature DQN & Double DQN](NatureDQN&DoubleDQN/README.md)：Nature DQN 与 Double DQN。
- [SARSA](SARSA/README.md)：SARSA（4×4 / 8×8，确定/随机，含扫网格示例）。
- [CSPMP](CSPMP/README.md)：CSPMP（FHR/IHR，自动切换滑动/非滑动转移建模）。
- [Q-learning 仿真与实机](O-Learning+实机代码/README.md)：Q-learning 仿真与实机串口映射示例。
- `frozenlake.py` 等：表格 Q-learning 进阶框架（地图生成、奖励塑形、可视化/GIF）。
- `figures/`：部分可视化示例输出。

## 全局依赖
- Python 3.10+（部分脚本在 3.12 验证）
- 核心：`gymnasium`、`numpy`、`torch`、`matplotlib`
- 可选：`pandas`、`seaborn`、`imageio`、`tqdm`

示例安装（CPU）：
```bash
pip install gymnasium numpy matplotlib
pip install torch          # 深度方法
pip install pandas seaborn imageio tqdm  # 如需扩展可视化/进度条
```
> GPU 版 PyTorch 请按官网选择与 CUDA 匹配的轮子。

## 结果总览（1000 轮测试/关键指标）

### DQN
| 地图 | 环境模式 | 奖励机制 | 测试成功率 | 平均步数 |
| --- | --- | --- | --- | --- |
| 4×4 | Deterministic | No Reward | 100.00% | 6.00 |
| 4×4 | Deterministic | With Reward | 100.00% | 6.00 |
| 8×8 | Deterministic | No Reward | 100.00% | 14.00 |
| 8×8 | Deterministic | With Reward | 100.00% | 14.00 |
| 4×4 | Stochastic | No Reward | 75.00% | 43.91 |
| 4×4 | Stochastic | With Reward | 75.20% | 44.30 |
| 8×8 | Stochastic | No Reward | 52.20% | 82.40 |
| 8×8 | Stochastic | With Reward | **69.80%** | **134.54** |

### Nature DQN
| 地图 | 环境模式 | 奖励机制 | 测试成功率 | 平均步数 |
| --- | --- | --- | --- | --- |
| 4×4 | Deterministic | No Reward | 100.00% | 6.00 |
| 4×4 | Deterministic | With Reward | 100.00% | 6.00 |
| 8×8 | Deterministic | No Reward | 100.00% | 14.00 |
| 8×8 | Deterministic | With Reward | 100.00% | 14.00 |
| 4×4 | Stochastic | No Reward | 81.40% | 47.27 |
| 4×4 | Stochastic | With Reward | 83.40% | 48.10 |
| 8×8 | Stochastic | No Reward | 85.40% | 108.32 |
| 8×8 | Stochastic | With Reward | **94.70%** | **107.19** |

### Double DQN
| 地图 | 环境模式 | 奖励机制 | 测试成功率 | 平均步数 |
| --- | --- | --- | --- | --- |
| 4×4 | Deterministic | No Reward | 100.00% | 6.00 |
| 4×4 | Deterministic | With Reward | 100.00% | 6.00 |
| 8×8 | Deterministic | No Reward | 100.00% | 14.00 |
| 8×8 | Deterministic | With Reward | 100.00% | 14.00 |
| 4×4 | Stochastic | No Reward | 83.40% | 48.10 |
| 4×4 | Stochastic | With Reward | 84.40% | 89.80 |
| 8×8 | Stochastic | No Reward | 85.10% | 96.75 |
| 8×8 | Stochastic | With Reward | **97.30%** | **112.19** |

### SARSA
**Deterministic**
| 地图 | 训练轮数 | 步数上限 | α | γ | ε | 测试成功率 | 平均步数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4×4 | 2000 | 100 | 0.2 | 0.99 | 1.0 | 100% | 6 |
| 8×8 | 8000 | 200 | 0.2 | 0.99 | 1.0 | 100% | 14 |

**Stochastic（最优）**
| 地图 | 训练轮数 | 最大步数 | α | γ | ε | 测试成功率 | 平均步数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4×4 | 60000  | 100 | 0.05 | 0.99 | 0.85 | 82% | 37.10 |
| 8×8 | 200000 | 200 | 0.2  | 0.99 | 0.85 | 74% | 67.38 |

### CSPMP
**确定性**
| 地图 | 链路 | 理论成功率 | 早停回合 | 测试成功率 | 成功平均步数 |
| --- | --- | --- | --- | --- | --- |
| 4×4 | FHR | 100.00% | 123.00 | 100.00% | 6.00 |
| 4×4 | IHR | 100.00% | 61.00  | 100.00% | 6.00 |
| 8×8 | FHR | 100.00% | 252.00 | 100.00% | 14.00 |
| 8×8 | IHR | 100.00% | 190.00 | 100.00% | 14.00 |

**随机打滑**
| 地图 | 链路 | 理论成功率 | 测试成功率 | 理论平均步数 | 测试平均步数 |
| --- | --- | --- | --- | --- | --- |
| 4×4 | FHR | 74.42%  | 74.40%  | 38.81 | 38.56 |
| 4×4 | IHR | 82.35%  | 82.50%  | 49.00 | 49.22 |
| 8×8 | FHR | 91.32%  | 92.30%  | 95.02 | 95.22 |
| 8×8 | IHR | 100.00% | 100.00% | 116.97 | 118.48 |

### Q-learning（仿真）
| 地图 | 环境模式 | 步数上限 | 理论成功率 | 实际成功率 | 平均步数 |
| --- | --- | --- | --- | --- | --- |
| 4×4 | Deterministic | 200  | 100.00% | 100.00% | 6.00 |
| 4×4 | Stochastic    | 200  | 82.35%  | 81.33%  | 47.82 |
| 8×8 | Deterministic | 1000 | 100.00% | 100.00% | 14.00 |
| 8×8 | Stochastic    | 1000 | 100.00% | 100.00% | 131.45 |

## 使用提示
- 训练日志与损失曲线（PNG）会输出到对应目录；部分脚本默认每隔固定回合打印进度。
- 滑动环境具有随机性，建议固定随机种子并适当增大训练轮数/步数上限。
- 可视化与 GIF 生成需要 `matplotlib`/`imageio` 支持；实机示例需正确配置串口号。

## 参考与致谢
基于“深度强化学习在无人机飞行控制上的应用”实验手册：介绍 RL 要素、FrozenLake 任务、DQN 及其改进（Nature/Double DQN），并给出仿真与实机验收要求。欢迎在此基础上调整超参数、奖励设计或扩展更复杂的算法与地图。