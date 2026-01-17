# FrozenLake Nature DQN & Double DQN

本目录提供 FrozenLake 环境的 Nature DQN 与 Double DQN 两种实现，支持 4x4/8x8 地图、确定/随机模式，以及可选的奖励塑形，输出训练日志、测试统计与损失曲线。

## 文件结构
- `MainCode/NatureDQN_no_reward_avgsteps_py38.py`：Nature DQN，标准奖励，可在脚本内开关奖励塑形。
- `MainCode/DoubleDQN_no_reward_with_plot.py`：Double DQN，标准奖励。
- `MainCode/DoubleDQN_with_reward_with_plot.py`：Double DQN，自定义奖励（终点正奖、陷阱负奖、步长惩罚）。
- 训练产物：损失曲线 `.png`、日志输出等。

## 环境依赖
- Python 3.10+（示例在 3.12 下运行）
- 必需：`torch`、`gymnasium`、`numpy`、`matplotlib`

安装示例：
```bash
pip install torch gymnasium numpy matplotlib
```

## 运行方式
在本目录下执行（确保依赖已安装）：
```bash
# Nature DQN（标准奖励；脚本内可切换 USE_REWARD_SHAPING）
python MainCode/NatureDQN_no_reward_avgsteps_py38.py

# Double DQN（标准奖励）
python MainCode/DoubleDQN_no_reward_with_plot.py

# Double DQN（自定义奖励）
python MainCode/DoubleDQN_with_reward_with_plot.py
```

> 具体超参数（地图大小、是否滑动、训练轮次、奖励设置等）可在脚本顶部的 Hyperparameters 区修改。

## 主要特性与超参数
- 经验回放、target 网络同步频率、ε-贪心探索（`EPSILON_START/END/DECAY`）
- 学习率与折扣因子：`LEARNING_RATE`、`DISCOUNT_FACTOR`
- 批量大小、经验池容量、预热阈值：`BATCH_SIZE`、`REPLAY_CAPACITY`、`MIN_REPLAY_SIZE`
- 地图/模式：`MAP_NAME`（4x4/8x8）、`IS_SLIPPERY`（确定/随机）
- 奖励塑形：goal/hole/step 奖励可在脚本中配置

## 备注
- `NatureDQN_no_reward_avgsteps_py38.py` 为主用脚本，可在内部开关 `USE_REWARD_SHAPING`；旧的带 reward 版本仅作为历史备份。

## Nature DQN 实验结果（1000 轮测试）

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

## Double DQN 实验结果（1000 轮测试）

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

## 可视化输出
运行脚本会在当前目录生成损失曲线 PNG（例如 `double_dqn_with_reward_loss.png` 等），日志中包含训练进度与测试成功率/平均步数。可根据需要调参并复现实验，建议固定随机种子以保证可重复性。
