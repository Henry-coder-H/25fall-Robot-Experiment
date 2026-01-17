# FrozenLake DQN 实验

本仓库聚焦于 FrozenLake 环境的 DQN 实现，包含 4x4（标准奖励）和 8x8（自定义奖励）两份脚本，提供训练/评估与损失曲线可视化。

## 文件结构
- `DQN.py`：4x4 FrozenLake，标准奖励，DQN（经验回放 + target 网络），固定随机种子，输出损失曲线与测试成功率/平均步数。
- `DQN_reward.py`：8x8 FrozenLake，自定义奖励（终点正奖、陷阱负奖、步长惩罚），DQN 训练与损失曲线，固定随机种子。
- `figures/` 与若干 `.png`：训练或评估的可视化结果示例。
- 其他：实验报告相关文档或示例文件。

## 环境依赖
- Python 3.10+（示例在 3.12 下运行）
- 必需：`torch`、`gymnasium`、`numpy`、`matplotlib`

安装示例（CPU 版）：
```bash
pip install torch gymnasium numpy matplotlib
```
如需 GPU 版 PyTorch，请按官网指令选择匹配 CUDA 版本安装。

## 运行方式
在仓库根目录执行（确保已安装依赖）：

- 4x4 DQN（默认训练 8000 回合，测试 1000 局）：
```bash
python DQN.py
```
训练过程每 500 回合打印进度，结束后输出测试成功率、平均步数，并生成 `loss_curve_4x4.png`。

- 8x8 自定义奖励 DQN：
```bash
python DQN_reward.py
```
同样打印训练进度，测试 1000 局的成功率与平均步数，生成 `loss_curve_8x8.png`。

## 主要特性与超参数（DQN 系列）
- 经验回放（容量 `REPLAY_CAPACITY`）、预热阈值（`MIN_REPLAY_SIZE`）
- target 网络同步频率（`TARGET_UPDATE_FREQUENCY`）
- ε-贪心探索：`EPSILON_START` → `EPSILON_END`，按 `EPSILON_DECAY` 衰减
- 学习率与折扣因子：`LEARNING_RATE`、`DISCOUNT_FACTOR`
- 训练/评估随机种子：`SEED`（含 `env.reset` 与 `action_space.seed`）
- 输出：训练进度日志、测试成功率/平均步数、损失曲线（原始 + 平滑）

## DQN 实验结果（1000 轮测试）

| 地图 | 环境模式 | 奖励机制 | 测试成功率 | 平均步数 |
| --- | --- | --- | --- | --- |
| 4x4 | Deterministic | No Reward | 100.00% | 6.00 |
| 4x4 | Deterministic | With Reward | 100.00% | 6.00 |
| 8x8 | Deterministic | No Reward | 100.00% | 14.00 |
| 8x8 | Deterministic | With Reward | 100.00% | 14.00 |
| 4x4 | Stochastic | No Reward | 75.00% | 43.91 |
| 4x4 | Stochastic | With Reward | 75.20% | 44.30 |
| 8x8 | Stochastic | No Reward | 52.20% | 82.40 |
| 8x8 | Stochastic | With Reward | **69.80%** | **134.54** |

## 可视化输出
训练结束会生成损失曲线 PNG（`loss_curve_4x4.png`、`loss_curve_8x8.png` 等）保存在仓库根目录。可根据需要调整超参数或奖励设计，建议固定随机种子以保证可复现性。
