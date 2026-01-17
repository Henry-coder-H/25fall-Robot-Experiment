# FrozenLake SARSA 实验

本目录提供基于 FrozenLake 的 SARSA 实现，支持 4x4/8x8 地图与确定/随机两种模式，可单次训练或进行超参数扫网格，并输出 Q 表。

## 文件结构
- `train_sarsa.py`：SARSA 训练与评估脚本，支持 4x4/8x8、det/stochastic、单次训练与扫网格。
- （训练产物）`*.npy`：保存的 Q 表；可自定义输出目录。

## 环境依赖
- Python 3.8+（推荐 3.10+）
- `gymnasium`
- `numpy`

安装示例：
```bash
pip install gymnasium numpy
```

## 运行方式
在 `SARSA` 目录下执行：
```bash
# 单次训练（确定/随机 + 4x4/8x8）
python train_sarsa.py --map 4x4 --mode det
python train_sarsa.py --map 4x4 --mode sto
python train_sarsa.py --map 8x8 --mode det
python train_sarsa.py --map 8x8 --mode sto

# 常用参数
--episodes N
--alpha A --gamma G --epsilon E
--epsilon_decay D --epsilon_min M
--eval_episodes K
```

### 扫网格示例
批量跑一组超参并分别保存 Q 表：
```bash
python train_sarsa.py --map 4x4 --mode sto --sweep --outdir sweep_4x4_sto
```

## 备注
- 4x4 默认最大步数 100；8x8 默认最大步数 200。
- Q 表以 `.npy` 保存到指定路径。

## SARSA 实验结果（1000 轮测试）

### Deterministic 模式
| 地图 | 训练轮数 | 步数上限 | α | γ | ε | 测试成功率 | 平均步数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4×4 | 2000 | 100 | 0.2 | 0.99 | 1.0 | 100% | 6 |
| 8×8 | 8000 | 200 | 0.2 | 0.99 | 1.0 | 100% | 14 |

### Stochastic 模式（最优结果）
| 地图 | 训练轮数 | 最大步数 | α | γ | ε | 测试成功率 | 平均步数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4×4 | 60000  | 100 | 0.05 | 0.99 | 0.85 | 82% | 37.10 |
| 8×8 | 200000 | 200 | 0.2  | 0.99 | 0.85 | 74% | 67.38 |
