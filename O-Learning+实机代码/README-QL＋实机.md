# FrozenLake Q-learning（仿真 + 实机映射）

本目录提供 Q-learning 在 FrozenLake 环境的仿真训练与评估（4x4 / 8x8，确定/随机），并给出将学到的策略映射到真实机器人（串口）的示例。

## 仿真部分

### 文件说明
- `QL4x4.py`：在 4x4 地图训练/测试 Q-learning。
- `QL8x8.py`：在 8x8 地图训练/测试 Q-learning。

### 依赖
- Python 3.10+
- `gymnasium`
- `numpy`
- `matplotlib`

示例安装：
```bash
pip install gymnasium numpy matplotlib
```

### 运行与流程
```bash
python QL4x4.py
python QL8x8.py
```
流程：训练 → 绘制训练成功率曲线 → 贪心策略测试，输出成功率与平均步数。

### 可调参数（脚本内）
- `NUM_EPISODES`：训练轮数
- `LEARNING_RATE`：学习率
- `DISCOUNT_FACTOR`：折扣因子
- `EPSILON_DECAY` / `EPSILON_MIN`：探索退火
- `IS_SLIPPERY`：是否启用打滑
- `MAX_STEPS_PER_EPISODE`：步数上限

### 仿真结果（1000 轮测试）
| 地图 | 环境模式 | 步数上限 | 理论成功率 | 实际成功率 | 平均步数 |
| --- | --- | --- | --- | --- | --- |
| 4×4 | Deterministic | 200  | 100.00% | 100.00% | 6.00 |
| 4×4 | Stochastic    | 200  | 82.35%  | 81.33%  | 47.82 |
| 8×8 | Deterministic | 1000 | 100.00% | 100.00% | 14.00 |
| 8×8 | Stochastic    | 1000 | 100.00% | 100.00% | 131.45 |

## 实机部分

### 文件说明
- `Q-learning_slippery.py`：在打滑环境训练后，将动作映射为串口指令控制机器人。
- `Q-learning_noslippery.py`：在不打滑环境训练，并进行实机动作控制与可视化测试。

### 运行
```bash
# 打滑环境
python Q-learning_slippery.py
# 不打滑环境
python Q-learning_noslippery.py
```

流程：训练 FrozenLake 策略 → ε-greedy 逐步转向贪心 → 测试阶段 `render_mode="human"` → 将动作映射为串口控制命令。

### 动作与机器人控制映射
| 动作 | 含义 |
| ---- | ---- |
| 0 | LEFT（左移） |
| 1 | DOWN（后退） |
| 2 | RIGHT（右移） |
| 3 | UP（前进） |

### 可调参数（脚本内）
- `NUM_EPISODES`：训练轮数
- `LEARNING_RATE`：学习率
- `DISCOUNT_FACTOR`：折扣因子
- `EPSILON_DECAY`：探索衰减速度
- `IS_SLIPPERY`：是否启用打滑
- 串口号（如 `COM6`）需与实际设备一致

## 备注
- 建议先在不打滑环境调试，再切换到打滑环境。
- 受随机性与采样噪声影响，训练曲线可能有波动；固定随机种子可提升复现性。
