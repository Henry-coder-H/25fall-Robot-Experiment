# FrozenLake CSPMP

本目录提供 FrozenLake 环境下的 CSPMP 实现，单文件脚本 `frozenlake.py` 支持有限步 (FHR) 与无限步 (IHR) 两套训练与评估流程，可自动根据是否滑动切换经验转移建模，并输出训练/评估统计与可视化。

## 原理与流程要点
- 单脚本同时覆盖 FHR / IHR 流程，依据 `IS_SLIPPERY` 自动切换经验转移建模：
  - 滑动：使用 Dirichlet 平滑的转移估计。
  - 不滑动：使用极大似然估计。
- 提供早停、步数上限、计划刷新频率、epsilon 退火等开关，便于稳定训练。
- 不保存策略文件，主要输出为统计日志与可视化图。

## 环境依赖
- Python 3.10+
- `gymnasium`
- `numpy`
- `matplotlib`
- `tqdm`

安装示例：
```bash
pip install gymnasium numpy matplotlib tqdm
```

## 运行方式
在 `CSPMP` 目录下执行：
```bash
python frozenlake.py
```

主要开关在脚本顶部：
- `IS_SLIPPERY`：是否滑动（决定使用 Dirichlet 平滑或极大似然）。
- `RUN_MAP_SIZES`：要跑的地图尺寸（如 4、8）。
- `USE_DEFAULT_MAP`：是否使用内置地图；若自定义，修改 `DEFAULT_DESC_4X4` / `DEFAULT_DESC_8X8`。

关键参数集中在 `get_mode_config` 中（步数上限、训练回合数、评估回合数、epsilon 退火、早停、可视化开关等）。

## 可视化输出
脚本会在当前目录生成 `figures_det` 或 `figures_stoch`：
- 训练曲线：如 `4x4_det_training_curves.png`、`8x8_stoch_training_curves.png`
- 访问热力图：如 `4x4_det_visited_states.png`、`8x8_stoch_visited_states.png`

## 备注
- 环境存在随机性和采样噪声，虽固定了随机种子，训练回报与收敛回合在多次试验间仍可能波动。
- 如需自定义地图，修改 `DEFAULT_DESC_4X4` / `DEFAULT_DESC_8X8`。

## CSPMP 实验结果（1000 轮测试）

### 确定性环境
| 地图 | 链路 | 理论成功率 | 早停回合 | 测试成功率 | 成功平均步数 |
| --- | --- | --- | --- | --- | --- |
| 4×4 | FHR | 100.00% | 123.00 | 100.00% | 6.00 |
| 4×4 | IHR | 100.00% | 61.00  | 100.00% | 6.00 |
| 8×8 | FHR | 100.00% | 252.00 | 100.00% | 14.00 |
| 8×8 | IHR | 100.00% | 190.00 | 100.00% | 14.00 |

### 随机打滑环境
| 地图 | 链路 | 理论成功率 | 测试成功率 | 理论平均步数 | 测试平均步数 |
| --- | --- | --- | --- | --- | --- |
| 4×4 | FHR | 74.42%  | 74.40%  | 38.81 | 38.56 |
| 4×4 | IHR | 82.35%  | 82.50%  | 49.00 | 49.22 |
| 8×8 | FHR | 91.32%  | 92.30%  | 95.02 | 95.22 |
| 8×8 | IHR | 100.00% | 100.00% | 116.97 | 118.48 |
