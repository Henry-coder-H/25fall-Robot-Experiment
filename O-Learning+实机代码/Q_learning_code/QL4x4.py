import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# ============================================================
# 1) 可复现性（Reproducibility）
# ============================================================
# 统一设置随机种子，确保每次运行结果尽量一致
SEED = 102
random.seed(SEED)              # Python 内置随机
np.random.seed(SEED)           # NumPy 的全局随机
rng = np.random.default_rng(SEED)  # 推荐：NumPy 新版随机生成器（更可控、更现代）

# ============================================================
# 2) 超参数（Hyperparameters）
# ============================================================
NUM_EPISODES = 20000           # 训练回合数
MAX_STEPS_PER_EPISODE = 200    # 每回合最大步数（防止一直走不结束）

LEARNING_RATE = 0.05           # Q-learning 学习率 α
DISCOUNT_FACTOR = 0.95         # 折扣因子 γ

# ε-greedy 探索参数
EPSILON = 1.0                  # 初始探索率（完全探索）
EPSILON_DECAY = 0.9995         # 每回合衰减比例
EPSILON_MIN = 0.001            # 最低探索率（不至于完全不探索）

IS_SLIPPERY = True             # FrozenLake 是否“打滑”（动作有随机偏移）

# ============================================================
# 3) 训练过程的统计记录（Logging / Plotting）
# ============================================================
WINDOW_SIZE = 200              # 统计窗口大小：看最近 200 局的平均表现
EVAL_INTERVAL = 50             # 每 50 局更新一次曲线点

episode_returns = []           # 每回合累计回报（可能是 -1/0/1 或其他）
episode_success = []           # 是否成功到达终点：成功记 1，否则 0
episode_hole = []              # 是否掉进洞：掉洞记 1，否则 0

# 用于画图的 x、y 序列（每隔 EVAL_INTERVAL 记录一次）
x = []
success_y = []                 # 成功率（%）
return_y = []                  # 平均回报（最近 WINDOW_SIZE 局）
hole_y = []                    # 掉洞率（%）

# ============================================================
# 4) 创建训练环境（Create env）
# ============================================================
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=IS_SLIPPERY,
    reward_schedule=(1, -1, 0),  # (到达终点, 掉洞, 冰面普通格) - 你的自定义奖励
    max_episode_steps=-1         # 禁用 TimeLimit（但你自己又手动限制了 MAX_STEPS_PER_EPISODE）
)

# 给环境的 RNG 设置 seed（尤其是 slippery 情况下，环境内部也有随机性）
env.reset(seed=SEED)

# 状态空间、动作空间大小
n_states = env.observation_space.n
n_actions = env.action_space.n

# Q 表初始化：形状 [状态数, 动作数]
Q = np.zeros((n_states, n_actions))

# ============================================================
# 5) Q-learning 训练主循环
# ============================================================
for episode in range(NUM_EPISODES):
    # reset() 返回：初始状态 state，以及 info 字典
    state, info = env.reset()

    total_reward = 0.0         # 本回合累计奖励
    final_reward = 0.0         # 回合结束那一步的奖励，用于判断“成功/掉洞”

    for step in range(MAX_STEPS_PER_EPISODE):
        # ----------------------------
        # ε-greedy 选动作（用 rng 保证可复现）
        # ----------------------------
        if rng.random() < EPSILON:
            # 探索：随机动作
            action = int(rng.integers(n_actions))
        else:
            # 利用：选择当前 Q 最大的动作
            action = int(np.argmax(Q[state]))

        # 与环境交互一步
        # step() 返回：next_state, reward, terminated, truncated, info
        next_state, reward, terminated, truncated, info = env.step(action)

        # done = True 表示回合结束（终止 or 截断）
        # terminated：达到终止条件（如到终点或掉洞）
        # truncated：达到时间/步数上限等外部限制（这里一般不靠它，因为 max_episode_steps=-1）
        done = terminated or truncated

        # ----------------------------
        # TD 目标（TD target）
        # ----------------------------
        # 若已终止：目标值就是 reward（不再 bootstrap）
        # 否则：reward + γ * max_a' Q(s', a')
        if done:
            td_target = reward
        else:
            td_target = reward + DISCOUNT_FACTOR * np.max(Q[next_state])

        # ----------------------------
        # Q-learning 更新公式
        # Q(s,a) ← Q(s,a) + α * (td_target - Q(s,a))
        # ----------------------------
        Q[state, action] += LEARNING_RATE * (td_target - Q[state, action])

        # 状态推进，累计回报
        state = next_state
        total_reward += reward

        # 若回合结束，记录最后一步奖励并跳出
        if done:
            final_reward = reward
            break

    # ----------------------------
    # ε 衰减：逐渐从探索过渡到利用
    # ----------------------------
    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

    # ----------------------------
    # 记录本回合结果
    # ----------------------------
    episode_returns.append(total_reward)

    # 这里假设：终点奖励为正（>0），掉洞奖励为负（<0）
    episode_success.append(1 if final_reward > 0 else 0)
    episode_hole.append(1 if final_reward < 0 else 0)

    # ----------------------------
    # 每隔 EVAL_INTERVAL 回合，统计最近 WINDOW_SIZE 回合表现，用于画曲线
    # ----------------------------
    if (episode + 1) % EVAL_INTERVAL == 0:
        recent_s = episode_success[-WINDOW_SIZE:]
        recent_r = episode_returns[-WINDOW_SIZE:]
        recent_h = episode_hole[-WINDOW_SIZE:]

        x.append(episode + 1)
        success_y.append(np.mean(recent_s) * 100.0)  # 成功率 %
        return_y.append(np.mean(recent_r))           # 平均回报
        hole_y.append(np.mean(recent_h) * 100.0)     # 掉洞率 %

    # 每 500 回合打印一次训练进度
    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1}/{NUM_EPISODES}, epsilon = {EPSILON:.4f}")

env.close()

# ============================================================
# 6) 训练曲线绘制（Plot）
# ============================================================
plt.figure()
plt.plot(x, success_y, label="Success Rate (%)")
# 如果想看平均回报曲线，取消注释：
# plt.plot(x, return_y, label="Avg Return (window)")
# 如果想看掉洞率曲线，取消注释：
# plt.plot(x, hole_y, label="Hole Rate (%)")

plt.xlabel("Episode")
plt.title(f"FrozenLake Q-learning (is_slippery={IS_SLIPPERY})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# 如果想保存图像而不是显示：
# plt.savefig("training_curves.png", dpi=200)

print("\n训练完成！开始评估策略...\n")

# ============================================================
# 7) 测试/评估（Evaluation）：纯贪心策略（greedy）
# ============================================================
test_episodes = 10000

# 创建一个新的测试环境，奖励设置与训练一致
test_env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=IS_SLIPPERY,
    reward_schedule=(1, -1, 0),
    max_episode_steps=-1
)
test_env.reset(seed=SEED + 989)  # 测试用不同 seed，避免和训练完全同序列

success_count = 0        # 成功次数
success_steps_sum = 0    # 成功回合的步数总和（用于算平均步数）

for episode in range(test_episodes):
    state, info = test_env.reset()

    done = False
    step = 0
    final_reward = 0.0

    # 每个测试回合最多走 MAX_STEPS_PER_EPISODE 步
    while not done and step < MAX_STEPS_PER_EPISODE:
        # 评估阶段不探索：直接选 Q 最大动作
        action = int(np.argmax(Q[state]))

        next_state, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

        state = next_state
        step += 1

        # 回合结束时检查是否成功
        if done:
            final_reward = reward
            if final_reward > 0:
                success_count += 1
                success_steps_sum += step
            break

# 计算成功率（%）
success_rate = success_count / test_episodes * 100.0
print(f"\n在 {test_episodes} 次测试中，成功 {success_count} 次，成功率 = {success_rate:.1f}%")

# 计算成功回合的平均步数
if success_count > 0:
    print(f"成功回合的平均步数 = {success_steps_sum / success_count:.2f}")
else:
    print("没有成功回合，无法计算成功回合平均步数")

test_env.close()
