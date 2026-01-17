# Q-learning_slippery.py
# “打滑”环境下的实机运行代码

import gymnasium as gym
import numpy as np
from SerialThread import *

# --------------------
# 超参数（可以自己调）
# --------------------
NUM_EPISODES = 5000  # 训练的总轮数
MAX_STEPS_PER_EPISODE = 100  # 每个回合最多步数

LEARNING_RATE = 0.8  # 学习率 alpha
DISCOUNT_FACTOR = 0.95  # 折扣因子 gamma

EPSILON = 1.0  # 初始探索率
EPSILON_DECAY = 0.999  # 每回合探索率衰减
EPSILON_MIN = 0.01  # 最小探索率

# 是否让环境“打滑”
# 这里选择打滑的环境
IS_SLIPPERY = True

# --------------------
# 创建环境（训练用：不渲染）
# --------------------
env = gym.make("FrozenLake-v1", is_slippery=IS_SLIPPERY, success_rate=1.0/3.0)

n_states = env.observation_space.n
n_actions = env.action_space.n

# Q 表：状态 × 动作
Q = np.zeros((n_states, n_actions))

# --------------------
# 训练 Q-learning
# --------------------
for episode in range(NUM_EPISODES):
    state, info = env.reset()

    for step in range(MAX_STEPS_PER_EPISODE):
        # ε-贪心选动作
        if np.random.rand() < EPSILON:
            action = env.action_space.sample()  # 随机探索
        else:
            action = np.argmax(Q[state])  # 利用当前最优动作

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-learning 更新
        best_next_q = np.max(Q[next_state])
        td_target = reward + DISCOUNT_FACTOR * best_next_q
        td_error = td_target - Q[state, action]
        Q[state, action] += LEARNING_RATE * td_error

        state = next_state

        if done:
            break

    # 衰减探索率
    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

    # 每隔一段打印一下进度
    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1}/{NUM_EPISODES}, epsilon = {EPSILON:.3f}")

env.close()

print("\n训练完成！开始评估策略...\n")

# --------------------
# 评估训练后的策略（渲染可视化）
# --------------------
test_episodes = 1
test_env = gym.make("FrozenLake-v1", is_slippery=IS_SLIPPERY, success_rate=1.0/3.0, render_mode="human")
st = SerialThread("COM6")
st.send().takeoff(50)
sleep(3)
st.send().speed(20)
sleep(1)
st.send().high(80)
sleep(2)

success_count = 0

for episode in range(test_episodes):
    state, info = test_env.reset()
    done = False
    step = 0
    print(f"=== 测试回合 {episode + 1} ===")

    while not done and step < MAX_STEPS_PER_EPISODE:
        test_env.render()

        # 使用贪心策略（不再探索）
        action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

        # ========= 在这里根据 action 控制机器人 =========
        # 坐标系：
        # 0: LEFT   -> 左移
        # 1: DOWN   -> 后退
        # 2: RIGHT  -> 右移
        # 3: UP     -> 前进
        if state != next_state:
            if next_state-state == -1:
                st.send().left(50)      # 向左移动
            elif next_state-state == 4:
                st.send().back(50)      # 向后移动
            elif next_state-state == 1:
                st.send().right(50)     # 向右移动
            elif next_state-state == -4:
                st.send().forward(50)   # 向前移动
        # ===============================================

        # 如果需要，给机器人一点时间执行动作
        sleep(2)

        state = next_state
        step += 1

        if done:
            test_env.render()
            if reward > 0:
                print("到达终点，成功 ✅")
                success_count += 1
            else:
                print("掉坑/失败 ❌")

            break
    st.send().land()
    sleep(2)
    st.shutdown()
    sleep(2)

print(f"\n在 {test_episodes} 次测试中，成功 {success_count} 次，成功率 = {success_count / test_episodes * 100:.1f}%")

test_env.close()
