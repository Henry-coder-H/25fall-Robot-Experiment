"""
Nature DQN for FrozenLake-v1 (Gymnasium) using experience replay.

- no-reward version: use environment's default reward (FrozenLake: goal=1, else=0).
- with-reward version: wrap env with a custom reward schedule (goal_reward, hole_penalty, step_penalty).

Key difference (from the PDF):
- Nature DQN target:   y = r + γ * max_a Q_target(s', a)
- Double DQN target:   y = r + γ * Q_target(s', argmax_a Q_online(s', a))

Run:
  python NatureDQN_no_reward.py

Tips:
- Set IS_SLIPPERY=True for stochastic mode; False for deterministic.
- Change MAP_NAME to "8x8" if needed.
"""
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --------------------
# Hyperparameters
# --------------------
SEED = 42

MAP_NAME = "8x8"          # "4x4" or "8x8"
IS_SLIPPERY = False        # stochastic if True
MAX_STEPS_PER_EPISODE = 200

NUM_EPISODES = 8000
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995     # per-episode decay

BATCH_SIZE = 64
REPLAY_CAPACITY = 50_000
MIN_REPLAY_SIZE = 500

TARGET_UPDATE_FREQUENCY = 500   # gradient steps between target sync

EVAL_EPISODES = 1000
PRINT_EVERY = 200

PLOT_LOSS = True
LOSS_SMOOTH_WINDOW = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------
# Utilities
# --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_one_hot(state: torch.Tensor, num_states: int) -> torch.Tensor:
    # state: (B,) long tensor
    return torch.eye(num_states, device=state.device)[state]


def select_action(policy_net: nn.Module, state: int, eps: float, n_states: int, n_actions: int) -> int:
    if random.random() < eps:
        return random.randrange(n_actions)
    with torch.no_grad():
        s = torch.tensor([state], device=device, dtype=torch.long)
        q = policy_net(to_one_hot(s, n_states))
        return int(q.argmax(dim=1).item())


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrozenLakeRewardWrapper(gym.Wrapper):
    """Apply custom reward schedule: (goal_reward, hole_penalty, step_penalty)."""
    def __init__(self, env, reward_schedule):
        super().__init__(env)
        self.goal_reward, self.hole_penalty, self.step_penalty = reward_schedule

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward = self.goal_reward if reward > 0 else self.hole_penalty
        else:
            reward = self.step_penalty
        return obs, reward, terminated, truncated, info


def make_env(use_reward_shaping: bool, reward_schedule=(5, -10, -0.01), seed: int = 0):
    env = gym.make(
        "FrozenLake-v1",
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
        max_episode_steps=MAX_STEPS_PER_EPISODE,
    )
    env.action_space.seed(seed)
    env.reset(seed=seed)

    if use_reward_shaping:
        env = FrozenLakeRewardWrapper(env, reward_schedule)
    return env


def sample_batch(buffer: deque, batch_size: int):
    batch = random.sample(buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
            np.array(next_states), np.array(dones, dtype=np.float32))


def plot_losses(steps, losses, title: str, filename: str):
    if not losses:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, linewidth=0.8, label="loss")
    if len(losses) >= LOSS_SMOOTH_WINDOW:
        window = LOSS_SMOOTH_WINDOW
        kernel = np.ones(window) / window
        smooth = np.convolve(losses, kernel, mode="valid")
        smooth_steps = steps[window - 1:]
        plt.plot(smooth_steps, smooth, linewidth=1.5, label=f"loss (MA {window})")
    plt.xlabel("Gradient Step")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved loss curve to {filename}")


def evaluate(policy_net: nn.Module, use_reward_shaping: bool, reward_schedule=(5, -10, -0.01)) -> tuple:
    """
    Returns:
      success_rate: success_count / EVAL_EPISODES
      avg_steps: average steps per episode (counts episodes that reach max steps too)
    """
    env = make_env(use_reward_shaping=use_reward_shaping, reward_schedule=reward_schedule, seed=SEED + 10_000)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    success = 0
    steps_total = 0

    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        truncated = False

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            with torch.no_grad():
                s = torch.tensor([obs], device=device, dtype=torch.long)
                q = policy_net(to_one_hot(s, n_states))
                action = int(q.argmax(dim=1).item())
            obs, reward, done, truncated, _ = env.step(action)

            if done or truncated:
                steps_total += step
                if done and reward > 0:
                    success += 1
                break
        else:
            steps_total += MAX_STEPS_PER_EPISODE

    env.close()
    success_rate = success / EVAL_EPISODES
    avg_steps = steps_total / EVAL_EPISODES if EVAL_EPISODES > 0 else 0.0
    return success_rate, avg_steps
def compute_loss_nature(policy_net: nn.Module, target_net: nn.Module, batch, n_states: int) -> torch.Tensor:
    """
    Nature DQN target:
      y = r                          if done
      y = r + γ * max_a Q_target(s', a)   otherwise
    """
    states, actions, rewards, next_states, dones = batch

    states_t = torch.tensor(states, device=device, dtype=torch.long)
    next_states_t = torch.tensor(next_states, device=device, dtype=torch.long)
    actions_t = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(1)
    rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
    dones_t = torch.tensor(dones, device=device, dtype=torch.float32)

    q_sa = policy_net(to_one_hot(states_t, n_states)).gather(1, actions_t).squeeze(1)

    with torch.no_grad():
        next_q_max = target_net(to_one_hot(next_states_t, n_states)).max(dim=1)[0]
        target = rewards_t + DISCOUNT_FACTOR * next_q_max * (1.0 - dones_t)

    return nn.MSELoss()(q_sa, target)


def train(use_reward_shaping: bool, reward_schedule=(5, -10, -0.01)):
    set_seed(SEED)
    env = make_env(use_reward_shaping=use_reward_shaping, reward_schedule=reward_schedule, seed=SEED)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy_net = QNetwork(n_states, n_actions).to(device)
    target_net = QNetwork(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay = deque(maxlen=REPLAY_CAPACITY)

    eps = EPSILON_START
    grad_step = 0
    losses = []
    steps = []

    start = time.time()

    for ep in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_return = 0.0

        while not (done or truncated):
            action = select_action(policy_net, obs, eps, n_states, n_actions)
            next_obs, reward, done, truncated, _ = env.step(action)
            ep_return += float(reward)

            replay.append((obs, action, reward, next_obs, float(done)))
            obs = next_obs

            if len(replay) >= MIN_REPLAY_SIZE:
                batch = sample_batch(replay, BATCH_SIZE)
                loss = compute_loss_nature(policy_net, target_net, batch, n_states)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(float(loss.item()))
                steps.append(grad_step)

                grad_step += 1

                if grad_step % TARGET_UPDATE_FREQUENCY == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        eps = max(EPSILON_END, eps * EPSILON_DECAY)

        if ep % PRINT_EVERY == 0:
            success_rate, avg_steps = evaluate(policy_net, use_reward_shaping, reward_schedule)
            print(
                f"[{ep:5d}/{NUM_EPISODES}] eps={eps:.3f} "
                f"last_return={ep_return:.3f} success@{EVAL_EPISODES}={success_rate:.3f} avg_steps={avg_steps:.2f}"
            )

    env.close()

    mode = "with_reward" if use_reward_shaping else "no_reward"
    algo = "nature_dqn"
    if PLOT_LOSS:
        plot_title = f"{algo} loss ({mode}) | map={MAP_NAME} slippery={IS_SLIPPERY}"
        plot_file = f"{algo}_{mode}_loss.png"
        plot_losses(steps, losses, plot_title, plot_file)

    print(f"Training finished in {time.time() - start:.1f}s")
    final_success, final_avg_steps = evaluate(policy_net, use_reward_shaping, reward_schedule)
    print(f"Final success rate over {EVAL_EPISODES} eval episodes: {final_success:.3f} | avg steps: {final_avg_steps:.2f}")
    return policy_net


if __name__ == "__main__":
    # Toggle here
    USE_REWARD_SHAPING = False

    # Reward shaping parameters (used only if USE_REWARD_SHAPING=True)
    REWARD_SCHEDULE = (5, -10, -0.01)

    net = train(USE_REWARD_SHAPING, REWARD_SCHEDULE)
