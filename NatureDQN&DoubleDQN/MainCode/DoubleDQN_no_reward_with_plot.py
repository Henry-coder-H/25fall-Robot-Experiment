"""
Double DQN (no reward shaping) for FrozenLake-v1 with avg steps output. (Python 3.8 compatible)
Target:
  y = r + gamma * Q_target(s', argmax_a Q_online(s', a))   (if not done)
"""
import random
import time
from collections import deque
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")  # safe for headless environments
import matplotlib.pyplot as plt

# --------------------
# Hyperparameters
# --------------------
SEED = 42

MAP_NAME = "8x8"          # "4x4" or "8x8"
IS_SLIPPERY = True
MAX_STEPS_PER_EPISODE = 300

NUM_EPISODES = 8000
TEST_EPISODES = 1000

LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

BATCH_SIZE = 128
REPLAY_CAPACITY = 100_000
MIN_REPLAY_SIZE = 2_000
TARGET_UPDATE_FREQUENCY = 500  # gradient steps

PRINT_EVERY = 500

PLOT_LOSS = True
LOSS_SMOOTH_WINDOW = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_one_hot(states: torch.Tensor, num_states: int) -> torch.Tensor:
    return torch.eye(num_states, device=states.device)[states]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


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


def make_env(seed: int):
    env = gym.make(
        "FrozenLake-v1",
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
        max_episode_steps=MAX_STEPS_PER_EPISODE,
    )
    env.action_space.seed(seed)
    env.reset(seed=seed)
    return env


def select_action(policy_net: nn.Module, state: int, epsilon: float, n_states: int, n_actions: int) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        s = torch.tensor([state], device=device, dtype=torch.long)
        q = policy_net(to_one_hot(s, n_states))
        return int(q.argmax(dim=1).item())


# --------------------
# Double DQN loss
# --------------------
def compute_loss_double(policy_net: nn.Module, target_net: nn.Module, batch, n_states: int) -> torch.Tensor:
    """
    Double DQN target:
      y = r                                              if done
      y = r + gamma * Q_target(s', argmax_a Q_online(s', a))  otherwise
    """
    states, actions, rewards, next_states, dones = batch

    states_t = torch.tensor(states, device=device, dtype=torch.long)
    next_states_t = torch.tensor(next_states, device=device, dtype=torch.long)
    actions_t = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(1)
    rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
    dones_t = torch.tensor(dones, device=device, dtype=torch.float32)

    q_sa = policy_net(to_one_hot(states_t, n_states)).gather(1, actions_t).squeeze(1)

    with torch.no_grad():
        # online selects
        q_next_online = policy_net(to_one_hot(next_states_t, n_states))
        best_next_actions = q_next_online.argmax(dim=1, keepdim=True)  # (B,1)
        # target evaluates
        q_next_target = target_net(to_one_hot(next_states_t, n_states)).gather(1, best_next_actions).squeeze(1)
        target = rewards_t + DISCOUNT_FACTOR * q_next_target * (1.0 - dones_t)

    return nn.MSELoss()(q_sa, target)
def plot_losses(steps, losses, title: str, filename: str) -> None:
    """Save a loss curve as a PNG image."""
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


def evaluate(policy_net: nn.Module) -> Tuple[float, float]:
    env = make_env(SEED + 10_000)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    success = 0
    steps_total = 0

    for ep in range(TEST_EPISODES):
        obs, _ = env.reset(seed=SEED + ep)
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
    return success / TEST_EPISODES, steps_total / TEST_EPISODES


def train() -> nn.Module:
    env = make_env(SEED)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy_net = QNetwork(n_states, n_actions).to(device)
    target_net = QNetwork(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(REPLAY_CAPACITY)

    epsilon = EPSILON_START
    global_step = 0
    start = time.time()


    losses = []
    loss_steps = []
    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset(seed=SEED + episode)
        done = False
        truncated = False

        while not (done or truncated):
            action = select_action(policy_net, state, epsilon, n_states, n_actions)
            next_state, reward, done, truncated, _ = env.step(action)
            buffer.push(state, action, reward, next_state, float(done or truncated))
            state = next_state

            if len(buffer) >= MIN_REPLAY_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = compute_loss_double(policy_net, target_net, batch, n_states)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                losses.append(float(loss.item()))
                loss_steps.append(global_step)
                global_step += 1
                if global_step % TARGET_UPDATE_FREQUENCY == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % PRINT_EVERY == 0:
            sr, avg_steps = evaluate(policy_net)
            print(f"[{episode}/{NUM_EPISODES}] eps={epsilon:.3f} success_rate={sr:.3f} avg_steps={avg_steps:.2f}")

    env.close()
    print(f"Training finished in {time.time() - start:.1f}s")
    if PLOT_LOSS:
        algo = "double_dqn"
        mode = "no_reward"
        plot_title = f"{algo} loss ({mode}) | map={MAP_NAME} slippery={IS_SLIPPERY}"
        plot_file = f"{algo}_{mode}_loss.png"
        plot_losses(loss_steps, losses, plot_title, plot_file)

    return policy_net


if __name__ == "__main__":
    set_seed(SEED)
    net = train()
    sr, avg_steps = evaluate(net)
    print(f"\nFinal test over {TEST_EPISODES} episodes: success_rate={sr*100:.2f}% | avg_steps={avg_steps:.2f}")