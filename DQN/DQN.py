"""DQN implementation for FrozenLake-v1 using experience replay."""
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
NUM_EPISODES = 8000  # training episodes
MAX_STEPS_PER_EPISODE = 200
TEST_EPISODES = 1000  # per user request
SEED = 42  # reproducibility

LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995  # per-episode decay

BATCH_SIZE = 64
REPLAY_CAPACITY = 50_000
MIN_REPLAY_SIZE = 500
TARGET_UPDATE_FREQUENCY = 500  # gradient steps

IS_SLIPPERY = True  # easier training when False

PLOT_LOSS = True
PLOT_LOSS_SMOOTH_WINDOW = 200  # moving average window

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def to_one_hot(state: np.ndarray, num_states: int):
    return torch.eye(num_states, device=device)[state]


def select_action(policy_net: nn.Module, state: int, epsilon: float, n_actions: int, n_states: int):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_tensor = to_one_hot(torch.tensor(state), n_states).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())


def compute_loss(policy_net, target_net, batch, n_states):
    states, actions, rewards, next_states, dones = batch

    states_tensor = to_one_hot(torch.tensor(states, device=device), n_states)
    next_states_tensor = to_one_hot(torch.tensor(next_states, device=device), n_states)
    actions_tensor = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(1)
    rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
    dones_tensor = torch.tensor(dones, device=device, dtype=torch.float32)

    q_values = policy_net(states_tensor).gather(1, actions_tensor).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states_tensor).max(dim=1)[0]
        target_q = rewards_tensor + DISCOUNT_FACTOR * next_q_values * (1 - dones_tensor)

    criterion = nn.MSELoss()
    return criterion(q_values, target_q)


def train():
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=IS_SLIPPERY)
    env.action_space.seed(SEED)
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
    reward_history = deque(maxlen=100)
    loss_history = []

    for episode in range(NUM_EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = select_action(policy_net, state, epsilon, n_actions, n_states)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(buffer) >= MIN_REPLAY_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = compute_loss(policy_net, target_net, batch, n_states)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1
                if global_step % TARGET_UPDATE_FREQUENCY == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                loss_history.append((global_step, loss.item()))

            if done:
                break

        reward_history.append(episode_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(reward_history) if reward_history else 0.0
            print(
                f"Episode {episode + 1}/{NUM_EPISODES} | "
                f"epsilon={epsilon:.3f} | avg reward (last 100)={avg_reward:.3f}"
            )

    env.close()
    return policy_net, n_states, n_actions, loss_history


def evaluate(policy_net: nn.Module, n_states: int, n_actions: int):
    eval_env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=IS_SLIPPERY)
    eval_env.action_space.seed(SEED + 10_000)
    success_count = 0
    steps_total = 0

    for _ in range(TEST_EPISODES):
        state, _ = eval_env.reset(seed=SEED + _)
        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            with torch.no_grad():
                state_tensor = to_one_hot(torch.tensor(state), n_states).unsqueeze(0)
                action = int(policy_net(state_tensor).argmax(dim=1).item())
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            state = next_state
            if terminated or truncated:
                steps_total += step
                if reward > 0:
                    success_count += 1
                break

    eval_env.close()
    success_rate = success_count / TEST_EPISODES * 100
    avg_steps = steps_total / TEST_EPISODES if TEST_EPISODES > 0 else 0.0
    print(
        f"\nTesting over {TEST_EPISODES} episodes: "
        f"success {success_count}, success rate = {success_rate:.2f}%, "
        f"avg steps = {avg_steps:.2f}"
    )


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    start = time.time()
    trained_net, n_states, n_actions, loss_history = train()
    if PLOT_LOSS and loss_history:
        steps, losses = zip(*loss_history)
        plt.figure(figsize=(8, 4))
        plt.plot(steps, losses, linewidth=0.6, alpha=0.4, label="loss (raw)")
        if PLOT_LOSS_SMOOTH_WINDOW and len(losses) >= PLOT_LOSS_SMOOTH_WINDOW:
            window = PLOT_LOSS_SMOOTH_WINDOW
            kernel = np.ones(window) / window
            smooth_losses = np.convolve(losses, kernel, mode="valid")
            smooth_steps = steps[window - 1 :]
            plt.plot(smooth_steps, smooth_losses, linewidth=1.5, color="C1", label=f"loss (MA {window})")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.title("DQN 8×8 Loss(Stochastic)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("8×8_stochastic.png", dpi=200)
        plt.close()
        print("Saved loss curve to loss_curve_4x4.png")
    print(f"Training finished in {time.time() - start:.1f}s. Starting evaluation...")
    evaluate(trained_net, n_states, n_actions)
