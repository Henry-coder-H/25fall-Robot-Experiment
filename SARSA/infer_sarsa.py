import argparse
import time
from typing import Optional, Tuple

try:
    import gymnasium as gym  # type: ignore
except ImportError:  # pragma: no cover
    import gym  # type: ignore
import numpy as np

from sarsa_agent import SarsaAgent
from viz import FrozenLakeVisualizer

MAP_STEP_LIMIT = {"4x4": 100, "8x8": 200}


def get_max_steps(map_size: str) -> int:
    return MAP_STEP_LIMIT[map_size]


def make_env(map_size: str, mode: str, seed: Optional[int] = None) -> gym.Env:
    is_slippery = mode == "sto"
    env = gym.make("FrozenLake-v1", map_name=map_size, is_slippery=is_slippery)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=get_max_steps(map_size))
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.seed(seed)
    return env


def reset_env(env: gym.Env, seed: Optional[int] = None) -> int:
    try:
        if seed is not None:
            obs, info = env.reset(seed=seed)
        else:
            obs, info = env.reset()
    except TypeError:
        obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs
    return int(obs)


def step_env(env: gym.Env, action: int):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, reward, done, info = out
    return int(obs), float(reward), bool(done), info


def default_q_path(map_size: str, mode: str) -> str:
    suffix = "det" if mode == "det" else "sto"
    return f"sarsa_q_{map_size}_{suffix}.npy"


def state_to_pos(state: int, ncol: int) -> Tuple[int, int]:
    row = state // ncol
    col = state % ncol
    return row, col


def parse_args():
    parser = argparse.ArgumentParser(description="Roll out a trained SARSA policy on FrozenLake.")
    parser.add_argument("--map", choices=["4x4", "8x8"], default="4x4", dest="map_size")
    parser.add_argument("--mode", choices=["det", "sto"], default="det")
    parser.add_argument("--qpath", type=str, default=None, help="Path to Q-table (.npy).")
    parser.add_argument("--sleep", type=float, default=0.4, help="Pause between frames (seconds).")
    parser.add_argument("--seed", type=int, default=123, help="Reset seed for rollout.")
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_env(args.map_size, args.mode, seed=args.seed)
    q_path = args.qpath or default_q_path(args.map_size, args.mode)

    q_table = np.load(q_path)
    n_states, n_actions = q_table.shape
    if n_states != env.observation_space.n or n_actions != env.action_space.n:
        raise ValueError(
            f"Q-table shape {q_table.shape} does not match env "
            f"({env.observation_space.n}, {env.action_space.n})"
        )

    agent = SarsaAgent.load(q_path, epsilon=0.0, epsilon_decay=1.0, epsilon_min=0.0)
    agent.epsilon = 0.0  # greedy for inference

    desc = env.unwrapped.desc
    viz = FrozenLakeVisualizer(desc, mode=args.mode, map_size=args.map_size, sleep=args.sleep)

    state = reset_env(env, seed=args.seed)
    viz.update(state, step_count=0, done=False)
    success = False

    for step in range(1, get_max_steps(args.map_size) + 1):
        action = agent.choose_action(state, greedy=True)
        next_state, reward, done, _ = step_env(env, action)
        viz.update(next_state, step_count=step, reward=reward, done=done)
        state = next_state
        if done:
            success = reward > 0
            break

    env.close()
    viz.finalize()
    print(f"Finished rollout | success={success} | steps={step} (cap={get_max_steps(args.map_size)})")


if __name__ == "__main__":
    main()

