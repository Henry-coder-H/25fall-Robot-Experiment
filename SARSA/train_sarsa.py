import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import gymnasium as gym  # type: ignore

    GYM_BACKEND = "gymnasium"
except ImportError:  # pragma: no cover
    import gym  # type: ignore

    GYM_BACKEND = "gym"

import numpy as np

from sarsa_agent import SarsaAgent

"""
命令示例：
  python train_sarsa.py --map 4x4 --mode det
  python infer_sarsa.py --map 4x4 --mode det --qpath sarsa_q_4x4_det.npy

一次性跑完多组超参（并分别保存 Q 表 + 输出 CSV 汇总）：
  python train_sarsa.py --map 4x4 --mode sto --sweep --outdir sweep_4x4_sto
"""

MAP_STEP_LIMIT = {"4x4": 100, "8x8": 200}


def get_max_steps(map_size: str) -> int:
    return MAP_STEP_LIMIT[map_size]


def make_env(
    map_size: str, mode: str, seed: Optional[int] = None, max_steps_override: Optional[int] = None
) -> gym.Env:
    is_slippery = mode == "sto"
    env = gym.make("FrozenLake-v1", map_name=map_size, is_slippery=is_slippery)
    if max_steps_override is None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=get_max_steps(map_size))
    elif max_steps_override > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps_override)
    # else: max_steps_override <= 0 => 不加 TimeLimit
    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            try:
                env.seed(seed)
            except Exception:
                pass
    return env


def reset_env(env: gym.Env, seed: Optional[int] = None) -> int:
    try:
        if seed is not None:
            obs, _info = env.reset(seed=seed)
        else:
            obs, _info = env.reset()
    except TypeError:
        obs = env.reset() if seed is None else env.reset(seed=seed)
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


def evaluate_policy(
    agent: SarsaAgent,
    map_size: str,
    mode: str,
    episodes: int = 1000,
    seed: Optional[int] = None,
    eval_max_steps: Optional[int] = None,
) -> Tuple[float, float, float]:
    env = make_env(map_size, mode, max_steps_override=eval_max_steps)
    if eval_max_steps is None:
        max_steps = get_max_steps(map_size)
    elif eval_max_steps > 0:
        max_steps = eval_max_steps
    else:
        max_steps = 10**9  # effectively no cap

    successes = 0
    steps_all: List[int] = []
    steps_success: List[int] = []

    for ep in range(episodes):
        ep_seed = None if seed is None else (seed + ep)
        state = reset_env(env, seed=ep_seed)

        for step in range(1, max_steps + 1):
            action = agent.choose_action(state, greedy=True)
            next_state, reward, done, _ = step_env(env, action)
            state = next_state
            if done:
                steps_all.append(step)
                if reward > 0:
                    successes += 1
                    steps_success.append(step)
                break
        else:
            # TimeLimit reached
            steps_all.append(max_steps)

    env.close()
    success_rate = successes / episodes if episodes > 0 else 0.0
    avg_steps = float(np.mean(steps_all)) if steps_all else 0.0
    avg_steps_success = float(np.mean(steps_success)) if steps_success else 0.0
    return success_rate, avg_steps, avg_steps_success


def default_q_path(map_size: str, mode: str) -> str:
    suffix = "det" if mode == "det" else "sto"
    return f"sarsa_q_{map_size}_{suffix}.npy"


def _fmt_float(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace("-", "m").replace(".", "p")


def build_sweep_q_path(outdir: str, map_size: str, mode: str, alpha: float, gamma: float, epsilon: float) -> str:
    suffix = "det" if mode == "det" else "sto"
    name = (
        f"sarsa_q_{map_size}_{suffix}"
        f"_a{_fmt_float(alpha)}_g{_fmt_float(gamma)}_e{_fmt_float(epsilon)}.npy"
    )
    return str(Path(outdir) / name)


def parse_args():
    parser = argparse.ArgumentParser(description="Tabular SARSA on FrozenLake (gym/gymnasium compatible).")
    parser.add_argument("--map", choices=["4x4", "8x8"], default="4x4", dest="map_size")
    parser.add_argument("--mode", choices=["det", "sto"], default="det")

    parser.add_argument("--episodes", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument(
        "--train_max_steps",
        type=int,
        default=None,
        help="Override max steps during training; <=0 means no TimeLimit, None means use default cap.",
    )

    # single-run hyperparameters
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--alpha_decay", type=float, default=1.0, help="Multiplicative decay per episode.")
    parser.add_argument("--alpha_min", type=float, default=0.01, help="Lower bound for alpha.")

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=0.9999)
    parser.add_argument("--epsilon_min", type=float, default=0.0001)

    parser.add_argument(
        "--adaptive_alpha",
        dest="adaptive_alpha",
        action="store_true",
        help="Enable per-(s,a) adaptive alpha: alpha_eff = max(alpha_min, alpha/sqrt(N(s,a))).",
    )
    parser.add_argument(
        "--no_adaptive_alpha",
        dest="adaptive_alpha",
        action="store_false",
        help="Disable adaptive alpha and use global alpha.",
    )
    parser.set_defaults(adaptive_alpha=True)

    parser.add_argument("--eval_episodes", type=int, default=1000)
    parser.add_argument(
        "--eval_max_steps",
        type=int,
        default=None,
        help="Override max steps during evaluation; <=0 means no cap, None means use training cap.",
    )

    # sweep mode
    parser.add_argument("--sweep", action="store_true", help="Run a small grid and save each Q-table.")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to save sweep outputs.")
    parser.add_argument("--alphas", type=float, nargs="+", default=None)
    parser.add_argument("--gammas", type=float, nargs="+", default=None)
    parser.add_argument("--epsilons", type=float, nargs="+", default=None)

    parser.add_argument("--qpath", type=str, default=None, help="Path to save Q-table (.npy) for single run.")
    return parser.parse_args()


def train_one(
    map_size: str,
    mode: str,
    episodes: int,
    train_max_steps: Optional[int],
    alpha: float,
    alpha_decay: float,
    alpha_min: float,
    adaptive_alpha: bool,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    epsilon_min: float,
    seed: int,
    log_interval: int,
    q_path: str,
    eval_episodes: int,
    eval_max_steps: Optional[int],
) -> Tuple[float, float, float]:
    random.seed(seed)
    np.random.seed(seed)

    env = make_env(map_size, mode, seed=seed, max_steps_override=train_max_steps)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = SarsaAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
    )

    visit_count = np.zeros((n_states, n_actions), dtype=np.int32)

    print(
        f"Training SARSA ({GYM_BACKEND}) | map={map_size} mode={mode} episodes={episodes} "
        f"max_steps={get_max_steps(map_size)} | alpha={alpha} gamma={gamma} eps={epsilon} "
        f"adaptive_alpha={adaptive_alpha}"
    )

    interval_td: List[float] = []
    max_steps = get_max_steps(map_size)
    if train_max_steps is not None:
        if train_max_steps > 0:
            max_steps = train_max_steps
        else:
            max_steps = 10**9  # effectively no cap

    for ep in range(1, episodes + 1):
        state = reset_env(env, seed=seed + ep)
        action = agent.choose_action(state, greedy=False)

        for _step in range(max_steps):
            next_state, reward, done, _ = step_env(env, action)

            visit_count[state, action] += 1
            base_alpha = agent.alpha
            if adaptive_alpha:
                alpha_eff = max(alpha_min, base_alpha / np.sqrt(visit_count[state, action]))
            else:
                alpha_eff = base_alpha
            agent.alpha = alpha_eff

            if done:
                old_q = agent.q_table[state, action]
                target = reward
                interval_td.append(float(target - old_q))
                agent.update(state, action, reward, next_state, None, True)
                agent.alpha = base_alpha
                break

            next_action = agent.choose_action(next_state, greedy=False)
            old_q = agent.q_table[state, action]
            target = reward + agent.gamma * agent.q_table[next_state, next_action]
            interval_td.append(float(target - old_q))
            agent.update(state, action, reward, next_state, next_action, False)
            agent.alpha = base_alpha
            state, action = next_state, next_action

        agent.decay_epsilon()
        if alpha_decay != 1.0:
            agent.alpha = max(alpha_min, agent.alpha * alpha_decay)

        if log_interval > 0 and ep % log_interval == 0:
            mean_td = float(np.mean(np.abs(interval_td))) if interval_td else 0.0
            interval_td = []
            print(f"Episode {ep}/{episodes} | epsilon={agent.epsilon:.4f} | alpha={agent.alpha:.4f} | mean|TD|={mean_td:.4f}")

    Path(q_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(q_path)
    print(f"Saved Q-table to {q_path}")

    agent.epsilon = 0.0
    success_rate, avg_steps, avg_steps_success = evaluate_policy(
        agent,
        map_size,
        mode,
        episodes=eval_episodes,
        seed=seed + 10_000_000,
        eval_max_steps=eval_max_steps,
    )
    print(
        f"Eval over {eval_episodes} episodes: success={success_rate * 100:.2f}% "
        f"| avg_steps={avg_steps:.2f} | avg_steps(success_only)={avg_steps_success:.2f}"
    )

    env.close()
    return success_rate, avg_steps, avg_steps_success


def main():
    args = parse_args()

    if not args.sweep:
        q_path = args.qpath or default_q_path(args.map_size, args.mode)
        train_one(
            map_size=args.map_size,
            mode=args.mode,
            episodes=args.episodes,
            train_max_steps=args.train_max_steps,
            alpha=args.alpha,
            alpha_decay=args.alpha_decay,
            alpha_min=args.alpha_min,
            adaptive_alpha=args.adaptive_alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            seed=args.seed,
            log_interval=args.log_interval,
            q_path=q_path,
            eval_episodes=args.eval_episodes,
            eval_max_steps=args.eval_max_steps,
        )
        return

    
    alphas = args.alphas if args.alphas is not None else [0.03, 0.05,0.1, 0.15,0.2]
    gammas = args.gammas if args.gammas is not None else [0.85,0.95, 0.99]
    epsilons = args.epsilons if args.epsilons is not None else [0.8, 0.85,0.9, 1.0]

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    results_path = Path(args.outdir) / f"sweep_results_{args.map_size}_{args.mode}.csv"

    total = len(alphas) * len(gammas) * len(epsilons)
    run_idx = 0

    rows: List[dict] = []
    for a in alphas:
        for g in gammas:
            for e in epsilons:
                run_idx += 1
                print(f"\n=== Sweep {run_idx}/{total} | alpha={a} gamma={g} epsilon={e} ===")
                q_path = build_sweep_q_path(args.outdir, args.map_size, args.mode, a, g, e)
                success_rate, avg_steps, avg_steps_success = train_one(
                    map_size=args.map_size,
                    mode=args.mode,
                    episodes=args.episodes,
                    train_max_steps=args.train_max_steps,
                    alpha=a,
                    alpha_decay=args.alpha_decay,
                    alpha_min=args.alpha_min,
                    adaptive_alpha=args.adaptive_alpha,
                    gamma=g,
                    epsilon=e,
                    epsilon_decay=args.epsilon_decay,
                    epsilon_min=args.epsilon_min,
                    seed=args.seed,
                    log_interval=args.log_interval,
                    q_path=q_path,
                    eval_episodes=args.eval_episodes,
                    eval_max_steps=args.eval_max_steps,
                )
                rows.append(
                    {
                        "map": args.map_size,
                        "mode": args.mode,
                        "episodes": args.episodes,
                        "alpha": a,
                        "gamma": g,
                        "epsilon": e,
                        "epsilon_decay": args.epsilon_decay,
                        "epsilon_min": args.epsilon_min,
                        "alpha_decay": args.alpha_decay,
                        "alpha_min": args.alpha_min,
                        "adaptive_alpha": int(bool(args.adaptive_alpha)),
                        "eval_episodes": args.eval_episodes,
                        "success_rate": success_rate,
                        "avg_steps": avg_steps,
                        "avg_steps_success": avg_steps_success,
                        "qpath": q_path,
                    }
                )

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved sweep summary to {results_path}")


if __name__ == "__main__":
    main()
