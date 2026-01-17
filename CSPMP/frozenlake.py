# 依赖导入
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# 运行开关
IS_SLIPPERY = False
RUN_MAP_SIZES = [4, 8]
USE_DEFAULT_MAP = True


# 默认地图
DEFAULT_DESC_4X4 = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]

DEFAULT_DESC_8X8 = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]


# 公共参数
USE_PATH_WEIGHTING = True
PATH_WEIGHT_POWER = 1.0
TIE_PROB_TOL = 1e-12

GLOBAL_SEED = 6114
SEED_BASE_TRAIN_FHR = 30000
SEED_BASE_TRAIN_IHR = 60000
SEED_BASE_VAL_FHR = 90000
SEED_BASE_VAL_IHR = 120000
SEED_BASE_TEST_FHR = 150000
SEED_BASE_TEST_IHR = 180000


@dataclass
class ModeConfig:
    is_slippery: bool
    transition_estimator: str
    dirichlet_alpha: float

    step_limit_4x4: int
    step_limit_8x8: int

    unlimited_max_steps_4x4: int
    unlimited_max_steps_8x8: int

    train_max_steps_ihr_4x4: int
    train_max_steps_ihr_8x8: int

    eval_max_steps_ihr_4x4: int
    eval_max_steps_ihr_8x8: int

    val_episodes_fhr: int
    val_episodes_ihr: int
    test_episodes_fhr: int
    test_episodes_ihr: int

    plan_every_fhr: int
    plan_every_ihr: int

    train_episodes_fhr_4x4: int
    train_episodes_fhr_8x8: int
    train_episodes_ihr_4x4: int
    train_episodes_ihr_8x8: int

    warmup_fhr: int
    warmup_ihr: int

    eps_start: float
    eps_end: float
    eps_decay_frac: float

    vi_max_iters: int
    vi_eps: float
    u_max_iters: int
    u_eps: float

    enable_early_stop: bool
    early_stop_min_fhr: int
    early_stop_min_ihr: int
    early_stop_smooth_k: int
    early_stop_gap_tol_fhr: float
    early_stop_gap_tol_ihr: float
    early_stop_gap_streak: int

    plot_training_curves: bool
    plot_visit_heatmap: bool
    heatmap_log_scale: bool
    heatmap_from: str


def get_mode_config(is_slippery: bool) -> ModeConfig:
    # 模式参数
    if is_slippery:
        return ModeConfig(
            is_slippery=True,
            transition_estimator="dirichlet",
            dirichlet_alpha=0.10,
            step_limit_4x4=100,
            step_limit_8x8=200,
            unlimited_max_steps_4x4=10000,
            unlimited_max_steps_8x8=20000,
            train_max_steps_ihr_4x4=5000,
            train_max_steps_ihr_8x8=8000,
            eval_max_steps_ihr_4x4=5000,
            eval_max_steps_ihr_8x8=8000,
            val_episodes_fhr=1000,
            val_episodes_ihr=1000,
            test_episodes_fhr=1000,
            test_episodes_ihr=1000,
            plan_every_fhr=50,
            plan_every_ihr=50,
            train_episodes_fhr_4x4=10000,
            train_episodes_fhr_8x8=15000,
            train_episodes_ihr_4x4=10000,
            train_episodes_ihr_8x8=10000,
            warmup_fhr=1000,
            warmup_ihr=1000,
            eps_start=1.0,
            eps_end=0.1,
            eps_decay_frac=0.80,
            vi_max_iters=120000,
            vi_eps=1e-12,
            u_max_iters=120000,
            u_eps=1e-12,
            enable_early_stop=True,
            early_stop_min_fhr=1500,
            early_stop_min_ihr=2000,
            early_stop_smooth_k=8,
            early_stop_gap_tol_fhr=0.002,
            early_stop_gap_tol_ihr=0.001,
            early_stop_gap_streak=5,
            plot_training_curves=True,
            plot_visit_heatmap=True,
            heatmap_log_scale=True,
            heatmap_from="BOTH",
        )

    return ModeConfig(
        is_slippery=False,
        transition_estimator="mle",
        dirichlet_alpha=0.10,
        step_limit_4x4=10,
        step_limit_8x8=20,
        unlimited_max_steps_4x4=100,
        unlimited_max_steps_8x8=200,
        train_max_steps_ihr_4x4=1000,
        train_max_steps_ihr_8x8=2000,
        eval_max_steps_ihr_4x4=100,
        eval_max_steps_ihr_8x8=200,
        val_episodes_fhr=200,
        val_episodes_ihr=200,
        test_episodes_fhr=500,
        test_episodes_ihr=500,
        plan_every_fhr=1,
        plan_every_ihr=1,
        train_episodes_fhr_4x4=1000,
        train_episodes_fhr_8x8=1000,
        train_episodes_ihr_4x4=1000,
        train_episodes_ihr_8x8=1000,
        warmup_fhr=10,
        warmup_ihr=10,
        eps_start=1.0,
        eps_end=0.0,
        eps_decay_frac=0.50,
        vi_max_iters=20000,
        vi_eps=1e-10,
        u_max_iters=20000,
        u_eps=1e-10,
        enable_early_stop=True,
        early_stop_min_fhr=0,
        early_stop_min_ihr=0,
        early_stop_smooth_k=8,
        early_stop_gap_tol_fhr=0.001,
        early_stop_gap_tol_ihr=0.001,
        early_stop_gap_streak=10,
        plot_training_curves=True,
        plot_visit_heatmap=True,
        heatmap_log_scale=True,
        heatmap_from="BOTH",
    )


def _reset_env(env: gym.Env, seed: Optional[int] = None):
    obs, info = env.reset(seed=seed)
    return obs, info


def _step_env(env: gym.Env, action: int):
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    return obs, float(reward), done, bool(terminated), bool(truncated), info


def create_env_from_desc(desc: List[str], is_slippery: bool, max_episode_steps: Optional[int]) -> gym.Env:
    if max_episode_steps is None:
        return gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery)
    return gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery, max_episode_steps=max_episode_steps)


def parse_map(desc: List[str], map_size: int) -> Tuple[int, int, np.ndarray, np.ndarray]:
    S = map_size * map_size
    is_goal = np.zeros(S, dtype=bool)
    is_hole = np.zeros(S, dtype=bool)
    start_state = None
    goal_state = None

    for r in range(map_size):
        for c in range(map_size):
            s = r * map_size + c
            ch = desc[r][c]
            if ch == "S":
                start_state = s
            elif ch == "G":
                goal_state = s
                is_goal[s] = True
            elif ch == "H":
                is_hole[s] = True

    if start_state is None or goal_state is None:
        raise ValueError("Map must contain S and G")
    return int(start_state), int(goal_state), is_goal, is_hole


def extract_P_true(env: gym.Env, S: int, A: int) -> np.ndarray:
    base = env.unwrapped
    if not hasattr(base, "P"):
        raise RuntimeError("Cannot find env.P")

    P_dict = base.P
    P_true = np.zeros((S, A, S), dtype=np.float64)

    for s in range(S):
        for a in range(A):
            for prob, sp, _r, _done in P_dict[s][a]:
                P_true[s, a, int(sp)] += float(prob)

    P_true = np.clip(P_true, 0.0, 1.0)
    P_true /= np.maximum(P_true.sum(axis=2, keepdims=True), 1e-15)
    return P_true


@dataclass
class EmpiricalModel:
    S: int
    A: int
    alpha: float
    estimator: str
    N_sa: np.ndarray
    N_sas: np.ndarray

    @staticmethod
    def create(S: int, A: int, alpha: float, estimator: str) -> "EmpiricalModel":
        return EmpiricalModel(
            S=S,
            A=A,
            alpha=alpha,
            estimator=estimator,
            N_sa=np.zeros((S, A), dtype=np.int64),
            N_sas=np.zeros((S, A, S), dtype=np.int64),
        )

    def update(self, s: int, a: int, sp: int):
        self.N_sa[s, a] += 1
        self.N_sas[s, a, sp] += 1

    def build_P_hat(self, is_terminal: np.ndarray) -> np.ndarray:
        # 经验转移估计
        S, A = self.S, self.A
        P_hat = np.zeros((S, A, S), dtype=np.float64)

        if self.estimator == "dirichlet":
            denom = self.N_sa.astype(np.float64) + self.alpha * S
            denom = np.maximum(denom, 1e-15)
            P_hat = (self.N_sas.astype(np.float64) + self.alpha) / denom[:, :, None]
        elif self.estimator == "mle":
            mask = self.N_sa > 0
            if np.any(mask):
                P_hat[mask] = self.N_sas[mask].astype(np.float64) / self.N_sa[mask].astype(np.float64)[:, None]
            P_hat[~mask] = 1.0 / S
        else:
            raise ValueError("Unknown estimator")

        term_states = np.where(is_terminal)[0]
        for s in term_states:
            P_hat[s, :, :] = 0.0
            P_hat[s, :, s] = 1.0

        P_hat = np.clip(P_hat, 0.0, 1.0)
        P_hat /= np.maximum(P_hat.sum(axis=2, keepdims=True), 1e-15)
        return P_hat


def finite_horizon_lexicographic_dp(
    P: np.ndarray,
    is_goal: np.ndarray,
    is_hole: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    S, A, _ = P.shape
    p = np.zeros((horizon + 1, S), dtype=np.float64)
    u = np.zeros((horizon + 1, S), dtype=np.float64)
    pi = np.zeros((horizon + 1, S), dtype=np.int32)

    p[0, is_goal] = 1.0
    u[0, :] = 0.0

    for t in range(1, horizon + 1):
        p[t, is_goal] = 1.0
        p[t, is_hole] = 0.0
        u[t, is_goal] = 0.0
        u[t, is_hole] = 0.0

    for t in range(1, horizon + 1):
        prev_p = p[t - 1, :]
        prev_u = u[t - 1, :]
        for s in range(S):
            if is_goal[s] or is_hole[s]:
                pi[t, s] = 0
                continue

            best_p = -1.0
            best_u = 1e300
            best_a = 0

            for a in range(A):
                probs = P[s, a, :]
                cand_p = float(np.dot(probs, prev_p))
                cand_u = float(np.dot(probs, (prev_u + prev_p)))

                if cand_p > best_p + TIE_PROB_TOL:
                    best_p, best_u, best_a = cand_p, cand_u, a
                elif abs(cand_p - best_p) <= TIE_PROB_TOL and USE_PATH_WEIGHTING:
                    if (cand_u ** PATH_WEIGHT_POWER) < (best_u ** PATH_WEIGHT_POWER):
                        best_u, best_a = cand_u, a

            p[t, s] = np.clip(best_p, 0.0, 1.0)
            u[t, s] = max(0.0, best_u)
            pi[t, s] = best_a

    return pi, p, u


def unbounded_reachability_vi(
    P: np.ndarray,
    is_goal: np.ndarray,
    is_hole: np.ndarray,
    vi_max_iters: int,
    vi_eps: float,
) -> np.ndarray:
    S, A, _ = P.shape
    V = np.zeros(S, dtype=np.float64)
    V[is_goal] = 1.0
    V[is_hole] = 0.0

    for _ in range(vi_max_iters):
        V_new = V.copy()
        for s in range(S):
            if is_goal[s] or is_hole[s]:
                continue
            q = P[s, :, :] @ V
            V_new[s] = float(np.max(q))
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        if delta < vi_eps:
            break

    return np.clip(V, 0.0, 1.0)


def build_Astar(P: np.ndarray, V: np.ndarray, is_goal: np.ndarray, is_hole: np.ndarray) -> List[List[int]]:
    S, A, _ = P.shape
    Astar = [[] for _ in range(S)]
    for s in range(S):
        if is_goal[s] or is_hole[s]:
            Astar[s] = [0]
            continue
        q = P[s, :, :] @ V
        best = float(np.max(q))
        Astar[s] = [a for a in range(A) if abs(float(q[a]) - best) <= TIE_PROB_TOL]
        if not Astar[s]:
            Astar[s] = [int(np.argmax(q))]
    return Astar


def speed_optimize_on_Astar(
    P: np.ndarray,
    V: np.ndarray,
    Astar: List[List[int]],
    is_goal: np.ndarray,
    is_hole: np.ndarray,
    u_max_iters: int,
    u_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    S, A, _ = P.shape
    u = np.zeros(S, dtype=np.float64)
    u[is_goal] = 0.0
    u[is_hole] = 0.0

    for _ in range(u_max_iters):
        u_new = u.copy()
        for s in range(S):
            if is_goal[s] or is_hole[s]:
                continue
            best_u = 1e300
            for a in Astar[s]:
                probs = P[s, a, :]
                val = float(np.dot(probs, (u + V)))
                if val < best_u:
                    best_u = val
            u_new[s] = best_u
        delta = float(np.max(np.abs(u_new - u)))
        u = u_new
        if delta < u_eps:
            break

    pi = np.zeros(S, dtype=np.int32)
    for s in range(S):
        if is_goal[s] or is_hole[s]:
            pi[s] = 0
            continue
        best_u = 1e300
        best_a = Astar[s][0]
        for a in Astar[s]:
            probs = P[s, a, :]
            val = float(np.dot(probs, (u + V)))
            if val < best_u - 1e-15:
                best_u, best_a = val, a
            elif abs(val - best_u) <= 1e-15 and USE_PATH_WEIGHTING:
                if (val ** PATH_WEIGHT_POWER) < (best_u ** PATH_WEIGHT_POWER):
                    best_a = a
        pi[s] = best_a

    return pi, u


def epsilon_by_progress(ep: int, total_eps: int, eps_start: float, eps_end: float, eps_decay_frac: float) -> float:
    if total_eps <= 1:
        return eps_end
    decay_eps = int(eps_decay_frac * total_eps)
    if ep <= 0:
        return eps_start
    if ep >= decay_eps:
        return eps_end
    return eps_start + (eps_end - eps_start) * (ep / max(1, decay_eps))


@dataclass
class EvalStats:
    success_rate: float
    success_count: int
    avg_steps_success: float
    avg_steps_all: float


def eval_policy_fhr(
    cfg: ModeConfig,
    desc: List[str],
    map_size: int,
    pi_fhr: np.ndarray,
    episodes: int,
    seed_base: int,
) -> Tuple[EvalStats, np.ndarray]:
    H = cfg.step_limit_4x4 if map_size == 4 else cfg.step_limit_8x8
    env = create_env_from_desc(desc, cfg.is_slippery, max_episode_steps=H)

    S = env.observation_space.n
    _, goal_state, _, _ = parse_map(desc, map_size)

    visited = np.zeros(S, dtype=np.int64)
    successes = 0
    steps_succ = []
    steps_all = []

    for ep in range(episodes):
        s, _ = _reset_env(env, seed=seed_base + map_size * 100000 + ep)
        done = False
        steps = 0
        while (not done) and (steps < H):
            visited[int(s)] += 1
            remaining = H - steps
            a = int(pi_fhr[remaining, int(s)])
            s, _r, done, _ter, _tru, _info = _step_env(env, a)
            steps += 1
        steps_all.append(steps)
        if int(s) == goal_state:
            successes += 1
            steps_succ.append(steps)

    sr = successes / max(1, episodes)
    return (
        EvalStats(
            sr,
            successes,
            float(np.mean(steps_succ)) if steps_succ else float("nan"),
            float(np.mean(steps_all)) if steps_all else float("nan"),
        ),
        visited,
    )


def eval_policy_ihr(
    cfg: ModeConfig,
    desc: List[str],
    map_size: int,
    pi_ihr: np.ndarray,
    episodes: int,
    seed_base: int,
    cap_steps: int,
) -> Tuple[EvalStats, np.ndarray]:
    env = create_env_from_desc(desc, cfg.is_slippery, max_episode_steps=cap_steps)

    S = env.observation_space.n
    _, goal_state, _, _ = parse_map(desc, map_size)

    visited = np.zeros(S, dtype=np.int64)
    successes = 0
    steps_succ = []
    steps_all = []

    for ep in range(episodes):
        s, _ = _reset_env(env, seed=seed_base + map_size * 100000 + ep)
        done = False
        steps = 0
        while (not done) and (steps < cap_steps):
            visited[int(s)] += 1
            a = int(pi_ihr[int(s)])
            s, _r, done, _ter, _tru, _info = _step_env(env, a)
            steps += 1
        steps_all.append(steps)
        if int(s) == goal_state:
            successes += 1
            steps_succ.append(steps)

    sr = successes / max(1, episodes)
    return (
        EvalStats(
            sr,
            successes,
            float(np.mean(steps_succ)) if steps_succ else float("nan"),
            float(np.mean(steps_all)) if steps_all else float("nan"),
        ),
        visited,
    )


def plot_training_curves(
    checkpoints_fhr: List[int],
    val_acc_fhr: List[float],
    checkpoints_ihr: List[int],
    val_acc_ihr: List[float],
    filename: str,
):
    # 训练曲线
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    if len(checkpoints_fhr) == len(val_acc_fhr) and len(checkpoints_fhr) > 0:
        ax.plot(checkpoints_fhr, val_acc_fhr, label="Val FHR")
    if len(checkpoints_ihr) == len(val_acc_ihr) and len(checkpoints_ihr) > 0:
        ax.plot(checkpoints_ihr, val_acc_ihr, label="Val IHR")

    ax.set_title("Training Curves")
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Success Rate")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_visited_states_heatmap(
    visited: np.ndarray,
    desc: List[str],
    map_size: int,
    heatmap_log_scale: bool,
    filename: str,
):
    # 访问热力图
    Z = visited.reshape((map_size, map_size)).astype(np.float64)
    Z_plot = np.log1p(Z) if heatmap_log_scale else Z

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(Z_plot, origin="upper")

    ax.set_title("Visited States")
    ax.set_xlabel("Col")
    ax.set_ylabel("Row")

    ax.set_xticks(np.arange(-0.5, map_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, map_size, 1), minor=True)
    ax.grid(which="minor", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(map_size):
        for c in range(map_size):
            ch = desc[r][c]
            if ch in ("S", "H", "G"):
                ax.text(c, r, ch, ha="center", va="center", fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def moving_average(vals: List[float], k: int) -> float:
    if not vals:
        return float("nan")
    k = max(1, min(k, len(vals)))
    return float(np.mean(vals[-k:]))


def should_stop_by_gap(
    ep_plus_1: int,
    val_acc_hist: List[float],
    theo_opt: float,
    min_episodes: int,
    gap_tol: float,
    streak_need: int,
    streak_now: int,
    smooth_k: int,
) -> Tuple[bool, int, float, float]:
    sm = moving_average(val_acc_hist, smooth_k)
    if (ep_plus_1 < min_episodes) or (not np.isfinite(sm)):
        return False, 0, sm, float("inf")

    gap = float(theo_opt - sm)
    if gap <= gap_tol:
        streak_now += 1
    else:
        streak_now = 0

    stop = streak_now >= streak_need
    return stop, streak_now, sm, gap


def compute_theoretical_optima(cfg: ModeConfig, desc: List[str], map_size: int) -> Tuple[float, float]:
    # 理论最优
    H = cfg.step_limit_4x4 if map_size == 4 else cfg.step_limit_8x8
    cap = cfg.unlimited_max_steps_4x4 if map_size == 4 else cfg.unlimited_max_steps_8x8

    env_fhr = create_env_from_desc(desc, cfg.is_slippery, max_episode_steps=H)
    env_ihr = create_env_from_desc(desc, cfg.is_slippery, max_episode_steps=cap)

    S = env_fhr.observation_space.n
    A = env_fhr.action_space.n
    start, _goal, is_goal, is_hole = parse_map(desc, map_size)
    is_terminal = is_goal | is_hole

    P_true_fhr = extract_P_true(env_fhr, S, A)
    term_states = np.where(is_terminal)[0]
    for s in term_states:
        P_true_fhr[s, :, :] = 0.0
        P_true_fhr[s, :, s] = 1.0

    _pi_star, p_star, _u_star = finite_horizon_lexicographic_dp(P_true_fhr, is_goal, is_hole, horizon=H)
    p_star_start_fhr = float(p_star[H, start])

    P_true_ihr = extract_P_true(env_ihr, S, A)
    for s in term_states:
        P_true_ihr[s, :, :] = 0.0
        P_true_ihr[s, :, s] = 1.0

    V_star = unbounded_reachability_vi(P_true_ihr, is_goal, is_hole, cfg.vi_max_iters, cfg.vi_eps)
    p_star_start_ihr = float(V_star[start])

    return p_star_start_fhr, p_star_start_ihr


def train_fhr(cfg: ModeConfig, desc: List[str], map_size: int, p_star_start_fhr: float) -> Dict:
    # 有限步训练
    H = cfg.step_limit_4x4 if map_size == 4 else cfg.step_limit_8x8
    env = create_env_from_desc(
        desc,
        cfg.is_slippery,
        max_episode_steps=cfg.unlimited_max_steps_4x4 if map_size == 4 else cfg.unlimited_max_steps_8x8,
    )

    S = env.observation_space.n
    A = env.action_space.n
    _s0, _g, is_goal, is_hole = parse_map(desc, map_size)
    is_terminal = is_goal | is_hole

    total_train_eps = cfg.train_episodes_fhr_4x4 if map_size == 4 else cfg.train_episodes_fhr_8x8
    model = EmpiricalModel.create(S, A, cfg.dirichlet_alpha, cfg.transition_estimator)

    pi_fhr = np.zeros((H + 1, S), dtype=np.int32)

    checkpoints: List[int] = []
    val_acc_hist: List[float] = []

    best_val_acc = -1.0
    best_pi_fhr = pi_fhr.copy()
    best_value_fhr = np.zeros(S, dtype=np.float64)

    gap_streak = 0

    iterator = tqdm(
        range(total_train_eps),
        desc=f"Train FHR {map_size}x{map_size}",
        leave=True,
    )

    for ep in iterator:
        eps = 1.0 if ep < cfg.warmup_fhr else epsilon_by_progress(
            ep - cfg.warmup_fhr,
            max(1, total_train_eps - cfg.warmup_fhr),
            cfg.eps_start,
            cfg.eps_end,
            cfg.eps_decay_frac,
        )

        s, _ = _reset_env(env, seed=SEED_BASE_TRAIN_FHR + map_size * 100000 + ep)
        steps, done = 0, False

        while (not done) and (steps < H):
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                remaining = H - steps
                a = int(pi_fhr[remaining, int(s)])

            sp, _r, done, _ter, _tru, _info = _step_env(env, int(a))
            model.update(int(s), int(a), int(sp))
            s = sp
            steps += 1

        if (ep + 1) % cfg.plan_every_fhr == 0:
            P_hat = model.build_P_hat(is_terminal=is_terminal)
            pi_fhr, pA, _uA = finite_horizon_lexicographic_dp(P_hat, is_goal, is_hole, horizon=H)

            val_stats, _ = eval_policy_fhr(cfg, desc, map_size, pi_fhr, cfg.val_episodes_fhr, SEED_BASE_VAL_FHR)
            val_acc = float(val_stats.success_rate)

            checkpoints.append(ep + 1)
            val_acc_hist.append(val_acc)

            if val_acc > best_val_acc + 1e-12:
                best_val_acc = val_acc
                best_pi_fhr = pi_fhr.copy()
                best_value_fhr = pA[H, :].copy()

            if cfg.enable_early_stop:
                stop, gap_streak, sm, gap = should_stop_by_gap(
                    ep_plus_1=ep + 1,
                    val_acc_hist=val_acc_hist,
                    theo_opt=p_star_start_fhr,
                    min_episodes=cfg.early_stop_min_fhr,
                    gap_tol=cfg.early_stop_gap_tol_fhr,
                    streak_need=cfg.early_stop_gap_streak,
                    streak_now=gap_streak,
                    smooth_k=cfg.early_stop_smooth_k,
                )
                if stop:
                    break

            iterator.set_postfix({"eps": f"{eps:.3f}", "val": f"{val_acc:.3f}"})

    return {
        "H": H,
        "best_pi_fhr": best_pi_fhr,
        "best_value_fhr": best_value_fhr,
        "checkpoints": checkpoints,
        "val_acc_hist": val_acc_hist,
    }


def train_ihr(cfg: ModeConfig, desc: List[str], map_size: int, p_star_start_ihr: float) -> Dict:
    # 无限步训练
    cap_eval = cfg.eval_max_steps_ihr_4x4 if map_size == 4 else cfg.eval_max_steps_ihr_8x8
    cap_train = cfg.train_max_steps_ihr_4x4 if map_size == 4 else cfg.train_max_steps_ihr_8x8

    env = create_env_from_desc(desc, cfg.is_slippery, max_episode_steps=max(cap_eval, cap_train))

    S = env.observation_space.n
    A = env.action_space.n
    _s0, _g, is_goal, is_hole = parse_map(desc, map_size)
    is_terminal = is_goal | is_hole

    total_train_eps = cfg.train_episodes_ihr_4x4 if map_size == 4 else cfg.train_episodes_ihr_8x8
    model = EmpiricalModel.create(S, A, cfg.dirichlet_alpha, cfg.transition_estimator)

    pi_ihr = np.zeros(S, dtype=np.int32)

    checkpoints: List[int] = []
    val_acc_hist: List[float] = []

    best_val_acc = -1.0
    best_pi_ihr = pi_ihr.copy()
    best_value_ihr = np.zeros(S, dtype=np.float64)

    gap_streak = 0

    iterator = tqdm(
        range(total_train_eps),
        desc=f"Train IHR {map_size}x{map_size}",
        leave=True,
    )

    for ep in iterator:
        eps = 1.0 if ep < cfg.warmup_ihr else epsilon_by_progress(
            ep - cfg.warmup_ihr,
            max(1, total_train_eps - cfg.warmup_ihr),
            cfg.eps_start,
            cfg.eps_end,
            cfg.eps_decay_frac,
        )

        s, _ = _reset_env(env, seed=SEED_BASE_TRAIN_IHR + map_size * 100000 + ep)
        done, steps = False, 0

        while (not done) and (steps < cap_train):
            if random.random() < eps:
                a = env.action_space.sample()
            else:
                a = int(pi_ihr[int(s)])

            sp, _r, done, _ter, _tru, _info = _step_env(env, int(a))
            model.update(int(s), int(a), int(sp))
            s = sp
            steps += 1

        if (ep + 1) % cfg.plan_every_ihr == 0:
            P_hat = model.build_P_hat(is_terminal=is_terminal)
            V = unbounded_reachability_vi(P_hat, is_goal, is_hole, cfg.vi_max_iters, cfg.vi_eps)
            Astar = build_Astar(P_hat, V, is_goal, is_hole)
            pi_ihr, _u = speed_optimize_on_Astar(
                P_hat,
                V,
                Astar,
                is_goal,
                is_hole,
                cfg.u_max_iters,
                cfg.u_eps,
            )

            val_stats, _ = eval_policy_ihr(cfg, desc, map_size, pi_ihr, cfg.val_episodes_ihr, SEED_BASE_VAL_IHR, cap_steps=cap_eval)
            val_acc = float(val_stats.success_rate)

            checkpoints.append(ep + 1)
            val_acc_hist.append(val_acc)

            if val_acc > best_val_acc + 1e-12:
                best_val_acc = val_acc
                best_pi_ihr = pi_ihr.copy()
                best_value_ihr = V.copy()

            if cfg.enable_early_stop:
                stop, gap_streak, sm, gap = should_stop_by_gap(
                    ep_plus_1=ep + 1,
                    val_acc_hist=val_acc_hist,
                    theo_opt=p_star_start_ihr,
                    min_episodes=cfg.early_stop_min_ihr,
                    gap_tol=cfg.early_stop_gap_tol_ihr,
                    streak_need=cfg.early_stop_gap_streak,
                    streak_now=gap_streak,
                    smooth_k=cfg.early_stop_smooth_k,
                )
                if stop:
                    break

            iterator.set_postfix({"eps": f"{eps:.3f}", "val": f"{val_acc:.3f}"})

    return {
        "cap_eval": cap_eval,
        "best_pi_ihr": best_pi_ihr,
        "best_value_ihr": best_value_ihr,
        "checkpoints": checkpoints,
        "val_acc_hist": val_acc_hist,
    }


def main():
    # 随机种子
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    cfg = get_mode_config(IS_SLIPPERY)

    mode_tag = "stoch" if cfg.is_slippery else "det"
    fig_dir = f"figures_{mode_tag}"
    os.makedirs(fig_dir, exist_ok=True)

    for map_size in RUN_MAP_SIZES:
        desc = DEFAULT_DESC_4X4 if (USE_DEFAULT_MAP and map_size == 4) else DEFAULT_DESC_8X8

        p_star_fhr, p_star_ihr = compute_theoretical_optima(cfg, desc, map_size)

        train_fhr_out = train_fhr(cfg, desc, map_size, p_star_start_fhr=p_star_fhr)
        train_ihr_out = train_ihr(cfg, desc, map_size, p_star_start_ihr=p_star_ihr)

        H = train_fhr_out["H"]
        pi_fhr_best = train_fhr_out["best_pi_fhr"]
        cap_eval = train_ihr_out["cap_eval"]
        pi_ihr_best = train_ihr_out["best_pi_ihr"]

        test_fhr_stats, test_fhr_vis = eval_policy_fhr(cfg, desc, map_size, pi_fhr_best, cfg.test_episodes_fhr, SEED_BASE_TEST_FHR)
        test_ihr_stats, test_ihr_vis = eval_policy_ihr(cfg, desc, map_size, pi_ihr_best, cfg.test_episodes_ihr, SEED_BASE_TEST_IHR, cap_steps=cap_eval)

        print(f"Map {map_size}x{map_size} mode {mode_tag}")
        print(f"FHR steps {H} success {test_fhr_stats.success_rate:.3%} avg {test_fhr_stats.avg_steps_success:.2f}")
        print(f"IHR cap {cap_eval} success {test_ihr_stats.success_rate:.3%} avg {test_ihr_stats.avg_steps_success:.2f}")

        if cfg.plot_training_curves:
            fname = os.path.join(fig_dir, f"{map_size}x{map_size}_{mode_tag}_training_curves.png")
            plot_training_curves(
                checkpoints_fhr=train_fhr_out["checkpoints"],
                val_acc_fhr=train_fhr_out["val_acc_hist"],
                checkpoints_ihr=train_ihr_out["checkpoints"],
                val_acc_ihr=train_ihr_out["val_acc_hist"],
                filename=fname,
            )

        if cfg.plot_visit_heatmap:
            if cfg.heatmap_from.upper() == "FHR":
                visited = test_fhr_vis
            elif cfg.heatmap_from.upper() == "IHR":
                visited = test_ihr_vis
            else:
                visited = test_fhr_vis + test_ihr_vis

            fname = os.path.join(fig_dir, f"{map_size}x{map_size}_{mode_tag}_visited_states.png")
            plot_visited_states_heatmap(visited, desc, map_size, cfg.heatmap_log_scale, fname)


if __name__ == "__main__":
    main()
