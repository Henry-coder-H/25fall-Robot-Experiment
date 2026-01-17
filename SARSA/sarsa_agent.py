import numpy as np
from pathlib import Path
from typing import Optional, Union


class SarsaAgent:
    """Tabular SARSA agent for discrete state/action spaces."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        q_table: Optional[np.ndarray] = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        if q_table is None:
            self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)
        else:
            self.q_table = np.array(q_table, dtype=np.float32)
            if self.q_table.shape != (n_states, n_actions):
                raise ValueError(
                    f"Q-table shape {self.q_table.shape} does not match ({n_states}, {n_actions})"
                )

    def choose_action(self, state: int, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon
        if np.random.rand() < eps:
            return int(np.random.randint(self.n_actions))
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_q))
        return int(np.random.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: Optional[int],
        done: bool,
    ) -> None:
        target = reward
        if not done and next_action is not None:
            # On-policy bootstrap with the next action chosen by the same policy
            target += self.gamma * self.q_table[next_state, next_action]
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.0,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.0,
    ) -> "SarsaAgent":
        q_table = np.load(path)
        n_states, n_actions = q_table.shape
        agent = cls(
            n_states,
            n_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            q_table=q_table,
        )
        return agent
