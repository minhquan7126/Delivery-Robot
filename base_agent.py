

from __future__ import annotations
import random
from collections import defaultdict
import numpy as np


class TabularAgentBase:
    def __init__(self, n_actions: int, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay_episodes: int = 8000, seed: int = 0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Q[state] -> np.array shape (n_actions,)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))

    def epsilon_at(self, episode: int) -> float:
        frac = min(1.0, episode / max(1, self.epsilon_decay_episodes))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
    
    def select_action(self, state: tuple, epsilon: float, action_mask=None) -> int:
        """epsilon-greedy, ho tro action_mask (Cach B - loai bo action khong hop le)."""
        valid_actions = np.arange(self.n_actions)
        if action_mask is not None:
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) == 0:
                valid_actions = np.arange(self.n_actions)  # fallback an toan

        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid_actions))

        q_values = self.Q[state]
        if action_mask is not None:
            masked_q = np.where(action_mask, q_values, -np.inf)
            return int(np.argmax(masked_q))
        return int(np.argmax(q_values))

    def greedy_action(self, state: tuple, action_mask=None) -> int:
        return self.select_action(state, epsilon=0.0, action_mask=action_mask)

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def n_states_visited(self) -> int:
        return len(self.Q)
