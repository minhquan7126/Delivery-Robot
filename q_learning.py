"""Q-learning (off-policy TD control)."""
import numpy as np
from agents.base_agent import TabularAgentBase


class QLearningAgent(TabularAgentBase):
    def update(self, state, action, reward, next_state, done, next_action_mask=None):
        q_sa = self.Q[state][action]
        if done:
            target = reward
        else:
            next_q = self.Q[next_state]
            if next_action_mask is not None:
                next_q = np.where(next_action_mask, next_q, -np.inf)
                best_next = np.max(next_q) if np.any(next_action_mask) else 0.0
            else:
                best_next = np.max(next_q)
            target = reward + self.gamma * best_next

        self.Q[state][action] = q_sa + self.alpha * (target - q_sa)
