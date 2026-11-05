"""SARSA (on-policy TD control)."""
from agents.base_agent import TabularAgentBase


class SarsaAgent(TabularAgentBase):
    def update(self, state, action, reward, next_state, next_action, done):
        q_sa = self.Q[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state][next_action]
        self.Q[state][action] = q_sa + self.alpha * (target - q_sa)
