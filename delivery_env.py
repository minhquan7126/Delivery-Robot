from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class _DiscreteSpace:
        def __init__(self, n):
            self.n = n

        def contains(self, x):
            return isinstance(x, (int, np.integer)) and 0 <= x < self.n

        def sample(self):
            return int(np.random.randint(self.n))

    class _MultiDiscreteSpace:
        def __init__(self, dims):
            self.nvec = np.array(dims)

    class _spaces:
        Discrete = _DiscreteSpace
        MultiDiscrete = _MultiDiscreteSpace

    spaces = _spaces()

    class _GymEnvStub:
        metadata = {}

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)
            return None

    class _gym:
        Env = _GymEnvStub

    gym = _gym()


ACTIONS = ["Up", "Down", "Left", "Right", "Pickup", "Drop"]
ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_PICKUP, ACTION_DROP = range(6)

# (d_row, d_col) cho 4 huong di chuyen
MOVE_DELTA = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}


@dataclass
class RewardConfig:
    """Cau hinh reward - cho phep thu nhieu bo tham so khac nhau (yeu cau thuc nghiem)."""
    step_penalty: float = -1.0
    invalid_action_penalty: float = -10.0
    pickup_success_reward: float = -1.0     # thuong la khong thuong lon, chi -1 nhu 1 buoc binh thuong
    success_reward: float = 50.0
    timeout_penalty: float = -20.0
    express_step_penalty: float = -0.5      # step penalty khi express dang active (setting 4)


def load_map(path: str) -> list[list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip("\n") != ""]
    width = max(len(line) for line in lines)
    grid = [list(line.ljust(width, ".")) for line in lines]
    return grid


class DeliveryRobotEnv(gym.Env):

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        map_path: str,
        enable_deadline: bool = False,
        max_steps: int = 30,
        enable_express: bool = False,
        invalid_action_mode: str = "penalty",  # 'penalty' | 'mask'
        reward_config: Optional[RewardConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert invalid_action_mode in ("penalty", "mask")

        self.grid = load_map(map_path)
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])

        self.enable_deadline = enable_deadline
        self.max_steps = max_steps
        self.enable_express = enable_express
        self.invalid_action_mode = invalid_action_mode
        self.reward_config = reward_config or RewardConfig()
        self.render_mode = render_mode

        # Tim vi tri dac biet tren map
        self.start_pos = (0, 0)
        self.pickup_pos = None
        self.dest_pos = None
        self.express_pos = None
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                ch = self.grid[r][c]
                if ch == "S":
                    self.start_pos = (r, c)
                elif ch == "P":
                    self.pickup_pos = (r, c)
                elif ch == "D":
                    self.dest_pos = (r, c)
                elif ch == "E":
                    self.express_pos = (r, c)
        assert self.pickup_pos is not None, "Map phai co 1 o 'P' (Pickup)"
        assert self.dest_pos is not None, "Map phai co 1 o 'D' (Destination)"
        if self.enable_express:
            assert self.express_pos is not None, "enable_express=True nhung map khong co o 'E'"

        # Gymnasium spaces
        self.action_space = spaces.Discrete(6)

        dims = [self.n_rows, self.n_cols, 2]  # row, col, carrying
        if self.enable_deadline:
            dims.append(self.max_steps + 1)  # steps_remaining: 0..max_steps
        if self.enable_express:
            dims.append(6)  # express_timer: 0..5
            dims.append(2)  # express_used: 0/1
        self._state_dims = dims
        self.observation_space = spaces.MultiDiscrete(dims)

        # bien noi bo cua episode hien tai
        self._row = 0
        self._col = 0
        self._carrying = 0
        self._steps_remaining = self.max_steps
        self._express_timer = 0
        self._express_used = 0

    # ------------------------------------------------------------------
    # Cac ham tien ich lien quan den state space (bao cao |S| x |A|)
    # ------------------------------------------------------------------
    @property
    def n_states(self) -> int:
        """Tich cac chieu cua state (bao gom ca o tuong - upper bound ly thuyet)."""
        return int(np.prod(self._state_dims))

    @property
    def n_actions(self) -> int:
        return self.action_space.n

    def describe_state_space(self) -> str:
        labels = ["row", "col", "carrying"]
        if self.enable_deadline:
            labels.append("steps_remaining")
        if self.enable_express:
            labels.extend(["express_timer", "express_used"])
        parts = ", ".join(f"{l}:{d}" for l, d in zip(labels, self._state_dims))
        return (
            f"State dims -> {parts}\n"
            f"|S| (upper bound) = {self.n_states}, |A| = {self.n_actions}, "
            f"|S x A| = {self.n_states * self.n_actions}"
        )

    # ------------------------------------------------------------------
    # State encode / decode - dung tuple lam key cho Q-table (dict-based)
    # ------------------------------------------------------------------
    def _get_state(self) -> tuple:
        state = [self._row, self._col, self._carrying]
        if self.enable_deadline:
            state.append(self._steps_remaining)
        if self.enable_express:
            state.append(self._express_timer)
            state.append(self._express_used)
        return tuple(state)

    def is_wall(self, r: int, c: int) -> bool:
        if r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols:
            return True
        return self.grid[r][c] == "#"

    # ------------------------------------------------------------------
    # Valid action mask - dung cho Cach B (invalid_action_mode='mask')
    # ------------------------------------------------------------------
    def valid_action_mask(self, state: Optional[tuple] = None) -> np.ndarray:
        """Tra ve mang bool[6] danh dau action nao hop le tai state hien tai."""
        if state is None:
            row, col, carrying = self._row, self._col, self._carrying
        else:
            row, col, carrying = state[0], state[1], state[2]

        mask = np.ones(6, dtype=bool)
        # Move actions: luon "hop le" ve mat cu phap (dung tuong se tu giu nguyen vi tri)
        # Pickup chi hop le neu dang o Pickup location va chua carrying
        mask[ACTION_PICKUP] = (carrying == 0) and ((row, col) == self.pickup_pos)
        # Drop chi hop le neu dang o Destination va dang carrying
        mask[ACTION_DROP] = (carrying == 1) and ((row, col) == self.dest_pos)
        return mask

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._row, self._col = self.start_pos
        self._carrying = 0
        self._steps_remaining = self.max_steps
        self._express_timer = 0
        self._express_used = 0
        return self._get_state(), {"action_mask": self.valid_action_mask()}

    def step(self, action: int):
        assert self.action_space.contains(action)
        rc = self.reward_config
        terminated = False
        truncated = False
        info = {}

        # step penalty phu thuoc express dang active hay khong (setting 4)
        if self.enable_express and self._express_timer > 0:
            reward = rc.express_step_penalty
        else:
            reward = rc.step_penalty

        if action in MOVE_DELTA:
            dr, dc = MOVE_DELTA[action]
            new_r, new_c = self._row + dr, self._col + dc
            if self.is_wall(new_r, new_c):
                # dung tuong / ra ngoai bien: giu nguyen vi tri, khong phat them
                new_r, new_c = self._row, self._col
            self._row, self._col = new_r, new_c

            # kich hoat express pass neu buoc vao dung o (setting 4)
            if self.enable_express and (self._row, self._col) == self.express_pos \
                    and self._express_used == 0:
                self._express_timer = 5
                self._express_used = 1

        elif action == ACTION_PICKUP:
            valid = (self._carrying == 0) and ((self._row, self._col) == self.pickup_pos)
            if valid:
                self._carrying = 1
                reward = rc.pickup_success_reward
            else:
                reward = rc.invalid_action_penalty
                info["invalid_action"] = "pickup"

        elif action == ACTION_DROP:
            valid = (self._carrying == 1) and ((self._row, self._col) == self.dest_pos)
            if valid:
                self._carrying = 0
                reward = rc.success_reward
                terminated = True
                info["success"] = True
            else:
                reward = rc.invalid_action_penalty
                info["invalid_action"] = "drop"

        # giam express timer sau moi buoc (neu dang active va khong vua duoc kich hoat)
        if self.enable_express and self._express_timer > 0 and not (
            action in MOVE_DELTA and (self._row, self._col) == self.express_pos
            and self._express_used == 1 and self._express_timer == 5
        ):
            self._express_timer = max(0, self._express_timer - 1)

        # xu ly deadline (setting 3, 4)
        if self.enable_deadline:
            self._steps_remaining = max(0, self._steps_remaining - 1)
            if self._steps_remaining == 0 and not terminated:
                reward = self.reward_config.timeout_penalty
                truncated = True
                info["timeout"] = True

        obs = self._get_state()
        info["action_mask"] = self.valid_action_mask()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Render don gian dang text - dung cho demo/evaluate.py
    # ------------------------------------------------------------------
    def render(self):
        display = [row.copy() for row in self.grid]
        r, c = self._row, self._col
        display[r][c] = "R" if self._carrying == 0 else "@"
        lines = ["".join(row) for row in display]
        header = f"carrying={self._carrying}"
        if self.enable_deadline:
            header += f" steps_left={self._steps_remaining}"
        if self.enable_express:
            header += f" express_timer={self._express_timer} express_used={self._express_used}"
        return header + "\n" + "\n".join(lines)


def make_env(setting: int, invalid_action_mode: str = "penalty",
             reward_config: Optional[RewardConfig] = None) -> DeliveryRobotEnv:
    """Factory tien loi de tao nhanh 1 trong 4 setting bat buoc cua de tai."""
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maps_dir = os.path.join(base, "maps")

    if setting == 1:
        return DeliveryRobotEnv(
            map_path=os.path.join(maps_dir, "map1_5x5.txt"),
            enable_deadline=False, enable_express=False,
            invalid_action_mode=invalid_action_mode, reward_config=reward_config,
        )
    elif setting == 2:
        return DeliveryRobotEnv(
            map_path=os.path.join(maps_dir, "map2_8x8_wall.txt"),
            enable_deadline=False, enable_express=False,
            invalid_action_mode=invalid_action_mode, reward_config=reward_config,
        )
    elif setting == 3:
        return DeliveryRobotEnv(
            map_path=os.path.join(maps_dir, "map3_8x8_deadline.txt"),
            enable_deadline=True, max_steps=30, enable_express=False,
            invalid_action_mode=invalid_action_mode, reward_config=reward_config,
        )
    elif setting == 4:
        return DeliveryRobotEnv(
            map_path=os.path.join(maps_dir, "map4_6x6_express.txt"),
            enable_deadline=True, max_steps=20, enable_express=True,
            invalid_action_mode=invalid_action_mode, reward_config=reward_config,
        )
    else:
        raise ValueError("setting phai la 1, 2, 3 hoac 4")
