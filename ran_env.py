"""Gymnasium environment for URLLC/eMBB slicing with puncturing actions."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RANSlicingEnv(gym.Env):
    """RAN slicing environment based on a minislot puncturing model."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        F: int = 12,
        Sigma: int = 10,
        minislots_per_slot: int = 14,
        lam: float = 0.5,
        deadline_D: int = 3,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.F = F
        self.Sigma = Sigma
        self.minislots_per_slot = minislots_per_slot
        self.T = self.Sigma * self.minislots_per_slot
        self.lam = lam
        self.deadline_D = deadline_D

        # Actions: 0..F-1 puncture selected subcarrier, F means defer.
        self.action_space = spaces.Discrete(self.F + 1)

        # State: [queue_len, min_deadline, puncture_counts(F)] all normalized to [0,1].
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2 + self.F,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._queue: deque[int] = deque()
        self._t = 0
        self._slot_idx = 0
        self._puncture_counts = np.zeros(self.F, dtype=np.int32)
        self._cw_tolerances = np.ones(self.F, dtype=np.int32)
        self._cw_outage_flags = np.zeros(self.F, dtype=bool)

    def _init_slot_codewords(self) -> None:
        self._puncture_counts.fill(0)
        self._cw_outage_flags.fill(False)
        self._cw_tolerances = self._rng.integers(1, 5, size=self.F, endpoint=False, dtype=np.int32)

    def _get_obs(self) -> np.ndarray:
        queue_len_norm = min(len(self._queue), self.T) / self.T
        if self._queue:
            min_deadline_norm = max(min(self._queue), 0) / self.deadline_D
        else:
            min_deadline_norm = 0.0

        puncture_norm = np.clip(self._puncture_counts / self.minislots_per_slot, 0.0, 1.0)
        obs = np.concatenate(([queue_len_norm, min_deadline_norm], puncture_norm)).astype(np.float32)
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._queue.clear()
        self._t = 0
        self._slot_idx = 0
        self._init_slot_codewords()

        return self._get_obs(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.F}].")
        if self._t >= self.T:
            raise RuntimeError("Episode is done. Call reset() before step().")

        current_slot = self._t // self.minislots_per_slot
        if current_slot != self._slot_idx:
            self._slot_idx = current_slot
            self._init_slot_codewords()

        # New URLLC arrivals for this minislot.
        arrivals = int(self._rng.poisson(self.lam))
        for _ in range(arrivals):
            self._queue.append(self.deadline_D)

        embb_outages_this_step = 0

        # Serve at most one packet, prioritizing oldest (FIFO queue).
        if self._queue and action < self.F:
            self._queue.popleft()
            self._puncture_counts[action] += 1
            if (
                not self._cw_outage_flags[action]
                and self._puncture_counts[action] > self._cw_tolerances[action]
            ):
                self._cw_outage_flags[action] = True
                embb_outages_this_step += 1

        # Deadline progression and violation accounting.
        updated_queue: deque[int] = deque()
        urllc_latency_violations = 0
        while self._queue:
            rem = self._queue.popleft() - 1
            if rem <= 0:
                urllc_latency_violations += 1
            else:
                updated_queue.append(rem)
        self._queue = updated_queue

        reward = -1.0 * urllc_latency_violations - 0.5 * embb_outages_this_step

        self._t += 1
        terminated = self._t >= self.T
        truncated = False

        info = {
            "time_step": self._t,
            "arrivals": arrivals,
            "queue_length": len(self._queue),
            "urllc_latency_violations": urllc_latency_violations,
            "embb_outages_this_step": embb_outages_this_step,
            "slot_idx": self._slot_idx,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        min_deadline = min(self._queue) if self._queue else None
        print(
            f"t={self._t}/{self.T} "
            f"slot={self._slot_idx}/{self.Sigma - 1} "
            f"queue={len(self._queue)} "
            f"min_deadline={min_deadline} "
            f"punctures={self._puncture_counts.tolist()} "
            f"tolerances={self._cw_tolerances.tolist()}"
        )


if __name__ == "__main__":
    env = RANSlicingEnv()
    episode_returns = []

    for ep in range(3):
        obs, info = env.reset(seed=ep)
        done = False
        total_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_returns.append(total_reward)
        print(f"Episode {ep + 1}: return={total_reward:.3f}")

    print(f"Mean reward over 3 random episodes: {np.mean(episode_returns):.3f}")
