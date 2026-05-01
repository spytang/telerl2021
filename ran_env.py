"""Gymnasium environment for URLLC/eMBB slicing with puncturing actions."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

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
        arrival_rate: float = 0.5,
        deadline_D: int = 3,
        obs_mode: str = "minimal",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        if obs_mode not in {"minimal", "rich"}:
            raise ValueError("obs_mode must be either 'minimal' or 'rich'.")

        self.F = F
        self.Sigma = Sigma
        self.minislots_per_slot = minislots_per_slot
        self.T = self.Sigma * self.minislots_per_slot
        self.arrival_rate = arrival_rate
        self.deadline_D = deadline_D
        self.obs_mode = obs_mode
        self.max_cw_tolerance = 4

        # Actions: 0..F-1 puncture selected subcarrier, F means defer.
        self.action_space = spaces.Discrete(self.F + 1)

        # Minimal state: [queue_len, min_deadline, puncture_counts(F)].
        # Rich state augments it with deadline/time/reliability-margin context.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._obs_dim(),),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._queue: Deque[int] = deque()
        self._t = 0
        self._slot_idx = 0
        self._puncture_counts = np.zeros(self.F, dtype=np.int32)
        self._cw_tolerances = np.ones(self.F, dtype=np.int32)
        self._cw_outage_flags = np.zeros(self.F, dtype=bool)
        self._prev_arrivals = 0
        self._prev_served = 0
        self._prev_violations = 0
        self._prev_outages = 0

    def _obs_dim(self) -> int:
        if self.obs_mode == "minimal":
            return 2 + self.F

        # queue_len, min_deadline, deadline histogram(D), slot/minislot/remaining,
        # puncture_counts(F), true Cw(F), remaining margin(F), outage flags(F),
        # previous arrivals/served/violations/outages.
        return 2 + self.deadline_D + 3 + (4 * self.F) + 4

    def _init_slot_codewords(self) -> None:
        self._puncture_counts.fill(0)
        self._cw_outage_flags.fill(False)
        self._cw_tolerances = self._rng.integers(1, 5, size=self.F, endpoint=False, dtype=np.int32)

    def _time_features(self) -> Tuple[float, float, float]:
        if self.T <= 1:
            return 0.0, 0.0, 0.0

        effective_t = min(self._t, self.T - 1)
        slot_idx = min(effective_t // self.minislots_per_slot, self.Sigma - 1)
        minislot_idx = effective_t % self.minislots_per_slot
        remaining_minislots = 0 if self._t >= self.T else self.minislots_per_slot - minislot_idx

        slot_norm = slot_idx / max(self.Sigma - 1, 1)
        minislot_norm = minislot_idx / max(self.minislots_per_slot - 1, 1)
        remaining_norm = remaining_minislots / self.minislots_per_slot
        return float(slot_norm), float(minislot_norm), float(remaining_norm)

    def _deadline_histogram(self) -> np.ndarray:
        hist = np.zeros(self.deadline_D, dtype=np.float32)
        for rem_deadline in self._queue:
            idx = int(np.clip(rem_deadline, 1, self.deadline_D)) - 1
            hist[idx] += 1.0
        return np.clip(hist / self.T, 0.0, 1.0)

    def _get_obs(self) -> np.ndarray:
        queue_len_norm = min(len(self._queue), self.T) / self.T
        if self._queue:
            min_deadline_norm = max(min(self._queue), 0) / self.deadline_D
        else:
            min_deadline_norm = 0.0

        puncture_norm = np.clip(self._puncture_counts / self.minislots_per_slot, 0.0, 1.0)
        minimal_obs = np.concatenate(([queue_len_norm, min_deadline_norm], puncture_norm))
        if self.obs_mode == "minimal":
            return minimal_obs.astype(np.float32)

        # In this tolerance-aware MDP version, Cw is treated as base-station
        # observable/estimable eMBB reliability margin from CSI, MCS/BLER
        # estimation, or decoder feedback rather than as a hidden variable.
        cw_norm = self._cw_tolerances / self.max_cw_tolerance
        remaining_margin = self._cw_tolerances - self._puncture_counts
        margin_norm = np.clip(remaining_margin / self.max_cw_tolerance, -1.0, 1.0)
        outage_flags = self._cw_outage_flags.astype(np.float32)
        slot_norm, minislot_norm, remaining_norm = self._time_features()

        prev_features = np.array(
            [
                min(self._prev_arrivals, self.T) / self.T,
                self._prev_served,
                min(self._prev_violations, self.T) / self.T,
                min(self._prev_outages, self.F) / self.F,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate(
            (
                [queue_len_norm, min_deadline_norm],
                self._deadline_histogram(),
                [slot_norm, minislot_norm, remaining_norm],
                puncture_norm,
                cw_norm,
                margin_norm,
                outage_flags,
                prev_features,
            )
        ).astype(np.float32)
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._queue.clear()
        self._t = 0
        self._slot_idx = 0
        self._prev_arrivals = 0
        self._prev_served = 0
        self._prev_violations = 0
        self._prev_outages = 0
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
        arrivals = int(self._rng.poisson(self.arrival_rate))
        for _ in range(arrivals):
            self._queue.append(self.deadline_D)

        embb_outages_this_step = 0
        served_this_step = 0

        # Serve at most one packet, prioritizing oldest (FIFO queue).
        if self._queue and action < self.F:
            self._queue.popleft()
            served_this_step = 1
            self._puncture_counts[action] += 1
            if (
                not self._cw_outage_flags[action]
                and self._puncture_counts[action] > self._cw_tolerances[action]
            ):
                self._cw_outage_flags[action] = True
                embb_outages_this_step += 1

        # Deadline progression and violation accounting.
        updated_queue: Deque[int] = deque()
        urllc_latency_violations = 0
        while self._queue:
            rem = self._queue.popleft() - 1
            if rem <= 0:
                urllc_latency_violations += 1
            else:
                updated_queue.append(rem)
        self._queue = updated_queue

        reward = -1.0 * urllc_latency_violations - 0.5 * embb_outages_this_step
        remaining_margin = self._cw_tolerances - self._puncture_counts
        puncture_norm = self._puncture_counts / self.minislots_per_slot
        margin_norm = np.clip(remaining_margin / self.max_cw_tolerance, -1.0, 1.0)
        puncture_count_variance = float(np.var(self._puncture_counts))
        puncture_count_norm_variance = float(np.var(puncture_norm))
        remaining_margin_variance = float(np.var(remaining_margin))
        remaining_margin_norm_variance = float(np.var(margin_norm))
        embb_throughput_variance = float(np.var(1.0 - self._cw_outage_flags.astype(np.float32)))

        self._t += 1
        terminated = self._t >= self.T
        truncated = False
        if not terminated:
            next_slot = self._t // self.minislots_per_slot
            if next_slot != self._slot_idx:
                self._slot_idx = next_slot
                self._init_slot_codewords()

        self._prev_arrivals = arrivals
        self._prev_served = served_this_step
        self._prev_violations = urllc_latency_violations
        self._prev_outages = embb_outages_this_step

        info = {
            "time_step": self._t,
            "arrivals": arrivals,
            "served": served_this_step,
            "queue_length": len(self._queue),
            "urllc_latency_violations": urllc_latency_violations,
            "embb_outages_this_step": embb_outages_this_step,
            "slot_idx": self._slot_idx,
            "action_slot_idx": current_slot,
            "puncture_count_variance": puncture_count_variance,
            "puncture_count_norm_variance": puncture_count_norm_variance,
            "remaining_margin_variance": remaining_margin_variance,
            "remaining_margin_norm_variance": remaining_margin_norm_variance,
            "embb_throughput_variance": embb_throughput_variance,
            "common_score_step": float(reward),
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
