"""Gymnasium environment for URLLC/eMBB slicing with puncturing actions."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

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
        reward_profile: str = "default",
        include_channel_state: Optional[bool] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.F = F
        self.Sigma = Sigma
        self.minislots_per_slot = minislots_per_slot
        self.T = self.Sigma * self.minislots_per_slot
        self.arrival_rate = arrival_rate
        self.deadline_D = deadline_D
        self.reward_profile = reward_profile
        if include_channel_state is None:
            # In research profiles we expose channel/codeword stress proxy
            # to avoid unnecessary partial observability.
            self.include_channel_state = reward_profile in {"risk_aware"}
        else:
            self.include_channel_state = include_channel_state

        # Actions: 0..F-1 puncture selected subcarrier, F means defer.
        self.action_space = spaces.Discrete(self.F + 1)

        # State: [queue_len, min_deadline, puncture_counts(F), optional budget_remaining(F)].
        obs_dim = 2 + self.F + (self.F if self.include_channel_state else 0)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._queue: Deque[int] = deque()
        self._t = 0
        self._slot_idx = 0
        self._puncture_counts = np.zeros(self.F, dtype=np.int32)
        self._cw_tolerances = np.ones(self.F, dtype=np.int32)
        self._cw_outage_flags = np.zeros(self.F, dtype=bool)
        self._last_action_deferred = False

    def _compute_reward(
        self,
        urllc_latency_violations: int,
        embb_outages_this_step: int,
        queue_length: int,
        action_deferred: bool,
        queue_urgency: float,
        embb_risk_proxy: float,
    ) -> float:
        if self.reward_profile == "default":
            return -1.0 * urllc_latency_violations - 0.5 * embb_outages_this_step

        if self.reward_profile == "urllc_heavy":
            # Research profile: prioritize URLLC deadline protection while
            # preserving a soft eMBB reliability signal and queue-pressure cost.
            return (
                -2.0 * urllc_latency_violations
                - 0.25 * embb_outages_this_step
                - 0.05 * queue_length
                - (0.02 if action_deferred and queue_length > 0 else 0.0)
            )

        if self.reward_profile == "risk_aware":
            # Research profile:
            # 1) URLLC violations receive dynamic weight when queue is urgent.
            # 2) eMBB is protected by outage + pre-outage risk proxy.
            # 3) defer is discouraged only when backlog exists.
            dynamic_urllc_weight = 1.8 + 1.7 * queue_urgency
            return (
                -dynamic_urllc_weight * urllc_latency_violations
                - 0.20 * embb_outages_this_step
                - 0.25 * embb_risk_proxy
                - 0.05 * queue_length
                - (0.05 + 0.10 * queue_urgency if action_deferred and queue_length > 0 else 0.0)
            )

        raise ValueError(
            f"Unknown reward_profile='{self.reward_profile}'. "
            "Use one of: default, urllc_heavy, risk_aware."
        )

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
        obs_parts = [np.array([queue_len_norm, min_deadline_norm], dtype=np.float32), puncture_norm.astype(np.float32)]
        if self.include_channel_state:
            # Remaining puncture budget proxy per codeword. In practical systems,
            # this can be estimated from CQI/MCS/HARQ-level knowledge.
            budget_remaining = np.clip(
                (self._cw_tolerances - self._puncture_counts) / np.maximum(self._cw_tolerances, 1),
                0.0,
                1.0,
            ).astype(np.float32)
            obs_parts.append(budget_remaining)

        obs = np.concatenate(obs_parts).astype(np.float32)
        return obs

    def _queue_urgency(self) -> float:
        if not self._queue:
            return 0.0

        min_deadline = min(self._queue)
        deadline_pressure = 1.0 - (max(min_deadline, 0) / self.deadline_D)
        backlog_pressure = min(len(self._queue), self.T) / self.T
        return float(np.clip(0.75 * deadline_pressure + 0.25 * backlog_pressure, 0.0, 1.0))

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
        self._init_slot_codewords()
        self._last_action_deferred = False

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
        embb_risk_proxy = 0.0

        # Serve at most one packet, prioritizing oldest (FIFO queue).
        if self._queue and action < self.F:
            self._queue.popleft()
            self._puncture_counts[action] += 1
            embb_risk_proxy = float(
                self._puncture_counts[action] / max(int(self._cw_tolerances[action]), 1)
            )
            if (
                not self._cw_outage_flags[action]
                and self._puncture_counts[action] > self._cw_tolerances[action]
            ):
                self._cw_outage_flags[action] = True
                embb_outages_this_step += 1
            self._last_action_deferred = False
        else:
            self._last_action_deferred = bool(self._queue and action == self.F)

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
        queue_urgency = self._queue_urgency()
        puncture_concentration = float(np.var(self._puncture_counts / max(self.minislots_per_slot, 1)))

        reward = self._compute_reward(
            urllc_latency_violations=urllc_latency_violations,
            embb_outages_this_step=embb_outages_this_step,
            queue_length=len(self._queue),
            action_deferred=self._last_action_deferred,
            queue_urgency=queue_urgency,
            embb_risk_proxy=embb_risk_proxy,
        )

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
            "reward_profile": self.reward_profile,
            "action_deferred": self._last_action_deferred,
            "queue_urgency": queue_urgency,
            "embb_risk_proxy": embb_risk_proxy,
            "puncture_concentration": puncture_concentration,
            "include_channel_state": self.include_channel_state,
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
