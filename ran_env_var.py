"""Variant RAN slicing environment with throughput-variance penalty."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from ran_env import RANSlicingEnv


class RANSlicingEnvVar(RANSlicingEnv):
    """Variance penalty encourages risk-averse puncturing, distributing
    eMBB disruption evenly across codewords (Alsenwi et al., 2021)."""

    def __init__(self, alpha: float = 0.3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.outage_window: Deque[float] = deque(maxlen=20)
        self.concentration_window: Deque[float] = deque(maxlen=20)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        obs, info = super().reset(seed=seed, options=options)
        self.outage_window.clear()
        self.concentration_window.clear()
        return obs, info

    def step(self, action: int):
        obs, baseline_reward, terminated, truncated, info = super().step(action)

        embb_outage_ratio = info["embb_outages_this_step"] / self.F
        puncture_concentration = float(info.get("puncture_concentration", 0.0))
        queue_urgency = float(info.get("queue_urgency", 0.0))

        self.outage_window.append(embb_outage_ratio)
        self.concentration_window.append(puncture_concentration)

        if len(self.outage_window) >= 2:
            outage_burstiness = float(np.var(self.outage_window))
            concentration_mean = float(np.mean(self.concentration_window))
            raw_penalty = 0.7 * concentration_mean + 0.3 * outage_burstiness
            # Do not over-penalize when URLLC queue is urgent.
            variance_penalty = self.alpha * (1.0 - queue_urgency) * raw_penalty
        else:
            variance_penalty = 0.0

        reward = baseline_reward - variance_penalty
        return obs, float(reward), terminated, truncated, info
