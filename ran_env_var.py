"""Variant RAN slicing environment with throughput-variance penalty."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ran_env import RANSlicingEnv


class RANSlicingEnvVar(RANSlicingEnv):
    """Variance penalty encourages risk-averse puncturing, distributing
    eMBB disruption evenly across codewords (Alsenwi et al., 2021)."""

    def __init__(self, alpha: float = 0.3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.throughput_window: deque[float] = deque(maxlen=20)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.throughput_window.clear()
        return obs, info

    def step(self, action: int):
        obs, baseline_reward, terminated, truncated, info = super().step(action)

        embb_outages_this_step = info["embb_outages_this_step"]
        embb_throughput = (self.F - embb_outages_this_step) / self.F
        self.throughput_window.append(embb_throughput)

        if len(self.throughput_window) >= 2:
            variance_penalty = self.alpha * np.var(self.throughput_window)
        else:
            variance_penalty = 0.0

        reward = baseline_reward - variance_penalty
        return obs, float(reward), terminated, truncated, info
