"""Variant RAN slicing environment with throughput-variance penalty."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from ran_env import RANSlicingEnv


class RANSlicingEnvVar(RANSlicingEnv):
    """Variance penalty encourages risk-averse puncturing, distributing
    eMBB disruption evenly across codewords (Alsenwi et al., 2021)."""

    def __init__(self, alpha: float = 0.05, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.throughput_window: Deque[float] = deque(maxlen=20)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        obs, info = super().reset(seed=seed, options=options)
        self.throughput_window.clear()
        return obs, info

    def step(self, action: int):
        obs, _, terminated, truncated, info = super().step(action)

        urllc_latency_violations = info["urllc_latency_violations"]
        embb_outages_this_step = info["embb_outages_this_step"]
        embb_throughput = (self.F - embb_outages_this_step) / self.F

        base_reward = (
            -1.0 * urllc_latency_violations
            - 0.5 * embb_outages_this_step
            + 0.3 * embb_throughput
        )

        self.throughput_window.append(embb_throughput)

        if len(self.throughput_window) >= 2:
            variance_penalty = self.alpha * np.var(self.throughput_window)
        else:
            variance_penalty = 0.0

        reward = base_reward - variance_penalty
        return obs, float(reward), terminated, truncated, info
