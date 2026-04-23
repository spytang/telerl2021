"""Variant RAN slicing environment with risk-aware reward shaping."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from ran_env import RANSlicingEnv


class RANSlicingEnvVar(RANSlicingEnv):
    """Risk-aware variant used to train a stabler policy than vanilla PPO.

    The reward keeps the original objective but adds two smooth penalties:
    1) short-term eMBB throughput variance, and
    2) puncturing imbalance across subcarriers inside the current slot.

    This nudges PPO away from repeatedly puncturing one subcarrier, which is
    a common failure mode that creates outages and unstable learning curves.
    """

    def __init__(
        self,
        alpha: float = 0.02,
        beta: float = 0.12,
        window_size: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.throughput_window: Deque[float] = deque(maxlen=self.window_size)

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

        embb_throughput = info["embb_throughput"]
        base_reward = info["base_reward"]

        self.throughput_window.append(embb_throughput)

        if len(self.throughput_window) >= 2:
            variance_penalty = self.alpha * float(np.var(self.throughput_window))
        else:
            variance_penalty = 0.0

        puncture_imbalance = float(np.std(self._puncture_counts / self.minislots_per_slot))
        imbalance_penalty = self.beta * puncture_imbalance

        reward = base_reward - variance_penalty - imbalance_penalty
        info["variance_penalty"] = variance_penalty
        info["imbalance_penalty"] = imbalance_penalty
        return obs, float(reward), terminated, truncated, info
