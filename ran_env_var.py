"""Variant RAN slicing environment with throughput-variance penalty."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from ran_env import RANSlicingEnv


class RANSlicingEnvVar(RANSlicingEnv):
    """Tolerance-aware reward variant with load/margin/throughput variance penalty."""

    def __init__(
        self,
        alpha: float = 0.3,
        load_variance_weight: float = 0.5,
        margin_variance_weight: float = 0.5,
        throughput_variance_weight: float = 0.5,
        rolling_variance_weight: float = 0.2,
        throughput_window_size: int = 20,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("obs_mode", "rich")
        super().__init__(**kwargs)
        self.alpha = alpha
        self.load_variance_weight = load_variance_weight
        self.margin_variance_weight = margin_variance_weight
        self.throughput_variance_weight = throughput_variance_weight
        self.rolling_variance_weight = rolling_variance_weight
        self.throughput_window: Deque[float] = deque(maxlen=throughput_window_size)

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
        obs, baseline_reward, terminated, truncated, info = super().step(action)

        embb_outages_this_step = info["embb_outages_this_step"]
        embb_throughput = (self.F - embb_outages_this_step) / self.F
        self.throughput_window.append(embb_throughput)

        load_variance = float(info.get("puncture_count_norm_variance", 0.0))
        margin_variance = float(info.get("remaining_margin_norm_variance", 0.0))
        throughput_variance = float(info.get("embb_throughput_variance", 0.0))
        rolling_variance = float(np.var(self.throughput_window)) if len(self.throughput_window) >= 2 else 0.0

        variance_penalty = self.alpha * (
            self.load_variance_weight * load_variance
            + self.margin_variance_weight * margin_variance
            + self.throughput_variance_weight * throughput_variance
            + self.rolling_variance_weight * rolling_variance
        )

        reward = baseline_reward - variance_penalty
        info.update(
            {
                "load_variance_penalty": load_variance,
                "margin_variance_penalty": margin_variance,
                "throughput_variance_penalty": throughput_variance,
                "rolling_throughput_variance_penalty": rolling_variance,
                "variance_penalty": float(variance_penalty),
                "common_score_step": float(baseline_reward),
            }
        )
        return obs, float(reward), terminated, truncated, info
