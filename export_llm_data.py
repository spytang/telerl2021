"""Export per-figure tabular data for LLM-friendly analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from evaluate_and_plot import (
    EPISODE_LENGTH,
    EPISODES,
    evaluate_fixed_rr,
    evaluate_ppo,
    load_eval_curve,
)


def _build_per_episode_df(agent: str, metrics: Dict[str, np.ndarray]) -> pd.DataFrame:
    n = len(metrics["total_reward"])
    return pd.DataFrame(
        {
            "agent": [agent] * n,
            "episode": np.arange(n, dtype=int),
            "total_reward": metrics["total_reward"],
            "total_violations": metrics["total_violations"],
            "total_outages": metrics["total_outages"],
            "viol_rate_per_step": metrics["total_violations"] / EPISODE_LENGTH,
            "outage_rate_per_step": metrics["total_outages"] / EPISODE_LENGTH,
            "per_episode_embb_var": metrics["per_episode_embb_var"],
        }
    )


def _summary_rows(agent: str, metrics: Dict[str, np.ndarray]) -> list[dict]:
    return [
        {
            "agent": agent,
            "metric": "reward",
            "mean": float(np.mean(metrics["total_reward"])),
            "std": float(np.std(metrics["total_reward"])),
        },
        {
            "agent": agent,
            "metric": "urllc_violation_rate_per_step",
            "mean": float(np.mean(metrics["total_violations"] / EPISODE_LENGTH)),
            "std": float(np.std(metrics["total_violations"] / EPISODE_LENGTH)),
        },
        {
            "agent": agent,
            "metric": "embb_outage_rate_per_step",
            "mean": float(np.mean(metrics["total_outages"] / EPISODE_LENGTH)),
            "std": float(np.std(metrics["total_outages"] / EPISODE_LENGTH)),
        },
        {
            "agent": agent,
            "metric": "embb_throughput_variance",
            "mean": float(np.mean(metrics["per_episode_embb_var"])),
            "std": float(np.std(metrics["per_episode_embb_var"])),
        },
    ]


def export_llm_data(output_dir: Path, n_episodes: int = EPISODES) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = evaluate_ppo("./models/baseline_ppo/final_model", n_episodes=n_episodes)
    var_metrics = evaluate_ppo("./models/var_ppo/final_model", n_episodes=n_episodes)
    rr_metrics = evaluate_fixed_rr(n_episodes=n_episodes)

    # Figure 1 data: training curves.
    baseline_ts, baseline_mean, baseline_std = load_eval_curve("./logs/baseline_ppo/evaluations.npz")
    var_ts, var_mean, var_std = load_eval_curve("./logs/var_ppo/evaluations.npz")
    baseline_mean_smooth = pd.Series(baseline_mean).rolling(window=5, min_periods=1, center=True).mean().to_numpy()
    var_mean_smooth = pd.Series(var_mean).rolling(window=5, min_periods=1, center=True).mean().to_numpy()

    fig1_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "agent": "baseline_ppo",
                    "timesteps": baseline_ts,
                    "mean_reward": baseline_mean,
                    "std_reward": baseline_std,
                    "smoothed_mean_reward": baseline_mean_smooth,
                    "lower": baseline_mean - baseline_std,
                    "upper": baseline_mean + baseline_std,
                }
            ),
            pd.DataFrame(
                {
                    "agent": "var_ppo",
                    "timesteps": var_ts,
                    "mean_reward": var_mean,
                    "std_reward": var_std,
                    "smoothed_mean_reward": var_mean_smooth,
                    "lower": var_mean - var_std,
                    "upper": var_mean + var_std,
                }
            ),
        ],
        ignore_index=True,
    )
    fig1_df.to_csv(output_dir / "figure1_training_curves.csv", index=False)

    # Figure 2 data: per-agent aggregate + per-episode source values.
    summary_rows = []
    for agent, metrics in (
        ("fixed_rr", rr_metrics),
        ("baseline_ppo", baseline_metrics),
        ("var_ppo", var_metrics),
    ):
        summary_rows.extend(_summary_rows(agent, metrics))

    pd.DataFrame(summary_rows).to_csv(output_dir / "figure2_performance_summary.csv", index=False)

    per_episode_df = pd.concat(
        [
            _build_per_episode_df("fixed_rr", rr_metrics),
            _build_per_episode_df("baseline_ppo", baseline_metrics),
            _build_per_episode_df("var_ppo", var_metrics),
        ],
        ignore_index=True,
    )
    per_episode_df.to_csv(output_dir / "figure2_per_episode_metrics.csv", index=False)

    # Figure 3 data: boxplot / scatter distribution source.
    fig3_df = per_episode_df[["agent", "episode", "per_episode_embb_var"]].copy()
    fig3_df.to_csv(output_dir / "figure3_variance_distribution.csv", index=False)

    # Optional machine-readable manifest for prompting LLMs.
    manifest = {
        "episode_length": EPISODE_LENGTH,
        "n_episodes": int(n_episodes),
        "files": {
            "figure1": "figure1_training_curves.csv",
            "figure2_summary": "figure2_performance_summary.csv",
            "figure2_source": "figure2_per_episode_metrics.csv",
            "figure3": "figure3_variance_distribution.csv",
        },
        "agents": ["fixed_rr", "baseline_ppo", "var_ppo"],
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Export complete: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-figure CSV/JSON data for LLM analysis.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./llm_exports"),
        help="Directory to write exported files.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EPISODES,
        help="Number of evaluation episodes for policy metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_llm_data(output_dir=args.output_dir, n_episodes=args.episodes)
