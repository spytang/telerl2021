"""Evaluate trained rich-observation agents and generate comparison plots."""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ran_env import RANSlicingEnv

EPISODES = 100
EPISODE_LENGTH = 140
SEED_BASE = 42

PolicyFn = Callable[[np.ndarray, int, RANSlicingEnv], int]


def apply_style() -> None:
    """Apply plotting style with fallback requested by user."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        if "seaborn-paper" in plt.style.available:
            plt.style.use("seaborn-paper")


def evaluate_policy(policy_fn: PolicyFn, n_episodes: int = EPISODES) -> Dict[str, np.ndarray]:
    """Evaluate any policy on the common tolerance-aware MDP and common score."""
    env = RANSlicingEnv(obs_mode="rich")

    common_scores: List[float] = []
    violations: List[int] = []
    outages: List[int] = []
    embb_var: List[float] = []
    puncture_var: List[float] = []
    margin_var: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED_BASE + ep)
        done = False
        step_idx = 0

        total_violations = 0
        total_outages = 0
        embb_throughput_series: List[float] = []
        puncture_var_series: List[float] = []
        margin_var_series: List[float] = []

        while not done:
            action = int(policy_fn(obs, step_idx, env))
            obs, _reward, terminated, truncated, info = env.step(action)
            total_violations += int(info.get("urllc_latency_violations", 0))
            outages_this_step = int(info.get("embb_outages_this_step", 0))
            total_outages += outages_this_step
            embb_throughput_series.append((env.F - outages_this_step) / env.F)
            puncture_var_series.append(float(info.get("puncture_count_variance", 0.0)))
            margin_var_series.append(float(info.get("remaining_margin_variance", 0.0)))
            done = terminated or truncated
            step_idx += 1

        common_scores.append(-1.0 * total_violations - 0.5 * total_outages)
        violations.append(total_violations)
        outages.append(total_outages)
        embb_var.append(float(np.var(embb_throughput_series)))
        puncture_var.append(float(np.mean(puncture_var_series)))
        margin_var.append(float(np.mean(margin_var_series)))

    env.close()

    return {
        "common_score": np.array(common_scores, dtype=np.float64),
        "total_violations": np.array(violations, dtype=np.float64),
        "total_outages": np.array(outages, dtype=np.float64),
        "per_episode_embb_var": np.array(embb_var, dtype=np.float64),
        "per_episode_puncture_var": np.array(puncture_var, dtype=np.float64),
        "per_episode_margin_var": np.array(margin_var, dtype=np.float64),
    }


def evaluate_ppo(model_path: str, n_episodes: int = EPISODES) -> Dict[str, np.ndarray]:
    """Evaluate a PPO checkpoint with rich observation and common score."""
    model = PPO.load(model_path)
    return evaluate_policy(
        lambda obs, _t, _env: int(model.predict(obs, deterministic=True)[0]),
        n_episodes=n_episodes,
    )


def load_eval_curve(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load EvalCallback npz and return timesteps, mean and std reward curves."""
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]
    mean = results.mean(axis=1)
    std = results.std(axis=1)
    return timesteps, mean, std


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def format_mean_std_sci(mean: float, std: float) -> str:
    return f"{mean:.2e} $\\pm$ {std:.2e}"


def add_best_marker(value: float, best: float, text: str) -> str:
    return f"\\textbf{{{text}}}" if np.isclose(value, best) else text


def main() -> None:
    apply_style()

    metrics = {
        "fixed_static": evaluate_policy(lambda _obs, _t, _env: 0, n_episodes=EPISODES),
        "fixed_rr": evaluate_policy(lambda _obs, t, env: t % env.F, n_episodes=EPISODES),
        "PPO": evaluate_ppo("./models/baseline_ppo/final_model", n_episodes=EPISODES),
        "VarPPO": evaluate_ppo("./models/var_ppo/final_model", n_episodes=EPISODES),
    }

    baseline_ts, baseline_mean, baseline_std = load_eval_curve("./logs/baseline_ppo/evaluations.npz")
    var_ts, var_mean, var_std = load_eval_curve("./logs/var_ppo/evaluations.npz")
    baseline_mean_smooth = pd.Series(baseline_mean).rolling(window=5, min_periods=1, center=True).mean().values
    var_mean_smooth = pd.Series(var_mean).rolling(window=5, min_periods=1, center=True).mean().values

    os.makedirs("./figures", exist_ok=True)

    fontsize = 12
    linewidth = 2

    plt.figure(figsize=(8, 5))
    plt.plot(baseline_ts, baseline_mean_smooth, color="steelblue", linewidth=linewidth, label="PPO")
    plt.fill_between(
        baseline_ts,
        baseline_mean - baseline_std,
        baseline_mean + baseline_std,
        color="steelblue",
        alpha=0.2,
    )
    plt.plot(var_ts, var_mean_smooth, color="darkorange", linewidth=linewidth, label="VarPPO")
    plt.fill_between(var_ts, var_mean - var_std, var_mean + var_std, color="darkorange", alpha=0.2)
    plt.xlabel("Training Timesteps", fontsize=fontsize)
    plt.ylabel("Mean Episode Reward", fontsize=fontsize)
    plt.title("Training Convergence Comparison", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("./figures/training_curves.png", dpi=200)
    plt.savefig("./figures/training_curves.pdf")
    plt.close()

    agents = ["fixed_static", "fixed_rr", "PPO", "VarPPO"]
    colors = {
        "fixed_static": "dimgray",
        "fixed_rr": "gray",
        "PPO": "steelblue",
        "VarPPO": "darkorange",
    }

    means_per_agent = {
        agent: [
            np.mean(metrics[agent]["common_score"]),
            np.mean(metrics[agent]["total_violations"] / EPISODE_LENGTH),
            np.mean(metrics[agent]["total_outages"] / EPISODE_LENGTH),
            np.mean(metrics[agent]["per_episode_embb_var"]),
            np.mean(metrics[agent]["per_episode_puncture_var"]),
            np.mean(metrics[agent]["per_episode_margin_var"]),
        ]
        for agent in agents
    }
    std_per_agent = {
        agent: [
            np.std(metrics[agent]["common_score"]),
            np.std(metrics[agent]["total_violations"] / EPISODE_LENGTH),
            np.std(metrics[agent]["total_outages"] / EPISODE_LENGTH),
            np.std(metrics[agent]["per_episode_embb_var"]),
            np.std(metrics[agent]["per_episode_puncture_var"]),
            np.std(metrics[agent]["per_episode_margin_var"]),
        ]
        for agent in agents
    }

    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    x = np.arange(len(agents))
    subplot_titles = [
        "Common Score",
        "URLLC Violation Rate",
        "eMBB Outage Rate",
        "eMBB Throughput Var",
        "Puncture Variance",
        "Margin Variance",
    ]

    for metric_idx, ax in enumerate(axes):
        vals = [means_per_agent[agent][metric_idx] for agent in agents]
        errs = [std_per_agent[agent][metric_idx] for agent in agents]
        ax.bar(x, vals, yerr=errs, capsize=4, color=[colors[agent] for agent in agents])
        ax.set_xticks(x)
        ax.set_xticklabels(agents, fontsize=9, rotation=25, ha="right")
        ax.set_title(subplot_titles[metric_idx], fontsize=fontsize)
        ax.set_ylabel("Value", fontsize=fontsize)

    fig.suptitle("Per-Agent Performance Metrics on Common Score", fontsize=fontsize)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./figures/performance_bar.png", dpi=200)
    plt.savefig("./figures/performance_bar.pdf")
    plt.close()

    variance_data = [metrics[agent]["per_episode_puncture_var"] for agent in agents]
    plt.figure(figsize=(9, 5))
    plt.boxplot(variance_data, labels=agents)
    rng = np.random.default_rng(0)
    for i, values in enumerate(variance_data, start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        plt.scatter(np.full_like(values, i, dtype=np.float64) + jitter, values, alpha=0.3, s=4, c="black")

    plt.ylabel("Per-Episode Puncture-Count Variance", fontsize=fontsize)
    plt.title("Distribution of Load-Balancing Variance per Episode", fontsize=fontsize)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("./figures/variance_dist.png", dpi=200)
    plt.savefig("./figures/variance_dist.pdf")
    plt.close()

    summary = {
        agent: {
            "score_mean": np.mean(metrics[agent]["common_score"]),
            "score_std": np.std(metrics[agent]["common_score"]),
            "viol_mean": np.mean(metrics[agent]["total_violations"]),
            "viol_std": np.std(metrics[agent]["total_violations"]),
            "out_mean": np.mean(metrics[agent]["total_outages"]),
            "out_std": np.std(metrics[agent]["total_outages"]),
            "embb_var_mean": np.mean(metrics[agent]["per_episode_embb_var"]),
            "embb_var_std": np.std(metrics[agent]["per_episode_embb_var"]),
            "puncture_var_mean": np.mean(metrics[agent]["per_episode_puncture_var"]),
            "puncture_var_std": np.std(metrics[agent]["per_episode_puncture_var"]),
            "margin_var_mean": np.mean(metrics[agent]["per_episode_margin_var"]),
            "margin_var_std": np.std(metrics[agent]["per_episode_margin_var"]),
        }
        for agent in agents
    }

    score_best = max(stats["score_mean"] for stats in summary.values())
    viol_best = min(stats["viol_mean"] for stats in summary.values())
    out_best = min(stats["out_mean"] for stats in summary.values())
    embb_var_best = min(stats["embb_var_mean"] for stats in summary.values())
    puncture_var_best = min(stats["puncture_var_mean"] for stats in summary.values())
    margin_var_best = min(stats["margin_var_mean"] for stats in summary.values())

    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print(
        "Agent & Common Score & URLLC Viol./ep & eMBB Outages/ep "
        "& eMBB Var & Puncture Var & Margin Var \\\\"
    )
    print("\\hline")

    for agent in agents:
        row = summary[agent]
        score_str = add_best_marker(
            float(row["score_mean"]),
            score_best,
            format_mean_std(float(row["score_mean"]), float(row["score_std"])),
        )
        viol_str = add_best_marker(
            float(row["viol_mean"]),
            viol_best,
            format_mean_std(float(row["viol_mean"]), float(row["viol_std"])),
        )
        out_str = add_best_marker(
            float(row["out_mean"]),
            out_best,
            format_mean_std(float(row["out_mean"]), float(row["out_std"])),
        )
        embb_var_str = add_best_marker(
            float(row["embb_var_mean"]),
            embb_var_best,
            format_mean_std_sci(float(row["embb_var_mean"]), float(row["embb_var_std"])),
        )
        puncture_var_str = add_best_marker(
            float(row["puncture_var_mean"]),
            puncture_var_best,
            format_mean_std_sci(float(row["puncture_var_mean"]), float(row["puncture_var_std"])),
        )
        margin_var_str = add_best_marker(
            float(row["margin_var_mean"]),
            margin_var_best,
            format_mean_std_sci(float(row["margin_var_mean"]), float(row["margin_var_std"])),
        )

        print(
            f"{agent} & {score_str} & {viol_str} & {out_str} & "
            f"{embb_var_str} & {puncture_var_str} & {margin_var_str} \\\\"
        )

    print("\\hline")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
