"""Evaluate trained agents on baseline environment and generate comparison plots."""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ran_env import RANSlicingEnv

EPISODES = 100
EPISODE_LENGTH = 140
SEED_BASE = 42


def apply_style() -> None:
    """Apply plotting style with fallback requested by user."""
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        if "seaborn-paper" in plt.style.available:
            plt.style.use("seaborn-paper")


def evaluate_ppo(model_path: str, n_episodes: int = EPISODES) -> Dict[str, np.ndarray]:
    """Evaluate a PPO checkpoint on baseline RANSlicingEnv."""
    model = PPO.load(model_path)
    env = RANSlicingEnv()

    rewards: List[float] = []
    violations: List[int] = []
    outages: List[int] = []
    embb_var: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED_BASE + ep)
        done = False

        total_reward = 0.0
        total_violations = 0
        total_outages = 0
        embb_throughput_series: List[float] = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            total_violations += int(info.get("urllc_latency_violations", 0))
            outages_this_step = int(info.get("embb_outages_this_step", 0))
            total_outages += outages_this_step
            embb_throughput_series.append((12 - outages_this_step) / 12)
            done = terminated or truncated

        rewards.append(total_reward)
        violations.append(total_violations)
        outages.append(total_outages)
        embb_var.append(float(np.var(embb_throughput_series)))

    env.close()

    return {
        "total_reward": np.array(rewards, dtype=np.float64),
        "total_violations": np.array(violations, dtype=np.float64),
        "total_outages": np.array(outages, dtype=np.float64),
        "per_episode_embb_var": np.array(embb_var, dtype=np.float64),
    }


def evaluate_fixed_rr(n_episodes: int = EPISODES) -> Dict[str, np.ndarray]:
    """Evaluate fixed round-robin policy action=t%12 on baseline env."""
    env = RANSlicingEnv()

    rewards: List[float] = []
    violations: List[int] = []
    outages: List[int] = []
    embb_var: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED_BASE + ep)
        del obs
        done = False
        t = 0

        total_reward = 0.0
        total_violations = 0
        total_outages = 0
        embb_throughput_series: List[float] = []

        while not done:
            action = t % 12
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            total_violations += int(info.get("urllc_latency_violations", 0))
            outages_this_step = int(info.get("embb_outages_this_step", 0))
            total_outages += outages_this_step
            embb_throughput_series.append((12 - outages_this_step) / 12)
            done = terminated or truncated
            t += 1

        rewards.append(total_reward)
        violations.append(total_violations)
        outages.append(total_outages)
        embb_var.append(float(np.var(embb_throughput_series)))

    env.close()

    return {
        "total_reward": np.array(rewards, dtype=np.float64),
        "total_violations": np.array(violations, dtype=np.float64),
        "total_outages": np.array(outages, dtype=np.float64),
        "per_episode_embb_var": np.array(embb_var, dtype=np.float64),
    }


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


def main() -> None:
    apply_style()

    baseline_metrics = evaluate_ppo("./models/baseline_ppo/final_model", n_episodes=EPISODES)
    var_metrics = evaluate_ppo("./models/var_ppo/final_model", n_episodes=EPISODES)
    rr_metrics = evaluate_fixed_rr(n_episodes=EPISODES)

    baseline_ts, baseline_mean, baseline_std = load_eval_curve("./logs/baseline_ppo/evaluations.npz")
    var_ts, var_mean, var_std = load_eval_curve("./logs/var_ppo/evaluations.npz")
    baseline_mean_smooth = pd.Series(baseline_mean).rolling(window=5, min_periods=1, center=True).mean().values
    var_mean_smooth = pd.Series(var_mean).rolling(window=5, min_periods=1, center=True).mean().values

    os.makedirs("./figures", exist_ok=True)

    fontsize = 12
    linewidth = 2

    # Figure 1: training curves
    plt.figure(figsize=(8, 5))
    plt.plot(
        baseline_ts,
        baseline_mean_smooth,
        color="blue",
        linewidth=linewidth,
        label="PPO (Baseline Reward)",
    )
    plt.fill_between(baseline_ts, baseline_mean - baseline_std, baseline_mean + baseline_std, color="blue", alpha=0.2)

    plt.plot(
        var_ts,
        var_mean_smooth,
        color="orange",
        linewidth=linewidth,
        label="PPO (Variance-Penalized)",
    )
    plt.fill_between(var_ts, var_mean - var_std, var_mean + var_std, color="orange", alpha=0.2)

    plt.xlabel("Training Timesteps", fontsize=fontsize)
    plt.ylabel("Mean Episode Reward", fontsize=fontsize)
    plt.title("Training Convergence Comparison", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("./figures/training_curves.png", dpi=200)
    plt.savefig("./figures/training_curves.pdf")
    plt.close()

    # Figure 2: grouped bars for performance metrics
    agents = ["fixed_rr", "baseline_ppo", "var_ppo"]
    metrics = {
        "fixed_rr": rr_metrics,
        "baseline_ppo": baseline_metrics,
        "var_ppo": var_metrics,
    }
    colors = {"fixed_rr": "gray", "baseline_ppo": "steelblue", "var_ppo": "darkorange"}

    means_per_agent = {
        "fixed_rr": [
            np.mean(rr_metrics["total_violations"] / EPISODE_LENGTH),
            np.mean(rr_metrics["total_outages"] / EPISODE_LENGTH),
            np.mean(rr_metrics["per_episode_embb_var"]),
        ],
        "baseline_ppo": [
            np.mean(baseline_metrics["total_violations"] / EPISODE_LENGTH),
            np.mean(baseline_metrics["total_outages"] / EPISODE_LENGTH),
            np.mean(baseline_metrics["per_episode_embb_var"]),
        ],
        "var_ppo": [
            np.mean(var_metrics["total_violations"] / EPISODE_LENGTH),
            np.mean(var_metrics["total_outages"] / EPISODE_LENGTH),
            np.mean(var_metrics["per_episode_embb_var"]),
        ],
    }
    std_per_agent = {
        "fixed_rr": [
            np.std(rr_metrics["total_violations"] / EPISODE_LENGTH),
            np.std(rr_metrics["total_outages"] / EPISODE_LENGTH),
            np.std(rr_metrics["per_episode_embb_var"]),
        ],
        "baseline_ppo": [
            np.std(baseline_metrics["total_violations"] / EPISODE_LENGTH),
            np.std(baseline_metrics["total_outages"] / EPISODE_LENGTH),
            np.std(baseline_metrics["per_episode_embb_var"]),
        ],
        "var_ppo": [
            np.std(var_metrics["total_violations"] / EPISODE_LENGTH),
            np.std(var_metrics["total_outages"] / EPISODE_LENGTH),
            np.std(var_metrics["per_episode_embb_var"]),
        ],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(agents))
    subplot_titles = [
        "URLLC Violation Rate",
        "eMBB Outage Rate",
        "eMBB Throughput Variance",
    ]

    for metric_idx, ax in enumerate(axes):
        vals = [means_per_agent[agent][metric_idx] for agent in agents]
        errs = [std_per_agent[agent][metric_idx] for agent in agents]
        ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=4,
            color=[colors[agent] for agent in agents],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(agents, fontsize=fontsize)
        ax.set_title(subplot_titles[metric_idx], fontsize=fontsize)
        ax.set_ylabel("Value", fontsize=fontsize)

    fig.suptitle("Per-Agent Performance Metrics", fontsize=fontsize)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./figures/performance_bar.png", dpi=200)
    plt.savefig("./figures/performance_bar.pdf")
    plt.close()

    # Figure 3: variance distribution
    variance_data = [
        rr_metrics["per_episode_embb_var"],
        baseline_metrics["per_episode_embb_var"],
        var_metrics["per_episode_embb_var"],
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(variance_data, labels=agents)
    rng = np.random.default_rng(0)
    for i, values in enumerate(variance_data, start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        plt.scatter(np.full_like(values, i, dtype=np.float64) + jitter, values, alpha=0.3, s=4, c="black")

    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))
    plt.ylabel("Per-Episode eMBB Throughput Variance", fontsize=fontsize)
    plt.title("Distribution of eMBB Throughput Variance per Episode", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig("./figures/variance_dist.png", dpi=200)
    plt.savefig("./figures/variance_dist.pdf")
    plt.close()

    summary = {
        "fixed_rr": {
            "reward_mean": np.mean(rr_metrics["total_reward"]),
            "reward_std": np.std(rr_metrics["total_reward"]),
            "viol_mean": np.mean(rr_metrics["total_violations"]),
            "viol_std": np.std(rr_metrics["total_violations"]),
            "out_mean": np.mean(rr_metrics["total_outages"]),
            "out_std": np.std(rr_metrics["total_outages"]),
            "var_mean": np.mean(rr_metrics["per_episode_embb_var"]),
            "var_std": np.std(rr_metrics["per_episode_embb_var"]),
        },
        "baseline_ppo": {
            "reward_mean": np.mean(baseline_metrics["total_reward"]),
            "reward_std": np.std(baseline_metrics["total_reward"]),
            "viol_mean": np.mean(baseline_metrics["total_violations"]),
            "viol_std": np.std(baseline_metrics["total_violations"]),
            "out_mean": np.mean(baseline_metrics["total_outages"]),
            "out_std": np.std(baseline_metrics["total_outages"]),
            "var_mean": np.mean(baseline_metrics["per_episode_embb_var"]),
            "var_std": np.std(baseline_metrics["per_episode_embb_var"]),
        },
        "var_ppo": {
            "reward_mean": np.mean(var_metrics["total_reward"]),
            "reward_std": np.std(var_metrics["total_reward"]),
            "viol_mean": np.mean(var_metrics["total_violations"]),
            "viol_std": np.std(var_metrics["total_violations"]),
            "out_mean": np.mean(var_metrics["total_outages"]),
            "out_std": np.std(var_metrics["total_outages"]),
            "var_mean": np.mean(var_metrics["per_episode_embb_var"]),
            "var_std": np.std(var_metrics["per_episode_embb_var"]),
        },
    }

    reward_best = max(stats["reward_mean"] for stats in summary.values())
    viol_best = min(stats["viol_mean"] for stats in summary.values())
    out_best = min(stats["out_mean"] for stats in summary.values())
    var_best = min(stats["var_mean"] for stats in summary.values())

    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Agent & Mean Reward & URLLC Viol./ep & eMBB Outages/ep & eMBB Var " + "\\")
    print("\\hline")

    for agent in agents:
        row = summary[agent]
        reward_str = format_mean_std(float(row["reward_mean"]), float(row["reward_std"]))
        viol_str = format_mean_std(float(row["viol_mean"]), float(row["viol_std"]))
        out_str = format_mean_std(float(row["out_mean"]), float(row["out_std"]))
        var_str = format_mean_std_sci(float(row["var_mean"]), float(row["var_std"]))

        if np.isclose(row["reward_mean"], reward_best):
            reward_str = f"\\textbf{{{reward_str}}}"
        if np.isclose(row["viol_mean"], viol_best):
            viol_str = f"\\textbf{{{viol_str}}}"
        if np.isclose(row["out_mean"], out_best):
            out_str = f"\\textbf{{{out_str}}}"
        if np.isclose(row["var_mean"], var_best):
            var_str = f"\\textbf{{{var_str}}}"

        print(f"{agent} & {reward_str} & {viol_str} & {out_str} & {var_str} " + "\\")

    print("\\hline")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
