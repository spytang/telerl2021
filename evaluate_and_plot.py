"""Evaluate trained agents on baseline environment and generate comparison plots."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ran_env import RANSlicingEnv

EPISODE_LENGTH = 140
SEED_BASE = 42
DEFAULT_EPISODES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED_BASE)
    parser.add_argument("--run-name", type=str, default="evaluate")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--mode", choices=["baseline", "research"], default="baseline")
    parser.add_argument("--traffic-model", type=str, default="poisson")
    parser.add_argument("--reward-profile", choices=["default", "urllc_heavy", "risk_aware"], default="default")
    return parser.parse_args()


def create_run_dir(run_name: str, output_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        if "seaborn-paper" in plt.style.available:
            plt.style.use("seaborn-paper")


def ensure_model(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained model: {model_path}. Run `python train.py` first."
        )


def evaluate_ppo(
    model_path: Path,
    n_episodes: int,
    seed_base: int,
    reward_profile: str,
) -> Dict[str, np.ndarray]:
    ensure_model(model_path)
    model = PPO.load(str(model_path))
    env = RANSlicingEnv(
        reward_profile=reward_profile,
        include_channel_state=reward_profile == "risk_aware",
    )

    rewards: List[float] = []
    violations: List[int] = []
    outages: List[int] = []
    embb_var: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
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


def evaluate_fixed_rr(n_episodes: int, seed_base: int, reward_profile: str) -> Dict[str, np.ndarray]:
    env = RANSlicingEnv(
        reward_profile=reward_profile,
        include_channel_state=reward_profile == "risk_aware",
    )
    rewards: List[float] = []
    violations: List[int] = []
    outages: List[int] = []
    embb_var: List[float] = []

    for ep in range(n_episodes):
        _, _ = env.reset(seed=seed_base + ep)
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


def load_eval_curve(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing eval curve file: {npz_path}. "
            "This commonly happens if training timesteps are lower than eval_freq "
            "(default eval_freq=5000 in train.py)."
        )
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]
    return timesteps, results.mean(axis=1), results.std(axis=1)


def save_figure(fig: plt.Figure, name: str, run_dir: Path) -> None:
    figures_dir = Path("./figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=200)
    fig.savefig(figures_dir / f"{name}.pdf")
    fig.savefig(run_dir / "figures" / f"{name}.png", dpi=200)
    fig.savefig(run_dir / "figures" / f"{name}.pdf")


def format_mean_std(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def format_mean_std_sci(mean: float, std: float) -> str:
    return f"{mean:.2e} $\\pm$ {std:.2e}"


def main() -> None:
    args = parse_args()
    if args.mode == "baseline":
        if args.traffic_model != "poisson" or args.reward_profile != "default":
            raise ValueError(
                "Baseline mode must keep --traffic-model=poisson and --reward-profile=default "
                "to preserve reproducibility."
            )
    run_dir = create_run_dir(args.run_name, args.output_dir)
    apply_style()

    baseline_metrics = evaluate_ppo(
        Path("./models/baseline_ppo/final_model.zip"),
        n_episodes=args.episodes,
        seed_base=args.seed,
        reward_profile=args.reward_profile,
    )
    var_metrics = evaluate_ppo(
        Path("./models/var_ppo/final_model.zip"),
        n_episodes=args.episodes,
        seed_base=args.seed,
        reward_profile=args.reward_profile,
    )
    rr_metrics = evaluate_fixed_rr(n_episodes=args.episodes, seed_base=args.seed, reward_profile=args.reward_profile)

    curve_warning: str | None = None
    try:
        baseline_ts, baseline_mean, baseline_std = load_eval_curve(Path("./logs/baseline_ppo/evaluations.npz"))
        var_ts, var_mean, var_std = load_eval_curve(Path("./logs/var_ppo/evaluations.npz"))
        baseline_mean_smooth = pd.Series(baseline_mean).rolling(window=5, min_periods=1, center=True).mean().values
        var_mean_smooth = pd.Series(var_mean).rolling(window=5, min_periods=1, center=True).mean().values
    except FileNotFoundError as exc:
        curve_warning = str(exc)
        baseline_ts = baseline_mean = baseline_std = np.array([], dtype=np.float64)
        var_ts = var_mean = var_std = np.array([], dtype=np.float64)
        baseline_mean_smooth = var_mean_smooth = np.array([], dtype=np.float64)

    fontsize = 12
    linewidth = 2

    if baseline_ts.size > 0 and var_ts.size > 0:
        fig1 = plt.figure(figsize=(8, 5))
        plt.plot(baseline_ts, baseline_mean_smooth, color="blue", linewidth=linewidth, label="PPO (Baseline Reward)")
        plt.fill_between(baseline_ts, baseline_mean - baseline_std, baseline_mean + baseline_std, color="blue", alpha=0.2)
        plt.plot(var_ts, var_mean_smooth, color="orange", linewidth=linewidth, label="PPO (Variance-Penalized)")
        plt.fill_between(var_ts, var_mean - var_std, var_mean + var_std, color="orange", alpha=0.2)
        plt.xlabel("Training Timesteps", fontsize=fontsize)
        plt.ylabel("Mean Episode Reward", fontsize=fontsize)
        plt.title("Training Convergence Comparison", fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        save_figure(fig1, "training_curves", run_dir)
        plt.close(fig1)
    elif curve_warning:
        print(f"[WARN] {curve_warning}")

    agents = ["fixed_rr", "baseline_ppo", "var_ppo"]
    agent_metrics = {"fixed_rr": rr_metrics, "baseline_ppo": baseline_metrics, "var_ppo": var_metrics}
    colors = {"fixed_rr": "gray", "baseline_ppo": "steelblue", "var_ppo": "darkorange"}

    means_per_agent = {
        agent: [
            np.mean(m["total_violations"] / EPISODE_LENGTH),
            np.mean(m["total_outages"] / EPISODE_LENGTH),
            np.mean(m["per_episode_embb_var"]),
        ]
        for agent, m in agent_metrics.items()
    }
    std_per_agent = {
        agent: [
            np.std(m["total_violations"] / EPISODE_LENGTH),
            np.std(m["total_outages"] / EPISODE_LENGTH),
            np.std(m["per_episode_embb_var"]),
        ]
        for agent, m in agent_metrics.items()
    }

    fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(agents))
    subplot_titles = ["URLLC Violation Rate", "eMBB Outage Rate", "eMBB Throughput Variance"]
    for metric_idx, ax in enumerate(axes):
        vals = [means_per_agent[agent][metric_idx] for agent in agents]
        errs = [std_per_agent[agent][metric_idx] for agent in agents]
        ax.bar(x, vals, yerr=errs, capsize=4, color=[colors[agent] for agent in agents])
        ax.set_xticks(x)
        ax.set_xticklabels(agents, fontsize=fontsize)
        ax.set_title(subplot_titles[metric_idx], fontsize=fontsize)
        ax.set_ylabel("Value", fontsize=fontsize)
    fig2.suptitle("Per-Agent Performance Metrics", fontsize=fontsize)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig2, "performance_bar", run_dir)
    plt.close(fig2)

    variance_data = [rr_metrics["per_episode_embb_var"], baseline_metrics["per_episode_embb_var"], var_metrics["per_episode_embb_var"]]
    fig3 = plt.figure(figsize=(8, 5))
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
    save_figure(fig3, "variance_dist", run_dir)
    plt.close(fig3)

    rows: List[Dict[str, float | int | str]] = []
    for agent, metrics in agent_metrics.items():
        for episode in range(args.episodes):
            rows.append(
                {
                    "agent_name": agent,
                    "episode": episode,
                    "total_reward": float(metrics["total_reward"][episode]),
                    "urllc_violation_rate": float(metrics["total_violations"][episode] / EPISODE_LENGTH),
                    "embb_outage_rate": float(metrics["total_outages"][episode] / EPISODE_LENGTH),
                    "embb_throughput_variance": float(metrics["per_episode_embb_var"][episode]),
                }
            )
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(run_dir / "evaluation" / "evaluation_metrics.csv", index=False)

    summary: Dict[str, Dict[str, float]] = {}
    for agent in agents:
        sub = eval_df[eval_df["agent_name"] == agent]
        summary[agent] = {
            "reward_mean": float(sub["total_reward"].mean()),
            "reward_std": float(sub["total_reward"].std(ddof=0)),
            "urllc_violation_rate_mean": float(sub["urllc_violation_rate"].mean()),
            "urllc_violation_rate_std": float(sub["urllc_violation_rate"].std(ddof=0)),
            "embb_outage_rate_mean": float(sub["embb_outage_rate"].mean()),
            "embb_outage_rate_std": float(sub["embb_outage_rate"].std(ddof=0)),
            "embb_throughput_variance_mean": float(sub["embb_throughput_variance"].mean()),
            "embb_throughput_variance_std": float(sub["embb_throughput_variance"].std(ddof=0)),
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "episodes": args.episodes,
                "seed": args.seed,
                "mode": args.mode,
                "traffic_model": args.traffic_model,
                "reward_profile": args.reward_profile,
                "baseline_policy": (
                    "Default baseline must remain unchanged and reproducible: "
                    "Poisson arrival + default reward. "
                    "Research mode may add explicit, comparable, reversible variants only."
                ),
                "model_paths": {
                    "baseline_ppo": "./models/baseline_ppo/final_model.zip",
                    "var_ppo": "./models/var_ppo/final_model.zip",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_reward_agent = max(summary, key=lambda k: summary[k]["reward_mean"])
    best_violation_agent = min(summary, key=lambda k: summary[k]["urllc_violation_rate_mean"])
    best_outage_agent = min(summary, key=lambda k: summary[k]["embb_outage_rate_mean"])
    best_var_agent = min(summary, key=lambda k: summary[k]["embb_throughput_variance_mean"])

    report = (
        "# AI Evaluation Report\n\n"
        f"- Episodes per agent: {args.episodes}\n"
        f"- Seed base: {args.seed}\n\n"
        "## Data-driven highlights\n"
        f"- Highest mean reward: **{best_reward_agent}** ({summary[best_reward_agent]['reward_mean']:.3f}).\n"
        f"- Lowest URLLC violation rate: **{best_violation_agent}** ({summary[best_violation_agent]['urllc_violation_rate_mean']:.4f}).\n"
        f"- Lowest eMBB outage rate: **{best_outage_agent}** ({summary[best_outage_agent]['embb_outage_rate_mean']:.4f}).\n"
        f"- Lowest throughput variance: **{best_var_agent}** ({summary[best_var_agent]['embb_throughput_variance_mean']:.2e}).\n\n"
        "Interpretation is descriptive only; no new algorithmic claims are introduced.\n"
    )
    if curve_warning:
        report += f"\nTraining-curve note: {curve_warning}\n"
    (run_dir / "ai_report.md").write_text(report, encoding="utf-8")

    reward_best = max(stats["reward_mean"] for stats in summary.values())
    viol_best = min(stats["urllc_violation_rate_mean"] for stats in summary.values())
    out_best = min(stats["embb_outage_rate_mean"] for stats in summary.values())
    var_best = min(stats["embb_throughput_variance_mean"] for stats in summary.values())

    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Agent & Mean Reward & URLLC Viol./ep & eMBB Outages/ep & eMBB Var " + "\\")
    print("\\hline")

    for agent in agents:
        row = summary[agent]
        reward_str = format_mean_std(float(row["reward_mean"]), float(row["reward_std"]))
        viol_str = format_mean_std(float(row["urllc_violation_rate_mean"]), float(row["urllc_violation_rate_std"]))
        out_str = format_mean_std(float(row["embb_outage_rate_mean"]), float(row["embb_outage_rate_std"]))
        var_str = format_mean_std_sci(float(row["embb_throughput_variance_mean"]), float(row["embb_throughput_variance_std"]))

        if np.isclose(row["reward_mean"], reward_best):
            reward_str = f"\\textbf{{{reward_str}}}"
        if np.isclose(row["urllc_violation_rate_mean"], viol_best):
            viol_str = f"\\textbf{{{viol_str}}}"
        if np.isclose(row["embb_outage_rate_mean"], out_best):
            out_str = f"\\textbf{{{out_str}}}"
        if np.isclose(row["embb_throughput_variance_mean"], var_best):
            var_str = f"\\textbf{{{var_str}}}"

        print(f"{agent} & {reward_str} & {viol_str} & {out_str} & {var_str} " + "\\")

    print("\\hline")
    print("\\end{tabular}")


if __name__ == "__main__":
    main()
