"""Generalization test across out-of-distribution URLLC arrival rates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ran_env import RANSlicingEnv
from ran_env_var import RANSlicingEnvVar

DEFAULT_EPISODES = 50
EPISODE_LENGTH = 140
SEED_BASE = 42
LAMBDA_LIST = [0.2, 0.5, 0.8, 1.0, 1.5]
SLA_THRESHOLD = 0.05

AgentStats = Dict[str, List[float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=SEED_BASE)
    parser.add_argument("--run-name", type=str, default="robustness")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--mode", choices=["baseline", "research"], default="baseline")
    parser.add_argument("--traffic-model", type=str, default="poisson")
    parser.add_argument("--reward-profile", choices=["default", "urllc_heavy"], default="default")
    return parser.parse_args()


def create_run_dir(run_name: str, output_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "robustness").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_model(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained model: {model_path}. Run `python train.py` first."
        )


def evaluate_agent(
    policy_fn: Callable[[np.ndarray, int], int],
    env,
    n_episodes: int,
    seed_base: int,
) -> AgentStats:
    violation_rates: List[float] = []
    outage_rates: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        done = False
        step_idx = 0
        total_urllc_violations = 0
        total_embb_outages = 0

        while not done:
            action = int(policy_fn(obs, step_idx))
            obs, _, terminated, truncated, info = env.step(action)
            total_urllc_violations += int(info.get("urllc_latency_violations", 0))
            total_embb_outages += int(info.get("embb_outages_this_step", 0))
            done = terminated or truncated
            step_idx += 1

        violation_rates.append(total_urllc_violations / EPISODE_LENGTH)
        outage_rates.append(total_embb_outages / EPISODE_LENGTH)

    return {"violation_rates": violation_rates, "outage_rates": outage_rates}


def save_figure(fig: plt.Figure, name: str, run_dir: Path) -> None:
    figures_dir = Path("./figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=200)
    fig.savefig(figures_dir / f"{name}.pdf")
    fig.savefig(run_dir / "figures" / f"{name}.png", dpi=200)
    fig.savefig(run_dir / "figures" / f"{name}.pdf")


def main() -> None:
    args = parse_args()
    if args.mode == "baseline":
        if args.traffic_model != "poisson" or args.reward_profile != "default":
            raise ValueError(
                "Baseline mode must keep --traffic-model=poisson and --reward-profile=default "
                "to preserve reproducibility."
            )
    run_dir = create_run_dir(args.run_name, args.output_dir)

    baseline_model_path = Path("./models/baseline_ppo/final_model.zip")
    var_model_path = Path("./models/var_ppo/final_model.zip")
    ensure_model(baseline_model_path)
    ensure_model(var_model_path)

    baseline_model = PPO.load(str(baseline_model_path))
    var_model = PPO.load(str(var_model_path))

    stats = {
        "fixed_rr": {"viol_mean": [], "viol_std": [], "out_mean": [], "out_std": []},
        "baseline_ppo": {"viol_mean": [], "viol_std": [], "out_mean": [], "out_std": []},
        "var_ppo": {"viol_mean": [], "viol_std": [], "out_mean": [], "out_std": []},
    }
    rows: List[Dict[str, float | int | str]] = []

    for lam in LAMBDA_LIST:
        rr_env = RANSlicingEnv(arrival_rate=lam, reward_profile=args.reward_profile)
        baseline_env = RANSlicingEnv(arrival_rate=lam, reward_profile=args.reward_profile)
        var_env = RANSlicingEnvVar(arrival_rate=lam, alpha=0.3, reward_profile=args.reward_profile)

        rr_metrics = evaluate_agent(lambda _obs, t: t % rr_env.F, rr_env, args.episodes, args.seed)
        baseline_metrics = evaluate_agent(
            lambda obs, _t: int(baseline_model.predict(obs, deterministic=True)[0]),
            baseline_env,
            args.episodes,
            args.seed,
        )
        var_metrics = evaluate_agent(
            lambda obs, _t: int(var_model.predict(obs, deterministic=True)[0]),
            var_env,
            args.episodes,
            args.seed,
        )

        rr_env.close()
        baseline_env.close()
        var_env.close()

        for agent_name, metrics in (("fixed_rr", rr_metrics), ("baseline_ppo", baseline_metrics), ("var_ppo", var_metrics)):
            viol = np.array(metrics["violation_rates"], dtype=np.float64)
            out = np.array(metrics["outage_rates"], dtype=np.float64)
            stats[agent_name]["viol_mean"].append(float(np.mean(viol)))
            stats[agent_name]["viol_std"].append(float(np.std(viol)))
            stats[agent_name]["out_mean"].append(float(np.mean(out)))
            stats[agent_name]["out_std"].append(float(np.std(out)))

            for ep in range(args.episodes):
                rows.append(
                    {
                        "agent_name": agent_name,
                        "lambda": lam,
                        "episode": ep,
                        "urllc_violation_rate": float(viol[ep]),
                        "embb_outage_rate": float(out[ep]),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "robustness" / "robustness_metrics.csv", index=False)

    fontsize = 12
    linewidth = 2
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plot_styles = {
        "fixed_rr": {"color": "gray", "linestyle": "--", "label": "fixed_rr"},
        "baseline_ppo": {"color": "blue", "linestyle": "-", "label": "baseline_ppo"},
        "var_ppo": {"color": "orange", "linestyle": "-", "label": "var_ppo"},
    }

    for agent_name, style in plot_styles.items():
        viol_mean = np.array(stats[agent_name]["viol_mean"])
        viol_std = np.array(stats[agent_name]["viol_std"])
        axes[0].plot(LAMBDA_LIST, viol_mean, color=style["color"], linestyle=style["linestyle"], linewidth=linewidth, label=style["label"])
        axes[0].fill_between(LAMBDA_LIST, viol_mean - viol_std, viol_mean + viol_std, color=style["color"], alpha=0.15)

        out_mean = np.array(stats[agent_name]["out_mean"])
        out_std = np.array(stats[agent_name]["out_std"])
        axes[1].plot(LAMBDA_LIST, out_mean, color=style["color"], linestyle=style["linestyle"], linewidth=linewidth, label=style["label"])
        axes[1].fill_between(LAMBDA_LIST, out_mean - out_std, out_mean + out_std, color=style["color"], alpha=0.15)

    axes[0].axhline(y=SLA_THRESHOLD, color="red", linestyle="--", linewidth=linewidth, label="SLA Threshold (5%)")
    axes[0].set_ylabel("URLLC Violation Rate", fontsize=fontsize)
    axes[1].set_ylabel("eMBB Outage Rate", fontsize=fontsize)

    for ax in axes:
        ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=linewidth, label="Training λ")
        ax.set_xlabel("URLLC Arrival Rate λ (packets/minislot)", fontsize=fontsize)
        ax.legend(fontsize=fontsize)

    fig.suptitle("Robustness to Out-of-Distribution URLLC Arrival Rates", fontsize=fontsize)
    fig.tight_layout()
    save_figure(fig, "robustness", run_dir)
    plt.close(fig)

    summary: Dict[str, Dict[str, Dict[str, float | bool]]] = {}
    for agent_name in ["fixed_rr", "baseline_ppo", "var_ppo"]:
        summary[agent_name] = {}
        for lam in LAMBDA_LIST:
            sub = df[(df["agent_name"] == agent_name) & (df["lambda"] == lam)]
            mean_viol = float(sub["urllc_violation_rate"].mean())
            summary[agent_name][str(lam)] = {
                "violation_rate_mean": mean_viol,
                "violation_rate_std": float(sub["urllc_violation_rate"].std(ddof=0)),
                "sla_threshold": SLA_THRESHOLD,
                "sla_exceeded": mean_viol > SLA_THRESHOLD,
            }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "episodes": args.episodes,
                "seed": args.seed,
                "lambdas": LAMBDA_LIST,
                "sla_threshold": SLA_THRESHOLD,
                "mode": args.mode,
                "traffic_model": args.traffic_model,
                "reward_profile": args.reward_profile,
                "baseline_policy": (
                    "Default baseline must remain unchanged and reproducible: "
                    "Poisson arrival + default reward. "
                    "Research mode may add explicit, comparable, reversible variants only."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# AI Robustness Report",
        "",
        f"- Episodes per lambda: {args.episodes}",
        f"- SLA threshold on URLLC violations: {SLA_THRESHOLD}",
        "",
        "## Observations",
    ]
    for agent in ["fixed_rr", "baseline_ppo", "var_ppo"]:
        sla_pass = [lam for lam in LAMBDA_LIST if not summary[agent][str(lam)]["sla_exceeded"]]
        report_lines.append(f"- {agent}: meets SLA at lambdas {sla_pass}.")
    report_lines.append("")
    report_lines.append("This report is descriptive and does not introduce new baselines or claims.")
    (run_dir / "ai_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
