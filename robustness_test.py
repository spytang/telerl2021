"""Generalization test across out-of-distribution URLLC arrival rates."""

from __future__ import annotations

import os
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from ran_env import RANSlicingEnv
from ran_env_var import RANSlicingEnvVar

EPISODES = 50
EPISODE_LENGTH = 140
SEED_BASE = 42
LAMBDA_LIST = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]


AgentStats = Dict[str, List[float]]


def evaluate_agent(
    policy_fn: Callable[[np.ndarray, int], int],
    env,
    n_episodes: int = EPISODES,
) -> AgentStats:
    """Evaluate one policy on one environment and return per-episode rates."""
    violation_rates: List[float] = []
    outage_rates: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=SEED_BASE + ep)
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

    return {
        "violation_rates": violation_rates,
        "outage_rates": outage_rates,
    }


def main() -> None:
    baseline_model = PPO.load("./models/baseline_ppo/final_model")
    var_model = PPO.load("./models/var_ppo/final_model")

    stats = {
        "fixed_rr": {"viol_mean": [], "viol_std": [], "out_mean": [], "out_std": []},
        "baseline_ppo": {"viol_mean": [], "viol_std": [], "out_mean": [], "out_std": []},
        "var_ppo": {"viol_mean": [], "viol_std": [], "out_mean": [], "out_std": []},
    }

    for lam in LAMBDA_LIST:
        rr_env = RANSlicingEnv(arrival_rate=lam)
        baseline_env = RANSlicingEnv(arrival_rate=lam)
        var_env = RANSlicingEnvVar(arrival_rate=lam, alpha=1.0)

        rr_metrics = evaluate_agent(lambda _obs, t: t % rr_env.F, rr_env)
        baseline_metrics = evaluate_agent(
            lambda obs, _t: int(baseline_model.predict(obs, deterministic=True)[0]),
            baseline_env,
        )
        var_metrics = evaluate_agent(
            lambda obs, _t: int(var_model.predict(obs, deterministic=True)[0]),
            var_env,
        )

        rr_env.close()
        baseline_env.close()
        var_env.close()

        for agent_name, metrics in (
            ("fixed_rr", rr_metrics),
            ("baseline_ppo", baseline_metrics),
            ("var_ppo", var_metrics),
        ):
            viol = np.array(metrics["violation_rates"], dtype=np.float64)
            out = np.array(metrics["outage_rates"], dtype=np.float64)
            stats[agent_name]["viol_mean"].append(float(np.mean(viol)))
            stats[agent_name]["viol_std"].append(float(np.std(viol)))
            stats[agent_name]["out_mean"].append(float(np.mean(out)))
            stats[agent_name]["out_std"].append(float(np.std(out)))

        print(
            f"λ={lam:.1f} | fixed_rr viol={stats['fixed_rr']['viol_mean'][-1]:.3f} "
            f"| baseline viol={stats['baseline_ppo']['viol_mean'][-1]:.3f} "
            f"| var_ppo viol={stats['var_ppo']['viol_mean'][-1]:.3f}"
        )
        print(
            f"      |        outage={stats['fixed_rr']['out_mean'][-1]:.3f} "
            f"|          outage={stats['baseline_ppo']['out_mean'][-1]:.3f}"
            f"|        outage={stats['var_ppo']['out_mean'][-1]:.3f}"
        )

    os.makedirs("./figures", exist_ok=True)

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
        axes[0].plot(
            LAMBDA_LIST,
            viol_mean,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=linewidth,
            label=style["label"],
        )
        axes[0].fill_between(
            LAMBDA_LIST,
            viol_mean - viol_std,
            viol_mean + viol_std,
            color=style["color"],
            alpha=0.15,
        )

        out_mean = np.array(stats[agent_name]["out_mean"])
        out_std = np.array(stats[agent_name]["out_std"])
        axes[1].plot(
            LAMBDA_LIST,
            out_mean,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=linewidth,
            label=style["label"],
        )
        axes[1].fill_between(
            LAMBDA_LIST,
            out_mean - out_std,
            out_mean + out_std,
            color=style["color"],
            alpha=0.15,
        )

    axes[0].axhline(
        y=0.05,
        color="red",
        linestyle="--",
        linewidth=linewidth,
        label="SLA Threshold (5%)",
    )
    axes[0].set_ylabel("URLLC Violation Rate", fontsize=fontsize)
    axes[1].set_ylabel("eMBB Outage Rate", fontsize=fontsize)

    for ax in axes:
        ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=linewidth, label="Training λ")
        ax.set_xlabel("URLLC Arrival Rate λ (packets/minislot)", fontsize=fontsize)
        ax.legend(fontsize=fontsize)

    fig.suptitle("Robustness to Out-of-Distribution URLLC Arrival Rates", fontsize=fontsize)
    fig.tight_layout()
    fig.savefig("./figures/robustness.png", dpi=200)
    fig.savefig("./figures/robustness.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
