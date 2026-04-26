"""Train PPO agents for RAN slicing and compare against fixed round-robin."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from ran_env import RANSlicingEnv
from ran_env_var import RANSlicingEnvVar

DEFAULT_TOTAL_TIMESTEPS = 100_000
DEFAULT_SEED = 42
DEFAULT_ARRIVAL_RATE = 0.5
DEFAULT_ALPHA = 0.3
DEFAULT_RUN_NAME = "train"
DEFAULT_OUTPUT_DIR = "runs"


class PlateauEarlyStopCallback(BaseCallback):
    """Optionally stop training if eval mean reward plateaus."""

    def __init__(
        self,
        eval_callback: EvalCallback,
        enabled: bool = False,
        patience_evals: int = 6,
        min_improvement: float = 1e-3,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_callback = eval_callback
        self.enabled = enabled
        self.patience_evals = patience_evals
        self.min_improvement = min_improvement
        self._seen_eval_count = 0
        self._best_mean = -np.inf
        self._no_improve_count = 0

    def _on_step(self) -> bool:
        eval_results = self.eval_callback.evaluations_results
        if eval_results is None:
            return True

        if len(eval_results) <= self._seen_eval_count:
            return True

        latest_eval = np.array(eval_results[-1], dtype=np.float64)
        mean_reward = float(np.mean(latest_eval))
        self._seen_eval_count = len(eval_results)

        if mean_reward > self._best_mean + self.min_improvement:
            self._best_mean = mean_reward
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if self.enabled and self._no_improve_count >= self.patience_evals:
            if self.verbose:
                print(
                    "Early stop: plateau detected after "
                    f"{self._no_improve_count} evaluation rounds."
                )
            return False

        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lambda", dest="arrival_rate", type=float, default=DEFAULT_ARRIVAL_RATE)
    parser.add_argument("--arrival-rate", dest="arrival_rate", type=float)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--mode", choices=["baseline", "research"], default="baseline")
    parser.add_argument("--traffic-model", type=str, default="poisson")
    parser.add_argument("--reward-profile", choices=["default", "urllc_heavy", "risk_aware"], default="default")
    return parser.parse_args()


def create_run_dir(run_name: str, output_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def evaluate_policy_on_baseline_env(
    model: PPO,
    n_episodes: int = 100,
    seed: int = DEFAULT_SEED,
    reward_profile: str = "default",
) -> Dict[str, np.ndarray]:
    env = RANSlicingEnv(
        arrival_rate=DEFAULT_ARRIVAL_RATE,
        reward_profile=reward_profile,
        include_channel_state=reward_profile == "risk_aware",
    )

    rewards: List[float] = []
    violations: List[int] = []
    outages: List[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        total_violations = 0
        total_outages = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            total_violations += int(info.get("urllc_latency_violations", 0))
            total_outages += int(info.get("embb_outages_this_step", 0))
            done = terminated or truncated

        rewards.append(total_reward)
        violations.append(total_violations)
        outages.append(total_outages)

    env.close()
    return {
        "rewards": np.array(rewards, dtype=np.float64),
        "violations": np.array(violations, dtype=np.float64),
        "outages": np.array(outages, dtype=np.float64),
    }


def evaluate_fixed_rr(
    n_episodes: int = 100,
    seed: int = DEFAULT_SEED,
    reward_profile: str = "default",
) -> Dict[str, np.ndarray]:
    env = RANSlicingEnv(
        arrival_rate=DEFAULT_ARRIVAL_RATE,
        reward_profile=reward_profile,
        include_channel_state=reward_profile == "risk_aware",
    )

    rewards: List[float] = []
    violations: List[int] = []
    outages: List[int] = []

    for ep in range(n_episodes):
        _, _ = env.reset(seed=seed + ep)
        done = False
        t = 0

        total_reward = 0.0
        total_violations = 0
        total_outages = 0

        while not done:
            action = t % 12
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            total_violations += int(info.get("urllc_latency_violations", 0))
            total_outages += int(info.get("embb_outages_this_step", 0))
            done = terminated or truncated
            t += 1

        rewards.append(total_reward)
        violations.append(total_violations)
        outages.append(total_outages)

    env.close()
    return {
        "rewards": np.array(rewards, dtype=np.float64),
        "violations": np.array(violations, dtype=np.float64),
        "outages": np.array(outages, dtype=np.float64),
    }


def summarize(metrics: Dict[str, np.ndarray]) -> Tuple[float, float, float, float]:
    rewards = metrics["rewards"]
    violations = metrics["violations"]
    outages = metrics["outages"]
    return (
        float(np.mean(rewards)),
        float(np.std(rewards)),
        float(np.mean(violations)),
        float(np.mean(outages)),
    )


def export_eval_csv(npz_path: Path, csv_path: Path) -> None:
    if not npz_path.exists():
        return

    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timesteps", "eval_mean_reward", "eval_std_reward"])
        for idx, t in enumerate(timesteps):
            vals = np.array(results[idx], dtype=np.float64)
            writer.writerow([int(t), float(np.mean(vals)), float(np.std(vals))])


def plateau_diagnosis(npz_path: Path) -> Dict[str, float | bool | str]:
    if not npz_path.exists():
        return {"available": False, "message": "No EvalCallback data found."}

    data = np.load(npz_path)
    results = np.array(data["results"], dtype=np.float64)
    means = np.mean(results, axis=1)

    if len(means) < 4:
        return {
            "available": True,
            "plateau_detected": False,
            "message": "Insufficient evaluations for plateau diagnosis.",
        }

    window = min(4, len(means) // 2)
    recent = means[-window:]
    previous = means[-2 * window : -window]
    recent_mean = float(np.mean(recent))
    previous_mean = float(np.mean(previous))
    delta = recent_mean - previous_mean
    plateau = abs(delta) < 1.0

    return {
        "available": True,
        "plateau_detected": plateau,
        "recent_mean": recent_mean,
        "previous_mean": previous_mean,
        "delta": float(delta),
        "window": window,
    }


def main() -> None:
    args = parse_args()
    if args.mode == "baseline":
        if args.traffic_model != "poisson" or args.reward_profile != "default":
            raise ValueError(
                "Baseline mode must keep --traffic-model=poisson and --reward-profile=default "
                "to preserve reproducibility."
            )
    np.random.seed(args.seed)

    run_dir = create_run_dir(args.run_name, args.output_dir)
    run_training_dir = run_dir / "training"
    run_training_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("./logs")
    models_dir = Path("./models")
    baseline_log_dir = logs_dir / "baseline_ppo"
    var_log_dir = logs_dir / "var_ppo"
    baseline_model_dir = models_dir / "baseline_ppo"
    var_model_dir = models_dir / "var_ppo"

    for path in [baseline_log_dir, var_log_dir, baseline_model_dir, var_model_dir]:
        path.mkdir(parents=True, exist_ok=True)

    config = {
        "run": {
            "run_name": args.run_name,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "run_dir": str(run_dir),
            "output_dir": args.output_dir,
        },
        "env": {
            "F": 12,
            "Sigma": 10,
            "minislots_per_slot": 14,
            "arrival_rate": args.arrival_rate,
            "deadline_D": 3,
            "traffic_model": args.traffic_model,
        },
        "reward_profile": args.reward_profile,
        "mode": args.mode,
        "baseline_policy": (
            "Default baseline must remain unchanged and reproducible: "
            "Poisson arrival + default reward. "
            "Research mode may add explicit, comparable, reversible variants only."
        ),
        "training": {
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "early_stop_enabled": args.early_stop,
            "eval_freq": 5000,
            "n_eval_episodes": 20,
        },
        "ppo": {
            "policy": "MlpPolicy",
            "policy_kwargs": {"net_arch": [64, 64]},
            "n_steps": 1400,
            "batch_size": 140,
            "n_epochs": 10,
            "learning_rate": 3e-4,
        },
        "variance_env": {"alpha": args.alpha},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    include_channel_state = args.reward_profile == "risk_aware"
    env_baseline = RANSlicingEnv(
        arrival_rate=args.arrival_rate,
        reward_profile=args.reward_profile,
        include_channel_state=include_channel_state,
    )
    env_var = RANSlicingEnvVar(
        alpha=args.alpha,
        arrival_rate=args.arrival_rate,
        reward_profile=args.reward_profile,
        include_channel_state=include_channel_state,
    )

    eval_env_baseline = Monitor(
        RANSlicingEnv(
            arrival_rate=args.arrival_rate,
            reward_profile=args.reward_profile,
            include_channel_state=include_channel_state,
        )
    )
    eval_env_var = Monitor(
        RANSlicingEnvVar(
            alpha=args.alpha,
            arrival_rate=args.arrival_rate,
            reward_profile=args.reward_profile,
            include_channel_state=include_channel_state,
        )
    )

    callback_baseline_eval = EvalCallback(
        eval_env_baseline,
        n_eval_episodes=20,
        eval_freq=5000,
        log_path=str(baseline_log_dir),
        best_model_save_path=str(baseline_model_dir),
        deterministic=True,
        verbose=0,
    )
    callback_var_eval = EvalCallback(
        eval_env_var,
        n_eval_episodes=20,
        eval_freq=5000,
        log_path=str(var_log_dir),
        best_model_save_path=str(var_model_dir),
        deterministic=True,
        verbose=0,
    )

    callback_baseline = CallbackList(
        [callback_baseline_eval, PlateauEarlyStopCallback(callback_baseline_eval, enabled=args.early_stop)]
    )
    callback_var = CallbackList(
        [callback_var_eval, PlateauEarlyStopCallback(callback_var_eval, enabled=args.early_stop)]
    )

    model_baseline = PPO(
        "MlpPolicy",
        env_baseline,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=1400,
        batch_size=140,
        n_epochs=10,
        learning_rate=3e-4,
        seed=args.seed,
        verbose=1,
    )

    model_var = PPO(
        "MlpPolicy",
        env_var,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=1400,
        batch_size=140,
        n_epochs=10,
        learning_rate=3e-4,
        seed=args.seed,
        verbose=1,
    )

    model_baseline.learn(total_timesteps=args.total_timesteps, callback=callback_baseline)
    model_var.learn(total_timesteps=args.total_timesteps, callback=callback_var)

    model_baseline.save(str(baseline_model_dir / "final_model"))
    model_var.save(str(var_model_dir / "final_model"))

    metrics_rr = evaluate_fixed_rr(n_episodes=100, seed=args.seed, reward_profile=args.reward_profile)
    metrics_baseline = evaluate_policy_on_baseline_env(
        model_baseline,
        n_episodes=100,
        seed=args.seed,
        reward_profile=args.reward_profile,
    )
    metrics_var = evaluate_policy_on_baseline_env(
        model_var,
        n_episodes=100,
        seed=args.seed,
        reward_profile=args.reward_profile,
    )

    rr_summary = summarize(metrics_rr)
    baseline_summary = summarize(metrics_baseline)
    var_summary = summarize(metrics_var)

    print("\nAgent         | Mean Reward | Std Reward | Mean Violations | Mean Outages")
    print("-" * 72)
    print(
        f"fixed_rr      | {rr_summary[0]:11.3f} | {rr_summary[1]:10.3f} |"
        f" {rr_summary[2]:15.3f} | {rr_summary[3]:12.3f}"
    )
    print(
        f"baseline_ppo  | {baseline_summary[0]:11.3f} | {baseline_summary[1]:10.3f} |"
        f" {baseline_summary[2]:15.3f} | {baseline_summary[3]:12.3f}"
    )
    print(
        f"var_ppo       | {var_summary[0]:11.3f} | {var_summary[1]:10.3f} |"
        f" {var_summary[2]:15.3f} | {var_summary[3]:12.3f}"
    )

    export_eval_csv(baseline_log_dir / "evaluations.npz", run_training_dir / "baseline_eval_metrics.csv")
    export_eval_csv(var_log_dir / "evaluations.npz", run_training_dir / "var_eval_metrics.csv")

    summary = {
        "fixed_rr": {
            "mean_reward": rr_summary[0],
            "std_reward": rr_summary[1],
            "mean_violations": rr_summary[2],
            "mean_outages": rr_summary[3],
        },
        "baseline_ppo": {
            "mean_reward": baseline_summary[0],
            "std_reward": baseline_summary[1],
            "mean_violations": baseline_summary[2],
            "mean_outages": baseline_summary[3],
            "plateau": plateau_diagnosis(baseline_log_dir / "evaluations.npz"),
        },
        "var_ppo": {
            "mean_reward": var_summary[0],
            "std_reward": var_summary[1],
            "mean_violations": var_summary[2],
            "mean_outages": var_summary[3],
            "plateau": plateau_diagnosis(var_log_dir / "evaluations.npz"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    env_baseline.close()
    env_var.close()
    eval_env_baseline.close()
    eval_env_var.close()


if __name__ == "__main__":
    main()
