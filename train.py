"""Train rich-observation PPO agents for RAN slicing and compare baselines."""

import os
from typing import Callable, Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from ran_env import RANSlicingEnv
from ran_env_var import RANSlicingEnvVar


PolicyFn = Callable[[np.ndarray, int, RANSlicingEnv], int]


def evaluate_policy_on_common_env(
    policy_fn: PolicyFn, n_episodes: int = 100, arrival_rate: float = 0.5
) -> Dict[str, np.ndarray]:
    """Evaluate any policy on the same rich-observation env and common score."""
    env = RANSlicingEnv(obs_mode="rich", arrival_rate=arrival_rate)

    common_scores: List[float] = []
    violations: List[int] = []
    outages: List[int] = []
    puncture_var: List[float] = []
    margin_var: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        step_idx = 0

        total_violations = 0
        total_outages = 0
        puncture_var_series: List[float] = []
        margin_var_series: List[float] = []

        while not done:
            action = int(policy_fn(obs, step_idx, env))
            obs, _reward, terminated, truncated, info = env.step(action)
            total_violations += int(info.get("urllc_latency_violations", 0))
            total_outages += int(info.get("embb_outages_this_step", 0))
            puncture_var_series.append(float(info.get("puncture_count_variance", 0.0)))
            margin_var_series.append(float(info.get("remaining_margin_variance", 0.0)))
            done = terminated or truncated
            step_idx += 1

        common_scores.append(-1.0 * total_violations - 0.5 * total_outages)
        violations.append(total_violations)
        outages.append(total_outages)
        puncture_var.append(float(np.mean(puncture_var_series)))
        margin_var.append(float(np.mean(margin_var_series)))

    env.close()
    return {
        "common_scores": np.array(common_scores, dtype=np.float64),
        "violations": np.array(violations, dtype=np.float64),
        "outages": np.array(outages, dtype=np.float64),
        "puncture_var": np.array(puncture_var, dtype=np.float64),
        "margin_var": np.array(margin_var, dtype=np.float64),
    }


def summarize(metrics: Dict[str, np.ndarray]) -> Tuple[float, float, float, float]:
    rewards = metrics["common_scores"]
    violations = metrics["violations"]
    outages = metrics["outages"]
    return (
        float(np.mean(rewards)),
        float(np.std(rewards)),
        float(np.mean(violations)),
        float(np.mean(outages)),
    )


def main() -> None:
    np.random.seed(42)

    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs/baseline_ppo", exist_ok=True)
    os.makedirs("./logs/var_ppo", exist_ok=True)
    os.makedirs("./models/baseline_ppo", exist_ok=True)
    os.makedirs("./models/var_ppo", exist_ok=True)

    env_baseline = RANSlicingEnv(obs_mode="rich")
    env_var = RANSlicingEnvVar(alpha=1.0)

    eval_env_baseline = Monitor(RANSlicingEnv(obs_mode="rich"))
    eval_env_var = Monitor(RANSlicingEnvVar(alpha=1.0))

    callback_baseline = EvalCallback(
        eval_env_baseline,
        n_eval_episodes=20,
        eval_freq=5000,
        log_path="./logs/baseline_ppo/",
        best_model_save_path="./models/baseline_ppo/",
        deterministic=True,
        verbose=0,
    )
    callback_var = EvalCallback(
        eval_env_var,
        n_eval_episodes=20,
        eval_freq=5000,
        log_path="./logs/var_ppo/",
        best_model_save_path="./models/var_ppo/",
        deterministic=True,
        verbose=0,
    )

    model_baseline = PPO(
        "MlpPolicy",
        env_baseline,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=1400,
        batch_size=140,
        n_epochs=10,
        learning_rate=3e-4,
        seed=42,
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
        seed=42,
        verbose=1,
    )

    model_baseline.learn(total_timesteps=200_000, callback=callback_baseline)
    model_var.learn(total_timesteps=200_000, callback=callback_var)

    model_baseline.save("./models/baseline_ppo/final_model")
    model_var.save("./models/var_ppo/final_model")

    metrics_static = evaluate_policy_on_common_env(lambda _obs, _t, _env: 0, n_episodes=100)
    metrics_rr = evaluate_policy_on_common_env(lambda _obs, t, env: t % env.F, n_episodes=100)
    metrics_baseline = evaluate_policy_on_common_env(
        lambda obs, _t, _env: int(model_baseline.predict(obs, deterministic=True)[0]),
        n_episodes=100,
    )
    metrics_var = evaluate_policy_on_common_env(
        lambda obs, _t, _env: int(model_var.predict(obs, deterministic=True)[0]),
        n_episodes=100,
    )

    static_summary = summarize(metrics_static)
    rr_summary = summarize(metrics_rr)
    baseline_summary = summarize(metrics_baseline)
    var_summary = summarize(metrics_var)

    print("\nAgent                | Common Score | Std Score | Mean Violations | Mean Outages")
    print("-" * 82)
    print(
        f"fixed_static        | {static_summary[0]:12.3f} | {static_summary[1]:9.3f} |"
        f" {static_summary[2]:15.3f} | {static_summary[3]:12.3f}"
    )
    print(
        f"fixed_rr            | {rr_summary[0]:12.3f} | {rr_summary[1]:9.3f} |"
        f" {rr_summary[2]:15.3f} | {rr_summary[3]:12.3f}"
    )
    print(
        f"baseline_ppo        | {baseline_summary[0]:12.3f} | {baseline_summary[1]:9.3f} |"
        f" {baseline_summary[2]:15.3f} | {baseline_summary[3]:12.3f}"
    )
    print(
        f"var_ppo             | {var_summary[0]:12.3f} | {var_summary[1]:9.3f} |"
        f" {var_summary[2]:15.3f} | {var_summary[3]:12.3f}"
    )

    env_baseline.close()
    env_var.close()
    eval_env_baseline.close()
    eval_env_var.close()


if __name__ == "__main__":
    main()
