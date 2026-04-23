"""Train PPO agents for RAN slicing and compare against fixed round-robin."""

import os
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from ran_env import RANSlicingEnv
from ran_env_var import RANSlicingEnvVar


def evaluate_policy_on_baseline_env(
    model: PPO, n_episodes: int = 100
) -> Dict[str, np.ndarray]:
    """Evaluate a PPO model on RANSlicingEnv for apples-to-apples comparison."""
    env = RANSlicingEnv()

    rewards: List[float] = []
    violations: List[int] = []
    outages: List[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
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


def evaluate_fixed_rr(n_episodes: int = 100) -> Dict[str, np.ndarray]:
    """Evaluate fixed round-robin policy: action(t) = t % 12."""
    env = RANSlicingEnv()

    rewards: List[float] = []
    violations: List[int] = []
    outages: List[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        del obs
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


def main() -> None:
    np.random.seed(42)

    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs/baseline_ppo", exist_ok=True)
    os.makedirs("./logs/var_ppo", exist_ok=True)
    os.makedirs("./logs/var_ppo_low_alpha", exist_ok=True)
    os.makedirs("./models/baseline_ppo", exist_ok=True)
    os.makedirs("./models/var_ppo", exist_ok=True)
    os.makedirs("./models/var_ppo_low_alpha", exist_ok=True)

    env_baseline = RANSlicingEnv()
    env_var = RANSlicingEnvVar(alpha=0.05)
    env_var_low_alpha = RANSlicingEnvVar(alpha=0.01)

    eval_env_baseline = Monitor(RANSlicingEnv())
    eval_env_var = Monitor(RANSlicingEnvVar(alpha=0.05))
    eval_env_var_low_alpha = Monitor(RANSlicingEnvVar(alpha=0.01))

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
    callback_var_low_alpha = EvalCallback(
        eval_env_var_low_alpha,
        n_eval_episodes=20,
        eval_freq=5000,
        log_path="./logs/var_ppo_low_alpha/",
        best_model_save_path="./models/var_ppo_low_alpha/",
        deterministic=True,
        verbose=0,
    )

    model_baseline = PPO(
        "MlpPolicy",
        env_baseline,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=560,
        batch_size=140,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42,
        verbose=1,
    )

    model_var = PPO(
        "MlpPolicy",
        env_var,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=560,
        batch_size=140,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42,
        verbose=1,
    )

    model_var_low_alpha = PPO(
        "MlpPolicy",
        env_var_low_alpha,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=560,
        batch_size=140,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42,
        verbose=1,
    )

    model_baseline.learn(total_timesteps=200_000, callback=callback_baseline)
    model_var.learn(total_timesteps=200_000, callback=callback_var)
    model_var_low_alpha.learn(total_timesteps=200_000, callback=callback_var_low_alpha)

    model_baseline.save("./models/baseline_ppo/final_model")
    model_var.save("./models/var_ppo/final_model")
    model_var_low_alpha.save("./models/var_ppo_low_alpha/final_model")

    metrics_rr = evaluate_fixed_rr(n_episodes=100)
    metrics_baseline = evaluate_policy_on_baseline_env(model_baseline, n_episodes=100)
    metrics_var = evaluate_policy_on_baseline_env(model_var, n_episodes=100)
    metrics_var_low_alpha = evaluate_policy_on_baseline_env(model_var_low_alpha, n_episodes=100)

    rr_summary = summarize(metrics_rr)
    baseline_summary = summarize(metrics_baseline)
    var_summary = summarize(metrics_var)
    var_low_alpha_summary = summarize(metrics_var_low_alpha)

    print("\nAgent              | Mean Reward | Std Reward | Mean Violations | Mean Outages")
    print("-" * 79)
    print(
        f"fixed_rr      | {rr_summary[0]:11.3f} | {rr_summary[1]:10.3f} |"
        f" {rr_summary[2]:15.3f} | {rr_summary[3]:12.3f}"
    )
    print(
        f"baseline_ppo  | {baseline_summary[0]:11.3f} | {baseline_summary[1]:10.3f} |"
        f" {baseline_summary[2]:15.3f} | {baseline_summary[3]:12.3f}"
    )
    print(
        f"var_ppo            | {var_summary[0]:11.3f} | {var_summary[1]:10.3f} |"
        f" {var_summary[2]:15.3f} | {var_summary[3]:12.3f}"
    )
    print(
        f"var_ppo_low_alpha  | {var_low_alpha_summary[0]:11.3f} | {var_low_alpha_summary[1]:10.3f} |"
        f" {var_low_alpha_summary[2]:15.3f} | {var_low_alpha_summary[3]:12.3f}"
    )

    env_baseline.close()
    env_var.close()
    env_var_low_alpha.close()
    eval_env_baseline.close()
    eval_env_var.close()
    eval_env_var_low_alpha.close()


if __name__ == "__main__":
    main()
