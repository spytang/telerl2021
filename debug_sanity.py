"""Quick sanity checks for RANSlicingEnv reward signal and policy baselines."""

from __future__ import annotations

import numpy as np

from ran_env import RANSlicingEnv


def run_policy(policy_name: str, action_fn, episodes: int = 20, seed_offset: int = 0):
    rewards = []
    violations = []

    for ep in range(episodes):
        env = RANSlicingEnv()
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        total_reward = 0.0
        total_violations = 0

        while not done:
            action = int(action_fn(env, obs))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            total_violations += int(info["urllc_latency_violations"])
            done = terminated or truncated

        rewards.append(total_reward)
        violations.append(total_violations)
        env.close()

    rewards_arr = np.array(rewards, dtype=np.float64)
    violations_arr = np.array(violations, dtype=np.int32)
    return {
        "policy": policy_name,
        "rewards": rewards_arr,
        "violations": violations_arr,
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
    }


def main() -> None:
    random_results = run_policy(
        "random",
        lambda env, obs: env.action_space.sample(),
        episodes=20,
        seed_offset=0,
    )

    defer_results = run_policy(
        "all_defer",
        lambda env, obs: env.F,
        episodes=20,
        seed_offset=100,
    )

    puncture0_results = run_policy(
        "all_puncture_0",
        lambda env, obs: 0,
        episodes=20,
        seed_offset=200,
    )

    print(
        "Random policy (20 eps): "
        f"{random_results['mean_reward']:.3f} ± {random_results['std_reward']:.3f}"
    )

    mean_defer_violations = float(np.mean(defer_results["violations"]))
    assert mean_defer_violations > 20, (
        "all-defer policy should cause many URLLC violations; "
        f"got mean {mean_defer_violations:.3f}"
    )

    print(
        "All-puncture-subcarrier-0 reward distribution: "
        f"min={np.min(puncture0_results['rewards']):.3f}, "
        f"p25={np.percentile(puncture0_results['rewards'], 25):.3f}, "
        f"median={np.median(puncture0_results['rewards']):.3f}, "
        f"p75={np.percentile(puncture0_results['rewards'], 75):.3f}, "
        f"max={np.max(puncture0_results['rewards']):.3f}"
    )

    print("\nPolicy              | Mean Reward | Std Reward | Mean Violations")
    print("-" * 64)
    for result in (random_results, defer_results, puncture0_results):
        mean_viol = float(np.mean(result["violations"]))
        print(
            f"{result['policy']:<19} | {result['mean_reward']:11.3f} |"
            f" {result['std_reward']:10.3f} | {mean_viol:15.3f}"
        )

    trained_ppo_reference_reward = -6.0
    if random_results["mean_reward"] >= trained_ppo_reference_reward:
        print("\n⚠️ WARNING: PPO performs no better than random. Reward shaping is broken.")


if __name__ == "__main__":
    main()
