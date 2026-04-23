"""Quick sanity checks for RANSlicingEnv reward signal and policy baselines."""

from __future__ import annotations

import copy

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


def urgency_aware_policy(env: RANSlicingEnv, obs: np.ndarray) -> int:
    """Simple non-learning baseline:
    - defer if queue empty;
    - puncture least-used subcarrier if queue exists, especially when urgent.
    """
    queue_len_norm = float(obs[0])
    min_deadline_norm = float(obs[1])
    puncture_norm = obs[2:]

    if queue_len_norm <= 0.0:
        return env.F

    if min_deadline_norm <= (1.0 / env.deadline_D):
        return int(np.argmin(puncture_norm))

    # queue_len_norm is normalized by T (typically 140), so any positive
    # value means there is backlog worth serving immediately.
    if queue_len_norm > 0.0:
        return int(np.argmin(puncture_norm))

    return env.F


def myopic_greedy_policy(env: RANSlicingEnv, obs: np.ndarray) -> int:
    del obs
    best_action = 0
    best_reward = -float("inf")
    for action in range(env.F + 1):
        env_clone = copy.deepcopy(env)
        _, reward, _, _, _ = env_clone.step(action)
        if reward > best_reward:
            best_reward = float(reward)
            best_action = action
    return best_action


def action_gap_diagnostic(episodes: int = 10, max_steps_per_ep: int = 30, seed_offset: int = 500):
    """Estimate one-step action distinguishability via max-min immediate reward gap."""
    gaps = []

    for ep in range(episodes):
        env = RANSlicingEnv()
        obs, _ = env.reset(seed=seed_offset + ep)

        done = False
        t = 0
        while not done and t < max_steps_per_ep:
            action_rewards = []
            for action in range(env.F + 1):
                env_clone = copy.deepcopy(env)
                _, r, _, _, _ = env_clone.step(action)
                action_rewards.append(float(r))

            gaps.append(max(action_rewards) - min(action_rewards))

            random_action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(random_action)
            done = terminated or truncated
            t += 1

        env.close()

    gap_arr = np.asarray(gaps, dtype=np.float64)
    return {
        "mean_gap": float(np.mean(gap_arr)),
        "p25_gap": float(np.percentile(gap_arr, 25)),
        "median_gap": float(np.median(gap_arr)),
        "p75_gap": float(np.percentile(gap_arr, 75)),
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

    urgency_results = run_policy(
        "urgency_aware",
        urgency_aware_policy,
        episodes=20,
        seed_offset=300,
    )

    myopic_results = run_policy(
        "myopic_greedy",
        myopic_greedy_policy,
        episodes=20,
        seed_offset=400,
    )

    print(
        "Random policy (20 eps): "
        f"{random_results['mean_reward']:.3f} ± {random_results['std_reward']:.3f}"
    )

    mean_random_violations = float(np.mean(random_results["violations"]))
    mean_defer_violations = float(np.mean(defer_results["violations"]))
    if mean_defer_violations <= mean_random_violations:
        raise AssertionError(
            "all-defer should incur more URLLC violations than random; "
            f"defer={mean_defer_violations:.3f}, random={mean_random_violations:.3f}"
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
    for result in (random_results, defer_results, puncture0_results, urgency_results, myopic_results):
        mean_viol = float(np.mean(result["violations"]))
        print(
            f"{result['policy']:<19} | {result['mean_reward']:11.3f} |"
            f" {result['std_reward']:10.3f} | {mean_viol:15.3f}"
        )

    if myopic_results["mean_reward"] <= random_results["mean_reward"]:
        print(
            "\n⚠️ WARNING: myopic-greedy does not outperform random; "
            "reward/action signal may still be weak."
        )
    else:
        print(
            "\n✅ Heuristic check passed: myopic-greedy outperforms random, "
            "suggesting learnable action signal exists."
        )

    gap_stats = action_gap_diagnostic(episodes=10, max_steps_per_ep=30, seed_offset=500)
    print("\nOne-step action-gap diagnostic (random rollouts):")
    print(
        f"  mean_gap={gap_stats['mean_gap']:.4f}, p25={gap_stats['p25_gap']:.4f}, "
        f"median={gap_stats['median_gap']:.4f}, p75={gap_stats['p75_gap']:.4f}"
    )


if __name__ == "__main__":
    main()
