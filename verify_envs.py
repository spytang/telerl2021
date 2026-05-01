"""Verification script for minimal/rich RAN slicing environments."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from ran_env import RANSlicingEnv
from ran_env_var import RANSlicingEnvVar


def run_random_episodes(env_factory, episodes: int = 5, seed_offset: int = 0):
    rewards = []
    latency_violations = []
    embb_outages = []

    for ep in range(episodes):
        env = env_factory()
        obs, info = env.reset(seed=seed_offset + ep)
        if obs.shape != env.observation_space.shape:
            raise AssertionError(
                f"reset observation shape {obs.shape} does not match "
                f"space {env.observation_space.shape}"
            )
        if not env.observation_space.contains(obs):
            raise AssertionError("reset observation is outside observation_space")
        del info

        done = False
        total_reward = 0.0
        total_viol = 0
        total_outages = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)
            if obs.shape != env.observation_space.shape:
                raise AssertionError(
                    f"step observation shape {obs.shape} does not match "
                    f"space {env.observation_space.shape}"
                )
            if not env.observation_space.contains(obs):
                raise AssertionError("step observation is outside observation_space")
            total_reward += reward
            total_viol += step_info["urllc_latency_violations"]
            total_outages += step_info["embb_outages_this_step"]
            done = terminated or truncated

        rewards.append(total_reward)
        latency_violations.append(total_viol)
        embb_outages.append(total_outages)

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_urllc_latency_violations": float(np.mean(latency_violations)),
        "mean_embb_outages": float(np.mean(embb_outages)),
    }


def assert_variance_env_reward_bounded(episodes: int = 5, seed_offset: int = 1000) -> None:
    for ep in range(episodes):
        base_env = RANSlicingEnv()
        var_env = RANSlicingEnvVar()

        base_obs, base_info = base_env.reset(seed=seed_offset + ep)
        var_obs, var_info = var_env.reset(seed=seed_offset + ep)
        del base_obs, base_info, var_obs, var_info

        done = False
        while not done:
            action = base_env.action_space.sample()
            _, base_reward, base_terminated, base_truncated, _ = base_env.step(action)
            _, var_reward, var_terminated, var_truncated, _ = var_env.step(action)

            if var_reward > base_reward + 1e-12:
                raise AssertionError(
                    "RANSlicingEnvVar produced higher reward than baseline under "
                    f"identical seed/action. base={base_reward}, var={var_reward}"
                )

            done = (base_terminated or base_truncated) and (var_terminated or var_truncated)


def main() -> None:
    print("Running Gymnasium check_env()...")
    check_env(RANSlicingEnv(obs_mode="minimal"))
    check_env(RANSlicingEnv(obs_mode="rich"))
    check_env(RANSlicingEnvVar())
    print("check_env() passed for minimal, rich, and variance environments.\n")

    minimal_env = RANSlicingEnv(obs_mode="minimal")
    rich_env = RANSlicingEnv(obs_mode="rich")
    var_env = RANSlicingEnvVar()
    print(f"minimal observation shape: {minimal_env.observation_space.shape}")
    print(f"rich observation shape: {rich_env.observation_space.shape}")
    print(f"var observation shape: {var_env.observation_space.shape}\n")
    minimal_env.close()
    rich_env.close()
    var_env.close()

    base_stats = run_random_episodes(lambda: RANSlicingEnv(obs_mode="rich"), episodes=5, seed_offset=0)
    var_stats = run_random_episodes(lambda: RANSlicingEnvVar(), episodes=5, seed_offset=0)

    print("RANSlicingEnv (5 random episodes):")
    print(f"  mean_reward: {base_stats['mean_reward']:.6f}")
    print(
        "  mean urllc_latency_violations: "
        f"{base_stats['mean_urllc_latency_violations']:.6f}"
    )
    print(f"  mean embb_outages: {base_stats['mean_embb_outages']:.6f}\n")

    print("RANSlicingEnvVar (5 random episodes):")
    print(f"  mean_reward: {var_stats['mean_reward']:.6f}")
    print(
        "  mean urllc_latency_violations: "
        f"{var_stats['mean_urllc_latency_violations']:.6f}"
    )
    print(f"  mean embb_outages: {var_stats['mean_embb_outages']:.6f}\n")

    assert_variance_env_reward_bounded(episodes=5, seed_offset=1000)
    print("Assertion passed: RANSlicingEnvVar reward is always <= baseline reward.")


if __name__ == "__main__":
    main()
