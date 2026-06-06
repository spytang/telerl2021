# Final Thesis Code Archive Manifest

This archive keeps the modern Gymnasium + Stable-Baselines3 implementation used
by the final thesis experiments.

## Included Runtime Code

- `ran_env.py`: base tolerance-aware RAN slicing Gymnasium environment.
- `ran_env_var.py`: VarPPO reward-shaping environment.
- `train.py`: trains PPO and VarPPO models.
- `evaluate_and_plot.py`: evaluates fixed baselines, PPO, and VarPPO and
  regenerates the main comparison figures.
- `robustness_test.py`: evaluates PPO and VarPPO under shifted URLLC arrival
  rates.
- `verify_envs.py`: environment compatibility and sanity checks.

## Included Reproducibility Artifacts

- `models/`: trained PPO and VarPPO checkpoints.
- `logs/`: Stable-Baselines3 evaluation logs used for training curves.
- `figures/`: generated thesis experiment figures.
- `README.md`: current experiment description.
- `requirements-final.txt`: dependencies for the final modern implementation.

## Intentionally Excluded Legacy Code

The old `phy/`, `rl/`, `phy_execute/`, `phy_results/`, and `experiments/`
trees belong to the original TensorFlow 1.x / USienaRL workflow and are not used
by the final thesis implementation described in `README.md`.
