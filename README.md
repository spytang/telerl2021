# telerl2021 (Fork) — DRL for URLLC/eMBB Resource Slicing

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-%3E%3D0.29-orange)
![Stable--Baselines3](https://img.shields.io/badge/Stable--Baselines3-%3E%3D2.0-brightgreen)

A research-focused reimplementation and extension of **InsaneMonster/telerl2021**, modernizing the original codebase for contemporary reinforcement learning tooling and reproducible experimentation in beyond-5G/6G radio resource management.

This fork is based on:

- **Saggese, Pasqualini, Moretti, Abrardo (2021)**, *Deep Reinforcement Learning for URLLC data management on top of scheduled eMBB traffic* ([arXiv:2103.01801](https://arxiv.org/abs/2103.01801))
- Original reference repository: [InsaneMonster/telerl2021](https://github.com/InsaneMonster/telerl2021)

---

## Project Overview

This fork reformulates the URLLC-over-eMBB slicing problem as a **Gymnasium-compatible MDP** and provides:

1. A baseline environment implementing puncturing-based URLLC scheduling on top of pre-allocated eMBB resources.
2. A reward-variant environment with an explicit **eMBB throughput-variance penalty** to encourage risk-averse behavior.
3. A standardized PPO training pipeline (Stable-Baselines3).
4. Reproducible evaluation, plotting, and robustness testing utilities suitable for thesis and paper workflows.

The objective remains aligned with the original work: satisfy stringent URLLC latency constraints while limiting eMBB reliability degradation caused by puncturing.

---

## Key Differences from Original

Compared to the original repository, this fork introduces substantial architectural and experimental changes:

- **Framework modernization**:
  - From TensorFlow 1.15 + USienaRL to **Gymnasium + Stable-Baselines3 PPO**.
- **Environment portability**:
  - Clean `gymnasium.Env` implementation (`ran_env.py`) with explicit `reset()`, `step()`, and `render()`.
- **New objective design**:
  - Added `RANSlicingEnvVar` (`ran_env_var.py`) with a rolling-window variance regularizer on eMBB throughput.
- **Expanded experiment stack**:
  - Dedicated scripts for environment verification, PPO training, post-training analysis, and distribution-shift robustness tests.
- **Reproducibility improvements**:
  - Deterministic seeding protocol (`seed=42`) across training/evaluation scripts.

This is therefore **not a drop-in replica**; it is a methodologically faithful but technically modernized research fork.

---

## Environment Design

### System Model

The simulation follows the Saggese et al. problem setting with explicit fixed parameters:

- Subcarriers: **F = 12**
- Slots per episode: **Σ = 10**
- Minislots per slot: **14**
- Episode horizon: **T = Σ × 14 = 140 minislots**

### eMBB Model

- One eMBB codeword is associated with each subcarrier at each slot (12 codewords/slot).
- Each codeword has puncturing tolerance class `Cw` sampled at episode start:
  - `Cw ~ Uniform{1, 2, 3, 4}`
- If punctures exceed tolerance, that codeword is counted as an outage.

### URLLC Traffic Model

- Arrivals follow a Poisson process per minislot:
  - `N_t ~ Poisson(λ)` with default **λ = 0.5 packets/minislot**
- Per-packet deadline: **D = 3 minislots**
- Packets are handled with oldest-first service logic.

### State, Action, Reward

- **State** (dimension `2 + F = 14`, normalized to `[0,1]`):
  1. URLLC queue length
  2. Minimum remaining deadline in queue
  3. Per-codeword puncture counters (12 values)

- **Action space** (`Discrete(F + 1) = 13`):
  - `0..11`: puncture selected subcarrier for oldest URLLC packet
  - `12`: defer transmission (allowed only when queue dynamics permit)

- **Baseline reward**:

```text
r_t = -1 * (URLLC latency violations at t)
      -0.5 * (eMBB outages at t)
```

- **Episode termination**: at minislot `t = 140`.

### Variance-Penalized Variant

`RANSlicingEnvVar` keeps transition dynamics identical and modifies only reward:

```text
r_t^var = r_t - α * Var(throughput_window)
```

- `throughput_t = (F - outages_t) / F`
- rolling window length = 20 steps
- default `α = 0.3`

Motivation: variance regularization encourages smoother puncturing patterns and risk-averse scheduling, consistent with reliability-centric RAN slicing objectives.

---

## Training Pipeline

`train.py` trains three agents:

1. **baseline_ppo** on `RANSlicingEnv`
2. **var_ppo** on `RANSlicingEnvVar(alpha=0.3)`
3. **fixed_slicing** non-learning round-robin baseline:
   - action `f_t = t mod F`

### PPO Configuration

- policy: `MlpPolicy`
- network: `net_arch=[64, 64]`
- `n_steps=1400` (10 episodes per rollout)
- `batch_size=140`
- `n_epochs=10`
- `learning_rate=3e-4`
- `total_timesteps=100000`
- eval callback every 5000 steps (`20` eval episodes)
- seed: `42`

### Outputs

- Trained models:
  - `./models/baseline_ppo`
  - `./models/var_ppo`
- Evaluation logs:
  - `./logs/{agent_name}/`
- Terminal summary:
  - `agent | mean_reward | std_reward`

---

## Evaluation & Visualization

`evaluate_and_plot.py` evaluates all three agents over 100 episodes and records per-episode:

- `total_reward`
- `urllc_violation_rate`
- `embb_outage_rate`
- `embb_throughput_variance`

It generates publication-style figures in both PNG and PDF under `./figures/`:

1. **Training Convergence Comparison** (`training_curves.*`)
   - PPO baseline vs PPO variance-penalized
   - mean ± 1 std reward across eval episodes
2. **Performance Metrics Comparison** (`performance_bar.*`)
   - grouped bars over three agents and three reliability/variance metrics
3. **Distribution of eMBB Throughput Variance** (`variance_dist.*`)
   - boxplot across agents

Plot style conventions:

- `plt.style.use("seaborn-v0_8-paper")`
- `fontsize=12`
- `linewidth=2`

Additionally, script output includes a thesis-ready LaTeX table:

```text
| Agent | Reward↑ | URLLC Viol.↓ | eMBB Outage↓ | eMBB Var↓ |
```

---

## Robustness Testing

`robustness_test.py` measures generalization under shifted URLLC load:

- `λ ∈ {0.2, 0.5, 0.8, 1.0, 1.5}` packets/minislot
- 50 evaluation episodes per λ and per PPO agent
- compares `baseline_ppo` vs `var_ppo` on URLLC violation rate

Output figure:

- `./figures/robustness.png`
- includes SLA reference line:
  - horizontal dashed red line at `y = 0.05` (5% violation threshold)

---

## Installation & Usage

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install "numpy>=1.24" "gymnasium>=0.29" "stable-baselines3>=2.0" "matplotlib>=3.7"
```

### 3) Verify environments

```bash
python verify_envs.py
```

### 4) Train agents

```bash
python train.py
```

> Note: `evaluate_and_plot.py` expects `./logs/*/evaluations.npz` to draw training curves.
> With default `eval_freq=5000`, use `--total-timesteps >= 5000` if you want training-curve files in smoke tests.

### 5) Evaluate and generate figures

```bash
python evaluate_and_plot.py
```

### 6) Robustness testing

```bash
python robustness_test.py
```

### 7) Build analysis bundle for next Codex round

```bash
python tools/build_analysis_bundle.py
```

Then hand over `analysis_bundle/` to Codex for the next round analysis.

### Baseline/Reproducibility Principle (Important)

- Default baseline must remain unchanged and reproducible:
  - traffic model: **Poisson arrival**
  - reward profile: **default reward**
- Research mode can reserve explicit interfaces for variants (e.g., `--mode`, `--traffic-model`, `--reward-profile`),
  but any variant must be clearly labeled, comparable against baseline, and reversible.
- Variants must never be presented as baseline results.

---

## Quick Run Sequence

```bash
python train.py
python evaluate_and_plot.py
python robustness_test.py
python tools/build_analysis_bundle.py
```

## Experiment Artifacts (AI/Codex-Friendly)

Each major script now creates a timestamped run directory:

```text
runs/<timestamp>_<run_name>/
```

Typical artifacts include:

- `config.json` (full run configuration)
- `summary.json` (aggregated metrics)
- `ai_report.md` (concise data-driven interpretation)
- CSV exports:
  - `training/*.csv`
  - `evaluation/evaluation_metrics.csv`
  - `robustness/robustness_metrics.csv`
- copied figures under `runs/.../figures/` while preserving compatibility with `./figures/`

For future AI/Codex analysis, prefer using:

- `config.json`
- `summary.json`
- `evaluation_metrics.csv`
- `robustness_metrics.csv`
- `ai_report.md`

rather than relying only on static figures.

## File Structure

```text
.
├── ran_env.py               # Baseline Gymnasium environment
├── ran_env_var.py           # Reward-variant env (variance penalty)
├── verify_envs.py           # Env checker + random-policy sanity tests
├── train.py                 # PPO training and fixed-slicing baseline evaluation
├── evaluate_and_plot.py     # 100-episode evaluation + publication figures + LaTeX table
├── robustness_test.py       # Cross-load generalization test over arrival rates
├── models/                  # Saved PPO models
├── logs/                    # EvalCallback outputs
└── figures/                 # Generated PNG/PDF plots
```

---

## Citation / References

If you use this fork in academic work, please cite the original paper:

```bibtex
@article{saggese2021deep,
  title={Deep Reinforcement Learning for URLLC data management on top of scheduled eMBB traffic},
  author={Saggese, Fabio and Pasqualini, Luca and Moretti, Marco and Abrardo, Andrea},
  journal={arXiv preprint arXiv:2103.01801},
  year={2021}
}
```

Primary references:

- Saggese et al., 2021: https://arxiv.org/abs/2103.01801
- Original repository: https://github.com/InsaneMonster/telerl2021

---

## Attribution

This repository is a **fork with substantial modifications** in environment engineering, learning pipeline, objective design, and experimental tooling. Credit for the original research problem setup and initial implementation concept belongs to the original authors and repository maintainers.
