# telerl2021 Fork: Tolerance-Aware DRL for URLLC/eMBB Puncturing

This repository is a modern Gymnasium + Stable-Baselines3 PPO fork of the
original `InsaneMonster/telerl2021` project for URLLC traffic scheduling on top
of scheduled eMBB resources.

The current version is designed for thesis-style experiments where the main
comparison is:

```text
fixed_static baseline < PPO < VarPPO
```

`fixed_rr` is retained as a stronger non-learning round-robin reference, but it
is not the only baseline and is not the main learning-method comparison.

## Main Changes from the Source Code

Compared with the source project and the earlier minimal-observation fork, this
version makes the following changes:

1. Modern RL stack:
   - Uses `gymnasium.Env`.
   - Uses Stable-Baselines3 `PPO`.
   - Removes dependence on the old TensorFlow 1.x / USienaRL training flow for
     the new experiments.

2. Tolerance-aware observation:
   - `ran_env.py` now supports both `obs_mode="minimal"` and `obs_mode="rich"`.
   - PPO and VarPPO are trained with `obs_mode="rich"`.
   - The rich observation exposes the current eMBB codeword tolerance `Cw`,
     remaining margin, outage flags, deadline context, slot/minislot context,
     and previous-step traffic statistics.

3. Practical MDP interpretation:
   - This version is not a strict POMDP with hidden `Cw`.
   - `Cw` is treated as a base-station observable or estimable reliability
     margin.
   - In a paper or thesis, this can be described as a tolerance-aware MDP,
     oracle-assisted MDP, or CSI-assisted MDP.
   - The practical justification is that a base station can estimate eMBB
     reliability margin from CSI, MCS, BLER prediction, link adaptation state,
     HARQ/decoder feedback, or related reliability indicators.

4. VarPPO reward shaping:
   - `ran_env_var.py` keeps the same transition dynamics as `ran_env.py`.
   - It adds variance/load-balancing penalties to encourage more even
     puncturing and more stable eMBB reliability.
   - The reward includes penalties based on puncture-count variance, remaining
     margin variance, eMBB throughput variance, and rolling throughput variance.

5. Common-score evaluation:
   - All agents are evaluated with the same score:

```text
common_score = -1.0 * total_urllc_violations
               -0.5 * total_embb_outages
```

   - This avoids making VarPPO look better merely because it was trained with a
     different reward.

6. Cleaner comparison set:
   - Performance plots compare four methods:
     `fixed_static`, `fixed_rr`, `PPO`, and `VarPPO`.
   - Robustness plots compare only `PPO` and `VarPPO`, so the figure directly
     shows the effect of variance-aware reward shaping under traffic shift.

## System Model

The simplified URLLC/eMBB puncturing environment uses fixed parameters:

```text
F = 12 subcarriers
Sigma = 10 slots per episode
minislots_per_slot = 14
episode length = 140 minislots
Cw in {1, 2, 3, 4}
action space = Discrete(F + 1)
```

At each minislot, URLLC packet arrivals follow a Poisson process with default
arrival rate `lambda = 0.5`. Each URLLC packet has deadline `D = 3` minislots.
The scheduler can puncture one eMBB subcarrier to serve one URLLC packet, or it
can defer.

Each slot has one eMBB codeword per subcarrier. Each codeword has a puncturing
tolerance `Cw`. If the number of punctures on that subcarrier exceeds `Cw`, the
corresponding eMBB codeword is counted as an outage.

## Observation Modes

`RANSlicingEnv` supports two observation modes.

### Minimal Observation

`obs_mode="minimal"` keeps the older compact state:

```text
[queue_length,
 min_remaining_deadline,
 per_subcarrier_puncture_counts]
```

With default `F = 12`, this has dimension `14`.

### Rich Observation

`obs_mode="rich"` is used by PPO and VarPPO. With default `F = 12` and
`deadline_D = 3`, it has dimension `60`.

It contains:

1. Queue length.
2. Minimum remaining deadline.
3. Deadline histogram.
4. Current slot index.
5. Minislot index within the current slot.
6. Remaining minislots in the current slot.
7. Per-subcarrier puncture counts.
8. Per-subcarrier tolerance `Cw`.
9. Per-subcarrier remaining margin `Cw - puncture_count`.
10. Per-subcarrier outage flags.
11. Previous arrivals, served packets, URLLC violations, and eMBB outages.

The key modeling point is that `Cw` is intentionally observable/estimable in
this version. This gives PPO useful reliability-margin information that a real
base station could approximate from channel and decoding feedback.

## Actions and Rewards

The action space is:

```text
0..F-1 : puncture the selected subcarrier and serve one URLLC packet
F      : defer
```

The base environment reward is:

```text
r_t = -1.0 * urllc_latency_violations_t
      -0.5 * embb_outages_t
```

The VarPPO environment reward is:

```text
r_t_var = r_t - alpha * (
    w_load   * Var(normalized_puncture_counts)
  + w_margin * Var(normalized_remaining_margins)
  + w_thr    * Var(per_codeword_eMBB_throughput)
  + w_roll   * Var(rolling_eMBB_throughput)
)
```

Default variance weights are defined in `ran_env_var.py`, and all new
hyperparameters have defaults.

## Compared Methods

### fixed_static

A naive static baseline. It always punctures subcarrier `0` when serving URLLC.
This is intentionally simple and usually weak because it concentrates damage on
one eMBB codeword.

### fixed_rr

A round-robin baseline. It punctures subcarriers in order:

```text
0, 1, 2, ..., 11, 0, 1, ...
```

It does not use `Cw`, but it naturally balances puncturing. It should be treated
as a stronger heuristic reference.

### PPO

Standard PPO trained on `RANSlicingEnv(obs_mode="rich")`. It uses the
tolerance-aware observation and the base reward.

### VarPPO

PPO trained on `RANSlicingEnvVar(obs_mode="rich")`. It sees the same rich
observation as PPO, but its training reward includes additional variance and
load-balancing penalties.

## Scripts

### Verify environments

```bash
python verify_envs.py
```

This checks:

- Gymnasium compatibility.
- Minimal observation shape.
- Rich observation shape.
- VarPPO observation shape.
- Random rollout validity.
- That VarPPO reward is never higher than the base reward under the same
  seed/action sequence because it subtracts an extra penalty.

Expected default shapes:

```text
minimal observation shape: (14,)
rich observation shape: (60,)
var observation shape: (60,)
```

### Train PPO and VarPPO

```bash
python train.py
```

This trains:

- `models/baseline_ppo/final_model.zip`
- `models/var_ppo/final_model.zip`

It also writes evaluation logs:

- `logs/baseline_ppo/evaluations.npz`
- `logs/var_ppo/evaluations.npz`

### Evaluate and plot

```bash
python evaluate_and_plot.py
```

This compares four methods:

```text
fixed_static
fixed_rr
PPO
VarPPO
```

It reports:

- Common score.
- URLLC violations.
- eMBB outages.
- eMBB throughput variance.
- Puncture-count variance.
- Remaining-margin variance.

It generates:

- `figures/training_curves.png`
- `figures/training_curves.pdf`
- `figures/performance_bar.png`
- `figures/performance_bar.pdf`
- `figures/variance_dist.png`
- `figures/variance_dist.pdf`

### Robustness test

```bash
python robustness_test.py
```

This evaluates only:

```text
PPO
VarPPO
```

under shifted URLLC arrival rates:

```text
lambda in {0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5}
```

It generates:

- `figures/robustness.png`
- `figures/robustness.pdf`

In the robustness figure, lower curves are better because both plotted metrics
are error rates:

- URLLC violation rate.
- eMBB outage rate.

## How to Read the Main Figures

### Training Curves

The training curves show each agent's own training/evaluation reward. PPO and
VarPPO rewards are not directly comparable because VarPPO subtracts an extra
variance penalty. Therefore, VarPPO can have a lower training curve while still
being better under the common score.

Use the training curve mainly to check whether each model is learning and
stabilizing.

### performance_bar

Use `Common Score` as the main performance metric. It is shared by every agent,
so it is the fairest comparison.

For the other panels:

- Lower URLLC violation rate is better.
- Lower eMBB outage rate is better.
- Lower eMBB throughput variance is more stable.
- Lower puncture-count variance means better load balancing.
- Lower margin variance means reliability margins are more evenly protected.

### variance_dist

This plot shows the distribution of puncture-count variance across episodes.
Lower values mean puncturing is more evenly spread across subcarriers.

### robustness

This figure compares only PPO and VarPPO. Lower curves mean better robustness
under shifted URLLC traffic load. If VarPPO rises more slowly as `lambda`
increases, it is more robust under traffic distribution shift.

## Installation

Create and activate an environment, then install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "numpy>=1.24" "gymnasium>=0.29" "stable-baselines3>=2.0" "matplotlib>=3.7" "pandas>=1.5"
```

Then run:

```bash
python verify_envs.py
python train.py
python evaluate_and_plot.py
python robustness_test.py
```

`evaluate_and_plot.py` and `robustness_test.py` require trained model files from
`train.py`.

## File Map

```text
ran_env.py            # Base Gymnasium environment with minimal/rich observations
ran_env_var.py        # VarPPO environment with variance-aware reward shaping
train.py              # PPO and VarPPO training
evaluate_and_plot.py  # Four-method evaluation and figures
robustness_test.py    # PPO vs VarPPO robustness under shifted arrival rates
verify_envs.py        # Environment checks and rollout sanity tests
README.md             # Current experiment description
```

## Reference

This fork is based on:

```bibtex
@article{saggese2021deep,
  title={Deep Reinforcement Learning for URLLC data management on top of scheduled eMBB traffic},
  author={Saggese, Fabio and Pasqualini, Luca and Moretti, Marco and Abrardo, Andrea},
  journal={arXiv preprint arXiv:2103.01801},
  year={2021}
}
```
