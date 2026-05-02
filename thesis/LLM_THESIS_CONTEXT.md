# LLM Thesis Maintenance Context

This note records the main facts learned during the May 2026 thesis review/editing pass. It is intended to help a future LLM or human editor continue the thesis work without re-discovering the whole repository.

## Repository And Thesis Structure

- Repository root: `/home/linya/projects/telerl2021`.
- Thesis main file: `thesis/main.tex`.
- Main body file included by `main.tex`: `thesis/converted_clean_body.tex`.
- Bibliography file: `thesis/references.bib`.
- Generated PDF: `thesis/main.pdf`.
- Main thesis topic: deep reinforcement learning for URLLC/eMBB puncturing and RAN slicing.
- Current codebase is a modern Gymnasium + Stable-Baselines3 PPO fork of `telerl2021`, not the old TensorFlow 1.x / USienaRL workflow.

## Core Code Facts To Keep Consistent

Primary files:

- `ran_env.py`: base Gymnasium environment `RANSlicingEnv`.
- `ran_env_var.py`: variance-aware reward variant `RANSlicingEnvVar`.
- `train.py`: trains PPO and VarPPO.
- `evaluate_and_plot.py`: evaluates four methods and generates main figures.
- `robustness_test.py`: evaluates PPO and VarPPO under shifted URLLC arrival rates.
- `verify_envs.py`: checks Gymnasium compatibility and observation shapes.
- `README.md`: current experiment description.

Default environment facts from `ran_env.py`:

- `F = 12` subcarriers.
- `Sigma = 10` slots per episode.
- `minislots_per_slot = 14`.
- Episode length `T = 140` minislots.
- Default URLLC arrival rate `lambda = 0.5`.
- URLLC deadline `D = 3` minislots.
- Action space is `Discrete(F + 1)`.
- Actions `0..F-1`: puncture selected subcarrier and serve at most one URLLC packet.
- Action `F`: defer.
- Each slot samples eMBB codeword tolerance `C_w` for each subcarrier from `{1,2,3,4}`.
- If puncture count on a subcarrier exceeds `C_w`, that eMBB codeword is counted as an outage.
- Base reward:
  `r_t = -1.0 * urllc_latency_violations_t - 0.5 * embb_outages_t`.

Observation modes:

- `obs_mode="minimal"`:
  `[queue_length, min_remaining_deadline, per_subcarrier_puncture_counts]`.
  Default shape `(14,)`.
- `obs_mode="rich"`:
  queue length, minimum deadline, deadline histogram, slot/minislot timing, remaining minislots, puncture counts, `C_w`, remaining margins, outage flags, previous arrivals/served/violations/outages.
  Default shape `(60,)`.
- PPO and VarPPO both use `obs_mode="rich"`.
- The thesis should describe the model as tolerance-aware MDP / CSI-assisted MDP / oracle-assisted reliability-margin MDP, not as a strict POMDP with hidden `C_w`.

VarPPO facts from `ran_env_var.py` and `train.py`:

- Class defaults in `RANSlicingEnvVar`:
  - `alpha = 0.3`
  - `load_variance_weight = 0.5`
  - `margin_variance_weight = 0.5`
  - `throughput_variance_weight = 0.5`
  - `rolling_variance_weight = 0.2`
  - `throughput_window_size = 20`
- Actual training and robustness scripts use `RANSlicingEnvVar(alpha=1.0)`.
- Thesis must distinguish class default `alpha=0.3` from experiment setting `alpha=1.0`.
- VarPPO reward subtracts penalties for:
  - normalized puncture-count variance
  - normalized remaining-margin variance
  - per-codeword eMBB throughput/outage-state variance
  - rolling eMBB throughput variance
- `common_score_step` remains the base reward in VarPPO info, for fair comparison.

PPO training facts from `train.py`:

- Trains two models:
  - baseline PPO on `RANSlicingEnv(obs_mode="rich")`
  - VarPPO on `RANSlicingEnvVar(alpha=1.0)`
- Stable-Baselines3 `PPO("MlpPolicy", ...)`.
- `policy_kwargs={"net_arch": [64, 64]}`.
- `n_steps = 1400`.
- `batch_size = 140`.
- `n_epochs = 10`.
- `learning_rate = 3e-4`.
- `seed = 42`.
- `total_timesteps = 200_000`.
- SB3 defaults retained unless explicitly set:
  - `gamma = 0.99`
  - `gae_lambda = 0.95`
  - `clip_range = 0.2`
  - `ent_coef = 0.0`
  - `vf_coef = 0.5`

Evaluation facts:

- `evaluate_and_plot.py` compares four methods:
  - `fixed_static`
  - `fixed_rr`
  - `PPO`
  - `VarPPO`
- Evaluation uses a common rich-observation base environment and common score:
  `common_score = -1.0 * total_URLLC_violations - 0.5 * total_eMBB_outages`.
- `robustness_test.py` compares only PPO and VarPPO.
- Robustness lambda list:
  `[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]`.
- Robustness episodes per lambda: `50`.
- Standard evaluation episodes: `100`.

Current figure/log/model artifacts exist:

- `figures/training_curves.png/.pdf`
- `figures/performance_bar.png/.pdf`
- `figures/variance_dist.png/.pdf`
- `figures/robustness.png/.pdf`
- `logs/baseline_ppo/evaluations.npz`
- `logs/var_ppo/evaluations.npz`
- `models/baseline_ppo/final_model.zip`
- `models/var_ppo/final_model.zip`
- Thesis copies of figures are in `thesis/`.

## Verified Numeric Results Used In Thesis

These were recomputed from the current model files and scripts during the review pass.

Base scenario, `lambda=0.5`, 100 episodes:

- `fixed_static`
  - common score: `-5.700`
  - URLLC violation rate: about `0.0071`
  - eMBB outage rate: about `0.0671`
- `fixed_rr`
  - common score: `-1.515`
  - URLLC violation rate: about `0.0071`
  - eMBB outage rate: about `0.0074`
- `PPO`
  - common score: `-1.290`
  - URLLC violation rate: about `0.0074`
  - eMBB outage rate: about `0.0036`
- `VarPPO`
  - common score: `-1.215`
  - URLLC violation rate: about `0.0072`
  - eMBB outage rate: about `0.0029`

Interpretation:

- PPO clearly reduces eMBB outage compared with fixed baselines.
- VarPPO is only modestly better than PPO in common score.
- VarPPO's clearest advantage is eMBB outage control, not a large URLLC violation reduction.

Robustness test highlights:

- At `lambda=1.5`, PPO and VarPPO both have URLLC violation rate about `0.517`.
- VarPPO eMBB outage rate is lower at high loads:
  - `lambda=1.0`: PPO about `0.0220`, VarPPO about `0.0170`
  - `lambda=1.2`: PPO about `0.0270`, VarPPO about `0.0221`
  - `lambda=1.5`: PPO about `0.0364`, VarPPO about `0.0286`

Interpretation:

- Do not claim VarPPO eliminates URLLC violations under extreme load.
- Do claim VarPPO improves eMBB outage control under traffic shift.
- High load is constrained by one URLLC service per minislot, `D=3`, and finite resources.

## Major Thesis Edits Already Made

Files changed:

- `thesis/main.tex`
- `thesis/converted_clean_body.tex`
- `thesis/main.pdf`

Main content changes:

- Added accurate abstract statements about current results.
- Replaced old/overstrong language such as:
  - zero-violation claims
  - strategy-collapse claims not directly supported by figures
  - "optimal algorithm choice"
  - claims that invoke Shannon limit for this abstract environment
- Corrected PPO hyperparameter table to match `train.py`.
- Added distinction between `RANSlicingEnvVar` class default `alpha=0.3` and actual experiment setting `alpha=1.0`.
- Added concrete current numbers for common score, violation rate, and outage rate.
- Reframed robustness conclusion around eMBB outage control.
- Removed `\nocite{*}` from `main.tex`.
- Added explicit in-text citations for all 43 entries in `references.bib`.

## Bibliography Status

After deleting `\nocite{*}`, all entries in `references.bib` are explicitly cited in the body.

Previously uncited entries that were integrated:

- `sun2019short`: Section 1.1, URLLC short-blocklength background.
- `luu2020`: Section 1.2.1, traditional/optimization-based slicing resource provisioning.
- `yang2019iov`: Section 1.2.2, IoV URLLC resource management.
- `sun2020vehicular`: Section 1.2.2, virtualized vehicular network resource slicing.
- `nahum2024`: Section 1.2.2, intent-aware RAN scheduling.
- `wang2022linkslice`: Section 1.2.2, fine-grained LinkSlice DRL slicing.
- `yagcioglu2026`: Section 1.2.2, DQN-ASRA style adaptive URLLC slicing.
- `nasir2019`: Section 1.2.2, MADRL dynamic power allocation.
- `ning2021`: Section 1.2.3, mission-critical puncturing under mixed services.
- `motalleb2023`: Section 1.2.4, Open RAN resource allocation using network slicing.
- `zangooei2024`: Section 1.2.4, constrained MARL for flexible Open RAN slicing.
- `rezazadeh2023`: Section 1.2.4, FDRL for scalable distributed 6G RAN slicing.
- `cui2026`: Section 5.3, future MADRL direction for B5G network slicing.

Do not restore `\nocite{*}` unless the school explicitly requires listing uncited references.

## Build And Validation Notes

Use:

```bash
cd /home/linya/projects/telerl2021/thesis
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Recent validation result:

- `latexmk` succeeds.
- No undefined references found in log.
- No missing figures found.
- No bibliography/Biber errors found.
- Log may contain harmless biblatex info such as missing optional style config files:
  - `gb7714-2015.dbx not found`
  - `biblatex-dm.cfg not found`
  These are info lines, not errors.

If running Python checks, use the conda environment:

```bash
/home/linya/miniconda3/envs/telerl/bin/python verify_envs.py
```

System `python` lacks `numpy`, but the `telerl` conda environment works.

`verify_envs.py` passes in `telerl` and reports:

- minimal observation shape `(14,)`
- rich observation shape `(60,)`
- var observation shape `(60,)`

## Suggested Future Improvements

Highest-value additions if time remains:

1. Add a compact numeric table in Chapter 4 with:
   - common score
   - URLLC violation rate
   - eMBB outage rate
   - puncture variance
   - margin variance
2. Add a small parameter-sensitivity study for VarPPO `alpha`, e.g. `0.3`, `0.5`, `1.0`, `2.0`.
3. Add a true ablation only if experiments are rerun, e.g.:
   - base PPO
   - load-variance-only PPO
   - full VarPPO

Important: do not invent ablation or sensitivity results. If no script/model/log exists, describe it only as future work.

## Quality Assessment Snapshot

For an undergraduate thesis, the current project is strong enough and likely in the good-to-excellent range if presented well:

- Strengths:
  - working codebase
  - reproducible Gymnasium environment
  - PPO and VarPPO models
  - evaluation plots and logs
  - robustness test
  - thesis now mostly aligned with current code
- Remaining risk:
  - claims must stay modest because VarPPO's improvement over PPO is real but small
  - model is abstract and single-cell
  - no true ablation or hyperparameter sensitivity yet

