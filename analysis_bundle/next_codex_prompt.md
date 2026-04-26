# Next Codex Prompt

请基于 `analysis_bundle/` 中的文件分析当前实验状态，并给出下一轮建议。

## Baseline Guardrail (必须遵守)
Default baseline must stay unchanged and reproducible: Poisson arrival + default reward. In research mode, new traffic_model/reward_profile variants are allowed only when explicitly labeled, comparable against baseline, reversible, and never presented as baseline.

## 本轮配置（来自 config.json）
```json
{
  "run_name": "robustness",
  "mode": "baseline",
  "traffic_model": "poisson",
  "reward_profile": "default",
  "seed": 42,
  "episodes": 50
}
```

## 本轮已收集结果文件
- README.md (source: README.md)
- ai_report.md (source: runs/20260427_013722_robustness/ai_report.md)
- config.json (source: runs/20260427_013722_robustness/config.json)
- evaluate_and_plot.py (source: evaluate_and_plot.py)
- evaluation_metrics.csv (source: runs/20260427_013517_evaluate/evaluation/evaluation_metrics.csv)
- robustness_metrics.csv (source: runs/20260427_013722_robustness/robustness/robustness_metrics.csv)
- robustness_test.py (source: robustness_test.py)
- summary.json (source: runs/20260427_013722_robustness/summary.json)
- train.py (source: train.py)

## 缺失文件
- 无

## 请 Codex 执行
- 分析当前模型问题与主要瓶颈（训练稳定性、URLLC违约率、eMBB outage/方差权衡）。
- 给出下一轮实验建议（先小步、可回退、可对照 baseline）。
- 给出下一轮建议运行命令（train / eval / robustness）。
- 若使用 research mode，请显式标注并保持与 baseline 的公平对照。