# Next Round Plan (Codex)

## 1) 本轮主要瓶颈判断

**结论：本轮首要瓶颈是 URLLC 违约（高负载下 SLA 失守），其次才是 eMBB outage/variance；训练稳定性不是当前第一矛盾。**

依据：
- 当前配置是 baseline guardrail（`mode=baseline`, `traffic_model=poisson`, `reward_profile=default`），可复现实验成立。  
- 鲁棒性结果显示三种策略（`fixed_rr` / `baseline_ppo` / `var_ppo`）在 λ=0.8、1.0、1.5 时 URLLC 违约率分别约 `0.063`、`0.150`、`0.517`，全部超过 SLA 阈值 `0.05`。  
- 且三种策略的 summary 几乎完全一致，说明当前瓶颈更像是**负载能力边界/策略区分度不足**，而不是单纯某个 reward 版本调参不稳。

## 2) 下一轮改动实验计划（小步、可回退、保留 baseline）

### Phase A：Baseline 复现实验（必须先跑，作为对照）
目标：确认你当前环境能稳定复现已有结论，作为后续任何改动的对照组。

- 固定：`mode=baseline`, `traffic_model=poisson`, `reward_profile=default`, `seed=42`
- 仅增加训练步数与评估 episode，降低统计噪声。
- 输出目录单独命名，便于回滚和对比。

### Phase B：Baseline 内微调（仍属于 baseline，不改 traffic/reward）
目标：优先缓解高负载 URLLC 违约，不破坏 baseline 定义。

建议仅动以下“训练超参”（一次只改 1~2 项）：
1. `total_timesteps`: 100k → 200k（先验证是否欠训练）。
2. `--early-stop`: 先关闭拿完整学习曲线；若 plateaus 明显，再开启并记录触发点。
3. 多 seed（42/43/44）重复，分离“真实收益”和“随机性波动”。

> 这一步可回退：只改命令参数，不改代码逻辑。

### Phase C：Research mode（必须显式标注，不可冒充 baseline）
目标：在公平对照下验证“是否必须改变 reward 才能压 URLLC 违约”。

- 显式使用：`--mode research`
- baseline 组仍保持默认（Poisson + default reward）
- research 组可加一个 URLLC-heavy reward 版本（例如提高 URLLC 违约惩罚权重），但必须：
  - 同样 timesteps、episodes、seed 集合
  - 同样 eval 与 robustness 流程
  - 报告中单独标注“research variant”，禁止称作 baseline

## 3) 可直接运行命令（train / eval / robustness）

> 以下命令都在仓库根目录执行。

### 3.1 Baseline 复现（推荐先跑）

```bash
python train.py \
  --run-name baseline_repro_t100k \
  --mode baseline \
  --traffic-model poisson \
  --reward-profile default \
  --seed 42 \
  --total-timesteps 100000

python evaluate_and_plot.py \
  --run-name baseline_repro_eval \
  --mode baseline \
  --traffic-model poisson \
  --reward-profile default \
  --seed 42 \
  --episodes 100

python robustness_test.py \
  --run-name baseline_repro_robust \
  --mode baseline \
  --traffic-model poisson \
  --reward-profile default \
  --seed 42 \
  --episodes 50
```

### 3.2 Baseline 内小步增强（不改 baseline 定义）

```bash
python train.py \
  --run-name baseline_t200k_seed42 \
  --mode baseline \
  --traffic-model poisson \
  --reward-profile default \
  --seed 42 \
  --total-timesteps 200000

python train.py \
  --run-name baseline_t200k_seed43 \
  --mode baseline \
  --traffic-model poisson \
  --reward-profile default \
  --seed 43 \
  --total-timesteps 200000

python train.py \
  --run-name baseline_t200k_seed44 \
  --mode baseline \
  --traffic-model poisson \
  --reward-profile default \
  --seed 44 \
  --total-timesteps 200000

python evaluate_and_plot.py \
  --run-name baseline_t200k_eval \
  --mode baseline \
  --traffic-model poisson \
  --reward-profile default \
  --seed 42 \
  --episodes 200

python robustness_test.py \
  --run-name baseline_t200k_robust \
  --mode baseline \
  --traffic-model poisson \
  --reward-profile default \
  --seed 42 \
  --episodes 100
```

### 3.3 Research 对照（显式标注）

> 前提：你已实现可选 research reward/profile（例如 `--reward-profile urllc_heavy`）。

```bash
python train.py \
  --run-name research_urllc_heavy_t200k \
  --mode research \
  --traffic-model poisson \
  --reward-profile urllc_heavy \
  --seed 42 \
  --total-timesteps 200000

python evaluate_and_plot.py \
  --run-name research_urllc_heavy_eval \
  --mode research \
  --traffic-model poisson \
  --reward-profile urllc_heavy \
  --seed 42 \
  --episodes 200

python robustness_test.py \
  --run-name research_urllc_heavy_robust \
  --mode research \
  --traffic-model poisson \
  --reward-profile urllc_heavy \
  --seed 42 \
  --episodes 100
```

## 4) 预期观察指标与 stop 条件

### 关键观察指标（按优先级）
1. **URLLC SLA 通过区间**：在 λ=0.8/1.0/1.5 下，`violation_rate_mean <= 0.05` 的覆盖范围是否扩大。  
2. **eMBB outage rate**：是否在 URLLC 改善时明显恶化。  
3. **eMBB throughput variance**：是否出现“平均 outage 降低但方差增大”的代价。  
4. **跨 seed 稳定性**：关键指标的 std 是否收敛（避免只在单 seed 有效）。

### 建议 stop / go 条件

**Stop（停止继续加算力）**
- 连续两轮（例如 100k→200k、200k→300k）在相同评估设置下：
  - λ=0.8 的 URLLC 违约均值改善 < 0.005（绝对值）
  - 且 eMBB outage/variance 无明显收益（改善 < 2%）
- 多 seed 结果方差不降反升，说明改动不稳。

**Go（进入下一阶段）**
- 在至少 2/3 seeds 下同时满足：
  - λ=0.8 的 URLLC 违约率降到 `<= 0.05`
  - eMBB outage 不高于 baseline 对照 +10%（相对）
  - eMBB variance 不高于 baseline 对照 +15%（相对）

---

## 一句话执行顺序
先做 **Baseline 复现** → 再做 **Baseline 内超参小步优化** → 最后再上 **Research reward 变体公平对照**。
