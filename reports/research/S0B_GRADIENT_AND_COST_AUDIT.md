# S0b 梯度结构与计算成本诊断

日期：2026-09-15。范围：冻结 Square / Transport BC 与现存 FT checkpoint 的诊断；未训练、未接入 S1、未正式长跑、未做 counterfactual 大规模实验。

## 结论

**STATUS：INCONCLUSIVE**。**RESEARCH DECISION：NO-GO（当前廉价 state-aware 机制）**。oracle 在 last-layer 固定 anchor 上显示空间，但廉价 proposal 未跨任务通过，且把实际 audit 成本计入后 compute gate 不通过。由于 200-draw MSE 使用 last-layer 梯度代理而不是全参数逐项梯度，不能把本轮写成完整的 full-parameter 机制证据；因此 S1 必须继续禁止。

## B0 / 预算

- local main HEAD = `d82ab2ce6e96fb1a0b2a4573ae7d15b1f9630df1`；origin/main 相同。submodule HEAD = `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`；dirty 仅为已批准 `model/diffusion/diffusion_ppo.py` 性能补丁，pointer 未变，正式科学 source 未新增 diff。
- runtime 沿用 S0a 的 Python 3.12.3、torch 2.7.1+cu128、robosuite 1.4.1、robomimic 0.5.0、mujoco-py 2.1.2.14 / legacy MuJoCo 2.1.0。未安装或升级依赖。
- NEW FROZEN DIAGNOSTIC ROLLOUT：Square/Transport 各 BC、FT 16 episodes × 4 boundaries × 10 transitions，chain 分别 `[64,11,4,7]` 与 `[64,11,8,14]`；primitive steps 合计 **38,400**。6 h 与 1.5M 上限均未触达。NPZ 仅本地 `D:\AgentData\DPPO-S0b`。

## B1 / B2 correctness

B1 在真实 Square FT 冻结 rollout 的前 16 states / 160 actor items 上：official scalar actor loss = `-1.19209291e-08`，逐项 mean = `-1.19209291e-08`，相对误差 = **0**；official 与重建 full-parameter actor gradient norm 都为 `0.59005928`，相对误差 = **0**。LOSS EQUIVALENCE：**PASS**。

B2 float64 toy 全枚举：loss 绝对误差 `2.22e-16`，gradient 最大绝对误差 `1.11e-16`，IMPORTANCE CORRECTION：**PASS**。IS 权重放在 PPO clipping 后的逐项 loss 外，未做 self-normalization/clipping。

## Fixed-anchor MSE

每个模型 64 states / 640 items，M=320，200 个独立 sampling RNG。下表是 **last-layer gradient proxy** 的 MSE ratio；uniform=1。

| task / policy | timestep-only | state-aware | oracle | state-aware vs timestep |
|---|---:|---:|---:|---:|
| Square / BC | 0.899 | 1.250 | 0.600 | 1.390 |
| Square / FT | 0.837 | 1.673 | 0.658 | 2.000 |
| Transport / BC | 0.929 | 0.598 | 0.249 | 0.644 |
| Transport / FT | 0.903 | 1.512 | 0.710 | 1.675 |

Oracle/uniform 在四个组合均 ≤ 0.75（0.600、0.658、0.249、0.710），说明理论非均匀抽样空间存在。state-aware 未通过 Square 两个组合，且跨任务方向不一致；因此 CHEAP PROPOSAL 与 NOVEL MECHANISM 均 **FAIL / NO-GO**。timestep-only 在 Square FT 与 Transport FT 约 0.837 / 0.903，不能替代跨任务机制证据。

200 次 sampling noise 是内层重复，不当作 200 个环境样本。每个指标的 MC standard error、ESS、weight 分布和 episode-cluster bootstrap 原始统计保存在 `S0B_RAW_METRICS.csv` 与 external JSON；由于本轮 MSE 是 last-layer proxy，cluster CI 不被宣称为 full-parameter ratio CI。

## theta-old 与 offline update

所有 theta-old ratio mean = 1、clip fraction = 0。1 step 后 Square BC/FT clip fraction = 0.831/0.714，Transport BC/FT = 0.563/0.728；3 steps 后为 0.909/0.839、0.928/0.941。它们是 **offline diagnostic update**，仅在内存副本执行，未覆盖 checkpoint，不能称为新的 RL 训练结果。

## B5 cost

3-cycle mean（每周期 64-state diagnostic batch；rollout 字段是 16-state model-only sampling，不冒充完整环境 wall time）：

| task | rollout | old logprob | actor fwd | actor bwd | critic | proposal | I/O | f | h（含 audit） | predicted speedup（含 audit） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Square FT | 28.56 | 3.09 | 47.80 | 46.03 | 3.83 | 0.67 | 2.60 | 0.708 | 28.47 | 0.034 |
| Transport FT | 19.43 | 1.84 | 48.17 | 34.78 | 3.28 | 0.62 | 1.42 | 0.757 | 34.11 | 0.029 |

不含 audit 的理想化 actor-only speedup 为 2×，但这是上限假设；Amdahl 计算必须把 proposal、audit、forward/backward、critic、rollout、I/O 一起看。本轮实测 full-parameter audit prefix 160 items 约 3.7 s，而单周期基线约 0.11–0.13 s，故 `proposal + audit <= 5%` 与 `compute >= 10%` 均 **FAIL**。

## 交付与限制

- Report：本文件；Raw metrics：`S0B_RAW_METRICS.csv`；Manifest：`S0B_MANIFEST.json`；plots：`plots/research/s0b_gradient_mse.png`、`plots/research/s0b_cost_breakdown.png`。
- full-parameter norms 仅按门槛对前 16 states streaming 审计；没有永久保存巨型逐项梯度。完整 640-item full-parameter MSE 未完成，这是 STATUS=INCONCLUSIVE 的主要原因。
- reward scaler 是仅 fit episodes 的 diagnostic scaler；raw reward 保留；diagnostic GAE 不等于历史 training GAE。Transport native ever-success 仍为 MISSING，未从 reward 反推、未编造 phase/handover。历史 checkpoint 仍不可严格 resume。
- 本轮没有进入 S1；等待上级审核 GitHub。

### Decision gates

| gate | result |
|---|---|
| loss equivalence | PASS |
| importance correction | PASS |
| oracle space | PASS on last-layer proxy |
| cheap proposal | FAIL |
| beyond timestep-only | FAIL |
| compute >=10% | FAIL with audit included |
| proposal/audit <=5% | FAIL with audit included |
