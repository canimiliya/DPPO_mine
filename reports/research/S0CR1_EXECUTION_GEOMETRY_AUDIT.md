# S0c-R1 执行动作分布几何审计

本报告对应已知测量实现错误修正与一次有界复核；不构成新算法结果。

## Final fields

- TASK: S0c-R1
- STATUS: **INCONCLUSIVE**
- CURRENT_GRADIENT_BUDGET: **STOP**
- IMPLEMENTATION_AUDIT: **PASS**
- MEASUREMENT_VALIDITY: **VALID**
- G1_CLASSIFICATION: **INCONCLUSIVE**
- ONLINE_MEASUREMENT_GATE: **NOT_RUN**
- COST_GATE: **NOT_RUN**
- ALGORITHM_EFFECTIVENESS: **NOT_TESTED**
- NEW_ENVIRONMENT_STEPS: **0**
- FORMAL_TRAINING_ITERATIONS: **0**
- OPTIMIZER_STEPS: **0**

## Pre-flight and runtime

- Remote base: `73e69c1d83048f4b34c8edda6ffbc6ed76b5f479`; local HEAD: `73e69c1d83048f4b34c8edda6ffbc6ed76b5f479`; origin HEAD: `73e69c1d83048f4b34c8edda6ffbc6ed76b5f479`; submodule: `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`.
- Full task wall-clock: 43.3861 s; diagnostic script seconds: 43.3032 s; 60-min limit reached: **False**.
- Runtime: Python 3.12.3, torch 2.7.1+cu128, CUDA 12.8, GPU NVIDIA GeForce RTX 5060 Ti.
- MuJoCo/mujoco-py/robomimic/robosuite records and TwoArmTransport source/hash are in the manifest; no environment was instantiated.

## Regression tests

- State-item indexing: **PASS**
- Gaussian KL: **PASS**
- Independent MMD: **PASS**
- Gate truth table: **PASS**
- Sampler equivalence: **PASS**

## Corrected semantics

- The state-item helper maps state `s` to `s*K + j`, including non-contiguous states; the same helper is used by item preparation, Gaussian path evaluation, and direction-budget evaluation.
- The corrected Gaussian formula is `KL(old || new) = 0.5*(new_lv-old_lv + (exp(old_lv)+(old_mu-new_mu)^2)/exp(new_lv)-1)`. The sampling std floor remains and the diagnostic is not an exact clipped-kernel KL.
- MMD² is an unbiased independent-sample estimator with all cross terms. Energy is independent; paired action L2 is auxiliary only. Negative MMD² values remain in CSV.
- Bandwidth is locked from Square FIT old-policy samples: wrong old flatten dimension 7; correct dimension 28; value 9.90487.
- Affine calibration is fit only on Square FIT and locked for all held-out/Transport values; coefficients and RMSE are in the manifest.

## Held-out and G1

### Square
Main held-out Spearman (candidate equal average): **0.264622**; pooled rho (not gated): 0.731297.
Per-candidate rho: {"D_all@0.001": 0.11033724340175952, "D_all@0.003": 0.28042521994134895, "D_all@0.010": 0.38563049853372433, "D_first@0.001": 0.059017595307917885, "D_first@0.003": 0.22067448680351903, "D_first@0.010": 0.20894428152492667, "D_last@0.001": 0.07111436950146627, "D_last@0.003": 0.4915689149560116, "D_last@0.010": 0.5538856304985337}
Affine calibration: {"gaussian_kl_mean": {"intercept": 5.473741119948655e-05, "slope": 0.015440025533449272, "n_fit": 288, "heldout_rmse": {"square": 0.00012041211995346803, "transport": 0.0005297703777950428}}, "gaussian_kl_sum": {"intercept": 5.473741054744033e-05, "slope": 5.514294928293742e-05, "n_fit": 288, "heldout_rmse": {"square": 0.00012041211955280426, "transport": 0.0004694843179145517}}, "gaussian_kl_max_step": {"intercept": 7.079413234011898e-05, "slope": 0.003515354195363983, "n_fit": 288, "heldout_rmse": {"square": 0.00012664282887906177, "transport": 0.0005341693277850021}}, "direct_mu_l2_normalized": {"intercept": -6.428524994834152e-05, "slope": 0.01686399479515633, "n_fit": 288, "heldout_rmse": {"square": 0.00011077607752673773, "transport": 0.0005296543866799013}}}
Matched-budget pairs: 9; ratio/noise-floor records are in the manifest and CSV.
### Transport
Main held-out Spearman (candidate equal average): **0.603556**; pooled rho (not gated): 0.672547.
Per-candidate rho: {"D_all@0.001": 0.32368035190615835, "D_all@0.003": 0.7543988269794721, "D_all@0.010": 0.8159824046920819, "D_first@0.001": 0.5381231671554252, "D_first@0.003": 0.6209677419354838, "D_last@0.001": 0.40469208211143687, "D_last@0.003": 0.597873900293255, "D_last@0.010": 0.7727272727272726}
Affine calibration: {"gaussian_kl_mean": {"intercept": 5.473741119948655e-05, "slope": 0.015440025533449272, "n_fit": 288, "heldout_rmse": {"square": 0.00012041211995346803, "transport": 0.0005297703777950428}}, "gaussian_kl_sum": {"intercept": 5.473741054744033e-05, "slope": 5.514294928293742e-05, "n_fit": 288, "heldout_rmse": {"square": 0.00012041211955280426, "transport": 0.0004694843179145517}}, "gaussian_kl_max_step": {"intercept": 7.079413234011898e-05, "slope": 0.003515354195363983, "n_fit": 288, "heldout_rmse": {"square": 0.00012664282887906177, "transport": 0.0005341693277850021}}, "direct_mu_l2_normalized": {"intercept": -6.428524994834152e-05, "slope": 0.01686399479515633, "n_fit": 288, "heldout_rmse": {"square": 0.00011077607752673773, "transport": 0.0005296543866799013}}}
Matched-budget pairs: 7; ratio/noise-floor records are in the manifest and CSV.

### G1
Classification: **INCONCLUSIVE**
Reason: 证据未同时满足两任务多尺度 residual、独立 A/B、energy、cluster CI、noise-floor 与廉价 baseline 排除条件。

## Previous S0c

The previous S0c Gaussian-index, KL-sign, bandwidth, MMD cross-term, calibration, bootstrap, and gate claims are not used as current scientific evidence. The old S0C report/CSV/code remain unchanged historical artifacts. Adapter equivalence and frozen-input provenance are retained only where independently reverified.

## Final Chinese summary

上一轮 S0c 的 state→item 索引、Gaussian KL 符号、独立 MMD 全 cross term、full-horizon bandwidth、affine calibration、episode-cluster bootstrap 与 Gate 逻辑已在本轮通过 CPU regression tests 并在 frozen 输入上重新执行；因此旧 S0c 的对应数值和 NO-GO 机制结论已撤回。修正后的 Gaussian/denoising-path 指标在 Square 与 Transport 上的主 held-out Spearman 分别为 0.264622 与 0.603556，是否能解释执行动作分布由本报告的候选级结果和锁定的廉价 baseline 审计限定。跨任务、跨多个尺度的稳定 residual mismatch 只有在独立 MMD、energy、A/B、cluster CI 和 noise-floor 条件同时满足时才成立；当前注册 Gate 判定为 INCONCLUSIVE，所以仍不自动进入 G2、S1、BDUA 或正式训练。

G2, R2, S1, BDUA, online/cost gates and formal training were not run. Awaiting upper-level research review.
