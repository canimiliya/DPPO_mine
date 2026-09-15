# S0c 执行动作分布几何低成本诊断

日期：2026-09-15；STATUS：**FAIL**。全程 DIAGNOSTIC_ONLY。

## 结论与硬边界

- CURRENT_GRADIENT_BUDGET = **STOP**；没有恢复 S0b/BDUA，不进入 S1。
- PHENOMENON_GATE = **FAIL**；ONLINE_MEASUREMENT_GATE = **NOT_RUN**；COST_GATE = **NOT_RUN**。
- NOVELTY_STATUS = **CANDIDATE_ONLY**；ALGORITHM_EFFECTIVENESS = **NOT_TESTED**。
- NEW_ENVIRONMENT_STEPS = **0**；FORMAL_TRAINING_ITERATIONS = **0**；未运行 MuJoCo、未做正式训练或 optimizer step。

## 设计、语义和数据

仅使用 `D:\AgentData\DPPO-S0b\square_ft_frozen.npz`（state_200）与 `transport_ft_frozen.npz`（state_100）。FIT 为 episode 0–7，TEST 为 8–15；每状态 64 样本，A/B 为独立 reference blocks，paired tape 与 reference tapes 不重合。动作是官方归一化坐标、官方最终 clipping 后的 `[0:act_steps]` 全执行 horizon。adapter 保留 DDPM 20 步、FT 10 步、std floor=0.1、noise clip 与 final action clip。
Square FIT old-policy median bandwidth（锁定用于两任务）= **0.259338**。无偏 MMD² 保留负值；energy distance 使用归一化欧氏距离。adapter 等价性：Square chain/action max abs = 0/0；Transport = 0/0。

## S0b 解释修正沿用

MSE 含 FIT items；BC/FT 各自重拟合；bootstrap 不是 proposal-ratio CI；ridge 常数列被标准化消去且无截距；ratio 越界不是 PPO 梯度阻断；audit 计时含 640 个 last-layer 与 160 个 full 梯度；成本是混合微基准。本轮未重跑 S0b。

## 文献可读范围与 novelty

Flow-DPPO 页面给出逐步 Gaussian KL 替代 ratio clipping；TruDi 明确约束完整 diffusion trajectory KL；NCDPO 使用预采样噪声条件的最终动作 PPO；WPPG 针对 implicit pushforward policy 的 Wasserstein proximal 更新。它们覆盖了相关 pathwise Gaussian、整路径 trust region、noise-conditioned action 或 Wasserstein 方向；“执行边缘分布对内部路径/噪声耦合无关变化不敏感”的区别在本轮只能记为 CANDIDATE_ONLY，不能宣称新颖。

## 廉价 latent 基线与 held-out 结果

Square FIT 预注册选择的最佳 latent baseline：**gaussian_kl_mean**。
- Square TEST：Spearman mean across candidates = **0.281854**；RMSE = **0.00397495**。
- Square same-policy reference noise floor：MMD² mean/q025/q975 = 0.000252569/-0.00198053/0.00322437；energy mean/q025/q975 = 0.000143331/-0.000979205/0.0016269；shuffle MMD abs error = 0。
- Transport TEST：Spearman mean across candidates = **-0.0649642**；RMSE = **0.0111504**。
- Transport same-policy reference noise floor：MMD² mean/q025/q975 = -2.33499e-05/-0.00834347/0.0108056；energy mean/q025/q975 = -3.52565e-05/-0.00435485/0.00441085；shuffle MMD abs error = 1.21102e-07。

### Square TEST 候选动作分布汇总（A/B 独立 blocks）

| candidate | actual Gaussian budget | paired action L2 | MMD² A/B | energy A/B |
|---|---:|---:|---:|---:|
| D_all@0.001 | 0.000394814 | 3.3923e-05 | -6.66323e-05 / 8.21787e-05 | -4.37176e-05 / 9.19862e-05 |
| D_all@0.003 | 0.00121076 | 0.000109333 | 0.00102459 / 0.00070416 | 0.000485436 / 0.000360932 |
| D_all@0.010 | 0.00389666 | 0.000372673 | 0.00238441 / 0.00234674 | 0.00120218 / 0.00127592 |
| D_first@0.001 | 0.000908606 | 0.000272014 | 0.0010511 / 0.00136015 | 0.000502309 / 0.000688496 |
| D_first@0.003 | 0.0025348 | 0.000553393 | 0.00313939 / 0.0025353 | 0.00161526 / 0.00124448 |
| D_first@0.010 | 0.00892524 | 0.00133703 | 0.00915086 / 0.00984912 | 0.00454486 / 0.00504414 |
| D_last@0.001 | 0.000353342 | 2.36465e-05 | -0.000136415 / 0.000182695 | -1.26065e-05 / 0.000155682 |
| D_last@0.003 | 0.00106664 | 7.10118e-05 | 0.00135125 / 0.000551931 | 0.000613931 / 0.000259978 |
| D_last@0.010 | 0.00353442 | 0.000234317 | 0.00207482 / 0.00207531 | 0.00110349 / 0.00113189 |

### Transport TEST 候选动作分布汇总（A/B 独立 blocks）

| candidate | actual Gaussian budget | paired action L2 | MMD² A/B | energy A/B |
|---|---:|---:|---:|---:|
| D_all@0.001 | 0.00110833 | 0.000201716 | 0.000515154 / 0.00143601 | 0.00025131 / 0.000752837 |
| D_all@0.003 | 0.00323602 | 0.000411834 | 0.00169043 / 0.0037864 | 0.000821819 / 0.00199775 |
| D_all@0.010 | 0.0104618 | 0.00106502 | 0.00874573 / 0.00989932 | 0.00434721 / 0.00495522 |
| D_first@0.001 | 0.000894782 | 0.000412014 | 0.000347438 / 0.000862916 | 0.000235658 / 0.000369151 |
| D_first@0.003 | 0.00288376 | 0.00102295 | 0.00306139 / 0.00354445 | 0.00143127 / 0.00171241 |
| D_first@0.010 | 0.00988036 | 0.00319287 | 0.0161158 / 0.0164426 | 0.00774681 / 0.00776872 |
| D_last@0.001 | 0.001199 | 9.94556e-05 | 0.000696387 / -5.3718e-05 | 0.000340025 / -3.58314e-05 |
| D_last@0.003 | 0.00349632 | 0.000290163 | 0.0023863 / 0.00341922 | 0.00124618 / 0.00170102 |
| D_last@0.010 | 0.0112803 | 0.000941257 | 0.0091885 / 0.00878565 | 0.00457742 / 0.00436417 |

## G1 判定

匹配 TEST Gaussian budget 的候选对数：Square 1，Transport 4；matched execution-distance 最大比 = 2.99795。判定：**FAIL**。失败原因是 held-out Spearman 未达到 0.8（Square 0.281854，Transport -0.0649642），且没有形成预注册的跨任务稳定 residual mismatch 证据；不能把不可分辨写成没有差距。

## G2 / S0c-3

**NOT_RUN**：G1 did not pass; S0c-3 prohibited。因此没有执行 4-pair m=4 paired-U estimator，也没有执行 online measurement/cost phase；没有用增加采样量、改 bandwidth 或换回归器救 gate。

## 资产与交付

- elapsed wall-clock = 34.9723 s；180-min limit reached = **NO**。
- 原始大数组与 noise tapes（若需复核）留在 `D:\AgentData\DPPO-S0c`，Git 只提交脚本、CSV、小 manifest/report。
- 详见 `S0C_RAW_METRICS.csv` 与 `S0C_MANIFEST.json`；完整命令、runtime、checkpoint 与 split 记录在 manifest。

## FINAL DECISION

**NO-GO**。

中文解释：在现有 Square/Transport FT checkpoint 上，Gaussian/denoising-path 指标不能充分解释最终执行动作分布：预注册最佳 latent baseline 在 held-out 上仅约 0.282 和 -0.065 的 Spearman。匹配 Gaussian budget 下虽观察到部分 action-distance 差异，但最高约 2.998、未达到稳定跨任务 residual mismatch 的 G1 条件，且 independent A/B 与 noise floor 不能把它升级为机制证据。4-pair 低成本测量因 G1 失败而未运行，不能声称提供额外信息。没有测试任何算法效果，因此本轮不值得继续设计新算法；等待上级新的研究决策。
