# Hopper DPPO / IDQL / DIPO engineering benchmark

日期：2026-09-11  
项目：`D:\Desktop\my_project\paper_reproduction\DPPO`  
环境：Hopper-v2 / `hopper-medium-v2`  
用途：论文 Figure 4 复现前的工程链路和 wall-clock 成本核查

## 结论边界

本报告是 engineering benchmark，不是 Figure 4 最终结果，也不比较三种算法的最终学习效果。每个方法只运行了 5 个 iteration，其中 iteration 0 是官方 `val_freq=10` 规则下的 evaluation，iteration 1–4 是真实训练 iteration。完整 1000-iteration 训练、Walker2D、HalfCheetah、5 seeds、Figure 18 和 ROBOMIMIC 均未启动。

所有方法使用同一台机器、同一环境、seed 42、同一个官方 Hopper pretrained Diffusion Policy：

`D:\Desktop\my_project\paper_reproduction\DPPO\checkpoints\official\hopper_medium_pre_diffusion_mlp_ta4_td20_state_3000.pt`

官方 DPPO 源码子模块仍固定为 v0.6 commit `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`。DPPO 本地运行额外应用了 `reports/PERFORMANCE_PATCH.diff` 中记录的数学等价 CUDA 向量化 patch；应用脚本为 `scripts/setup/apply_performance_patch.ps1` 和 `.sh`。该 patch 不改变 DPPO 数学定义和论文超参数。IDQL/DIPO 没有套用该 DPPO 性能 patch，也没有额外优化；本轮仅对两个 agent 加入了边界计时 instrumentation，便于拆分 rollout/update。

## 配置来源与实际覆盖

配置来源均为官方目录：

`D:\Desktop\my_project\paper_reproduction\DPPO\source\dppo_v0.6\cfg\gym\finetune\hopper-v2`

| 方法 | 官方 config | 论文规模关键值 | 本轮唯一科学配置覆盖 |
|---|---|---|---|
| DPPO | `ft_ppo_diffusion_mlp.yaml` | `n_envs=40`, `n_steps=500`, `batch_size=50000`, `update_epochs=5`, `ft_denoising_steps=10` | `train.n_train_itr=5` |
| IDQL | `ft_idql_diffusion_mlp.yaml` | `n_envs=40`, `n_steps=500`, `replay_ratio=128`, `batch_size=1000`, `eval_sample_num=20` | `train.n_train_itr=5` |
| DIPO | `ft_dipo_diffusion_mlp.yaml` | `n_envs=40`, `n_steps=500`, `replay_ratio=64`, `batch_size=1000`, `action_gradient_steps=10` | `train.n_train_itr=5` |

每个实际 resolved config 已保存到对应日志目录的 `.hydra/config.yaml`、`.hydra/overrides.yaml` 和 `.hydra/hydra.yaml`。

## 训练 iteration 时间

计时使用 CUDA synchronize；`rollout` 包括该 iteration 的环境采样段，`update` 包括模型更新段，`total` 是从 iteration 开始到保存/日志完成的 wall-clock。Iteration 0 是 evaluation，未计入训练 median/mean。

| 方法 | itr 0 eval total (s) | itr 1 train (s) | itr 2 train (s) | itr 3 train (s) | itr 4 train (s) | train median (s) | train mean (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DPPO | 14.098 | 16.160 | 15.230 | 14.313 | 15.494 | **15.362** | **15.299** |
| IDQL | 10.319 | 31.574 | 31.632 | 32.388 | 33.464 | **32.010** | **32.264** |
| DIPO | 9.223 | 35.571 | 36.879 | 36.554 | 36.834 | **36.694** | **36.459** |

### Rollout / update 分解

| 方法 | training rollout median (s) | training rollout mean (s) | update median (s) | update mean (s) |
|---|---:|---:|---:|---:|
| DPPO | 13.886 | 13.698 | 1.286 | 1.381 |
| IDQL | 10.321 | 10.292 | 21.717 | 21.964 |
| DIPO | 9.632 | 9.485 | 27.025 | 26.965 |

DPPO 的 `[PERF]` 日志同时包含 `prep_s`；其训练 iteration 中 `prep` 均小于 0.35 s。DPPO 的向量化 patch 已实际生效，PPO update 约 1.2–1.8 s，而上一轮官方 baseline 为约 73.5–78.9 s。

## Telemetry

采样脚本为 `scripts/diagnostics/monitor_telemetry.py`，每 1 秒采样 WSL host CPU 与 NVIDIA GPU。由于不同方法运行时长不同，样本数只用于运行健康核查，不作 CPU benchmark。

| 方法 | samples | CPU avg / max (%) | GPU util avg / max (%) | VRAM min–max (MiB) | temp max (C) |
|---|---:|---:|---:|---:|---:|
| DPPO | 96 | 6.691 / 9.345 | 16.865 / 93 | 1915–3310 | 57 |
| IDQL | 157 | 5.791 / 11.185 | 27.045 / 46 | 2066–2330 | 54 |
| DIPO | 171 | 5.665 / 12.072 | 31.327 / 60 | 1881–2359 | 58 |

没有观察到显存耗尽、GPU 温度异常或 CPU oversubscription。每个方法的原始 telemetry CSV 位于 `reports/telemetry_*_hopper_benchmark.csv`。

## 健康性核查

- DPPO：PASS。iteration 0–4 完成，PPO update 日志存在，loss、reward 和 KL 均为 finite；无 exception。DPPO patch 生效。
- IDQL：PASS。iteration 0–4 完成，actor/critic update 时间均为正，actor loss 和 reward 均为 finite；无 exception。
- DIPO：PASS。iteration 0–4 完成，critic/actor update 时间均为正，actor/critic loss 和 reward 均为 finite；无 exception。
- 三种方法均出现官方 Gym 的维护告警和 Box bound precision warning；它们不是本轮运行失败。

## 完整 1000 iteration wall-clock 粗估

按官方 `val_freq=10`，将 1000 个 iteration 近似拆成 900 个训练 iteration 和 100 个 evaluation iteration，得到本机 Hopper-based 粗估：

| 方法 | 1000 iteration 估计 |
|---|---:|
| DPPO | 约 **4.16 小时** |
| IDQL | 约 **8.27 小时** |
| DIPO | 约 **9.45 小时** |

若忽略 evaluation、简单把训练 median 外推到 1000 次，则分别约为 4.27、8.89、10.19 小时。实际完整运行还会受到 checkpoint、日志、首次 CUDA warm-up、机器负载和 episode 分布影响。

将 Hopper 速度作为 Walker2D/HalfCheetah 的 lower/rough proxy，三算法 × 三环境 × 单 seed 的总机器 wall-clock 粗估约 **70 小时**（约 2.9 天）；Walker2D/HalfCheetah 本轮没有实测，不能把该数字当作它们的精确预测。

## 日志与证据路径

| 内容 | 本地路径 |
|---|---|
| DPPO resolved config / run log | `D:\Desktop\my_project\paper_reproduction\DPPO\logs\gym-finetune\hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10\2026-09-11_12-55-49_42` |
| IDQL resolved config / run log | `D:\Desktop\my_project\paper_reproduction\DPPO\logs\gym-finetune\hopper-medium-v2_idql_diffusion_mlp_ta4_td20\2026-09-11_12-57-38_42` |
| DIPO resolved config / run log | `D:\Desktop\my_project\paper_reproduction\DPPO\logs\gym-finetune\hopper-medium-v2_dipo_diffusion_mlp_ta4_td20\2026-09-11_13-00-36_42` |
| DPPO telemetry | `D:\Desktop\my_project\paper_reproduction\DPPO\reports\telemetry_dppo_hopper_benchmark.csv` |
| IDQL telemetry | `D:\Desktop\my_project\paper_reproduction\DPPO\reports\telemetry_idql_hopper_benchmark.csv` |
| DIPO telemetry | `D:\Desktop\my_project\paper_reproduction\DPPO\reports\telemetry_dipo_hopper_benchmark.csv` |
| Raw benchmark stdout | `D:\Desktop\my_project\paper_reproduction\DPPO\reports\dppo_hopper_benchmark_stdout.log`, `idql_hopper_benchmark_stdout.log`, `dipo_hopper_benchmark_stdout.log` |

## 工程结论与下一阶段

在本机当前环境中，DPPO 的向量化 patch 显著降低了 PPO update 的纯计算成本；IDQL 和 DIPO 的官方 replay/update 计算量更大。三条链路均可正常执行，但这 5 iteration 结果只支持工程成本和健康性判断，不支持 reward、收敛速度或算法优劣结论。

建议下一步由导师/上级综控 AI 审核本报告和两个 commit 后，再单独批准是否启动 Hopper 单 seed 的完整 Figure 4 训练。`Formal Figure 4 training` 仍 **NOT COMPLETE**。
