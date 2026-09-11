# Run log

All commands below were launched from the official checkout with the project-local WSL environment and headless MuJoCo variables set by the wrapper scripts.

## Completed

| Date/time | Test | Result | Output |
|---|---|---|---|
| 2026-09-08 | GPU CUDA smoke | PASS; CUDA available, RTX 5060 Ti, capability `(12, 0)`, finite 1024x1024 matmul | terminal diagnostic |
| 2026-09-08 | MuJoCo import | PASS after OSMesa/GLFW/GLEW and `patchelf` setup | terminal diagnostic |
| 2026-09-08 | D4RL Gym step | PASS for all three environments; reset and one step returned finite observations/rewards | terminal diagnostic |
| 2026-09-08 | DPPO model forward/loss/backward | PASS; finite samples, log-probabilities, loss and gradients | terminal diagnostic |
| 2026-09-08 22:21:55 | Hopper Figure 4 DPPO smoke | PASS; 2 iterations, 2 envs, 4 steps, checkpoints `state_0.pt`/`state_1.pt` | `logs\gym-finetune\hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10\2026-09-08_22-21-55_42` |
| 2026-09-08 22:22:30 | Hopper Figure 18 Gaussian-MLP smoke | PASS; 2 iterations, 2 envs, 4 steps, checkpoints `state_0.pt`/`state_1.pt` | `logs\gym-finetune\hopper-medium-v2_nopre_ppo_gaussian_mlp_ta1\2026-09-08_22-22-30_42` |
| 2026-09-08 22:27:40 | Hopper Figure 4 DPPO throughput benchmark | PASS; paper rollout settings `n_envs=40`, `n_steps=500`, 5 iterations; per-iteration times `92.7558`, `92.9952`, `91.1520`, `91.4278` s were emitted for iterations 1–4; iteration 0 was initialization/evaluation | `logs\gym-finetune\hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10\2026-09-08_22-27-40_42` |
| 2026-09-11 12:55:47 | Hopper DPPO engineering benchmark | PASS; official scale, seed 42, 5 iterations, iteration 0 eval and 1–4 training; patched training median 15.362 s/iteration | `logs\gym-finetune\hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10\2026-09-11_12-55-49_42` |
| 2026-09-11 12:57:36 | Hopper IDQL engineering benchmark | PASS; official scale, seed 42, 5 iterations, iteration 0 eval and 1–4 training; training median 32.010 s/iteration | `logs\gym-finetune\hopper-medium-v2_idql_diffusion_mlp_ta4_td20\2026-09-11_12-57-38_42` |
| 2026-09-11 13:00:35 | Hopper DIPO engineering benchmark | PASS; official scale, seed 42, 5 iterations, iteration 0 eval and 1–4 training; training median 36.694 s/iteration | `logs\gym-finetune\hopper-medium-v2_dipo_diffusion_mlp_ta4_td20\2026-09-11_13-00-36_42` |

The two smoke evaluations report zero completed episodes because four rollout steps are shorter than a full episode. This is an expected smoke-test limitation.

| 2026-09-11 20:11:29–2026-09-12 00:21:05 | Hopper Figure 4 DPPO formal run | COMPLETE; seed 42, 1000 iterations, 100 evaluation points, final training step 72,000,000; no NaN/Inf, exception, or CUDA OOM | `logs\gym-finetune\hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10\2026-09-11_20-11-30_42` |

## Formal campaign status

Hopper Figure 4 DPPO seed=42 full run: **COMPLETE**. The 2026-09-11 three-method runs remain engineering benchmarks only; they must not be presented as final Figure 4 results. IDQL and DIPO full runs have not been performed. This repository still does not claim a complete multi-seed or three-environment Figure 4 reproduction.

The formal run's public evidence copies are under `reports\evidence\fig4_hopper_dppo_seed42\` (`run.log`, `config.yaml`, and `overrides.yaml`). The original local run directory also retains `result.pkl` and all checkpoints, including the final local-only `state_999.pt`.
