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

The two smoke evaluations report zero completed episodes because four rollout steps are shorter than a full episode. This is an expected smoke-test limitation.

## Formal campaign status

Not complete at the time of this file's creation. Use the scripts under `scripts\train` to launch seed 42 and then the prepared five-seed campaign. Formal results must be parsed from the official run logs/checkpoints only. The benchmark implies approximately 25.4 hours for 1000 iterations of one Figure 4 run at this machine/runtime, before any queueing or recovery overhead.
