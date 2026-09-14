# Figure 4 ROBOMIMIC Square DPPO seed 42

## Status

**P5-A FORMAL RUN: PASS — 201/201 iterations completed.**

This is a single-seed state-observation result. It is not a claim of complete
multi-seed ROBOMIMIC reproduction or complete Figure 4 reproduction.

## Run definition and provenance

- Benchmark/task: ROBOMIMIC Square / NutAssemblySquare
- Method: DPPO fine-tuning from the official state checkpoint
- Observation: low-dimensional state, dimension 23; camera observation disabled
- Action: dimension 7
- Seed: 42
- Environment: 50 parallel environments, 400 rollout steps, headless MuJoCo
- Official config: `source/dppo_v0.6/cfg/robomimic/finetune/square/ft_ppo_diffusion_mlp.yaml`
- Denoising: `K=20`, fine-tuning `K'=10`, horizon 4, action steps 4
- Pretrained policy: `square_pre_diffusion_mlp_ta4_td20/.../checkpoint/state_8000.pt`
- Checkpoint SHA256: `368CE587294896B224EB549BA5F3A41CDE3EBD249A1F6625EADD6926695C14B1`
- Normalization: `data/robomimic/square/normalization.npz`
- Normalization SHA256: `68EC0ABFD989D5F0121F0E9A1DD074B49F859C167FF941C70D2EFF8006074FF3`
- Source submodule: `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`

The resolved Hydra configuration contains only path/logdir overrides relative
to the official YAML; the scientific settings remain unchanged, including
`n_train_itr=201`, `n_envs=50`, `n_steps=400`, `batch_size=10000`,
`update_epochs=10`, `val_freq=10`, `seed=42`, and `device=cuda:0`. The actor
uses the official `[1024, 1024, 1024]` MLP and `time_dim=32`.

The pre-start Square environment gate passed a 400-step state-only episode
with finite observations and rewards. The 2-env/3-iteration DPPO smoke also
passed with finite rollout, PPO, GAE, critic, and KL values.

## Completion and timing

- Completed rows: 201 (`itr=0` through `itr=200`)
- Automatic evaluation points: 21 (`itr=0, 10, ..., 200`)
- Final raw environment steps: 14,400,000
- Final high-level decision steps: 3,600,000 (`environment_steps / 4`)
- Run-log timestamp span: 2026-09-14 07:43:54.607 to 10:24:42.784
- Actual wall-clock: 9,648.177 s = 2.6800 h
- First 10 result rows: 497.326 s; average 49.733 s/row
- First-10 full-run projection: 9,996.254 s = 2.7767 h

Evaluation was performed automatically by the original training loop at
iterations `0, 10, ..., 200` (`val_freq=10`). The final evaluation at
iteration 200 was written after `state_200.pt` was saved.

## Success and reward results

| Quantity | Value |
|---|---:|
| Initial eval success rate (itr 0) | 0.4400 |
| Final eval success rate (itr 200) | 0.9950 |
| Best eval success rate | 1.0000 |
| First best-success iteration | 150 |
| Best-success environment steps | 10,800,000 |
| Initial eval reward | 67.8150 |
| Final eval reward (itr 200) | 293.6750 |
| Best eval reward | 293.6750 at itr 200 |

First evaluation points reaching the requested thresholds were: 0.50 at
iteration 10 / 720,000 steps; 0.70 at iteration 30 / 2,160,000 steps; 0.80
at iteration 40 / 2,880,000 steps; 0.90 at iteration 60 / 4,320,000 steps;
and 0.95 at iteration 100 / 7,200,000 steps. These are observed evaluation
points; no interpolation was used.

## Health and implementation audit

- NaN/Inf: none detected in the formal numeric results or training metrics
- Python exception: none during the corrected formal run
- CUDA OOM: none
- 50-env workers: stable through all 201 result rows
- Checkpoint saving: `state_0.pt`, `state_100.pt`, and `state_200.pt` saved
- Performance patch: active; submodule diff is only the approved CUDA denoising-discount vectorization in `model/diffusion/diffusion_ppo.py`
- Benchmark instrumentation: absent from the formal training source
- Reward/success definition: unchanged; original DPPO evaluation path used

One pre-start background wrapper attempt exited before environment construction
because of argument passing and produced no result. The corrected launch used
the same scientific configuration in a fresh formal run directory; this shell
issue is not counted as a formal training failure.

## Raw-data audit

`reports/P5A_RAW_DATA_AUDIT.json` is **PASS**. The first evaluation point
(itr 0), middle point (itr 100), and final point (itr 200) match between
`result.pkl` and the corresponding `run.log` values for success, reward, and
environment steps.

## Artifacts

- CSV: `results/fig4_robomimic_square_dppo_seed42.csv`
- Success curve: `plots/fig4_robomimic_square_dppo_seed42_success.png`
- Reward curve: `plots/fig4_robomimic_square_dppo_seed42_reward.png`
- Raw audit: `reports/P5A_RAW_DATA_AUDIT.json`
- Public evidence: `reports/evidence/fig4_robomimic_square_dppo_seed42/`
- Local run, result pickle, and checkpoints: `D:\AgentData\DPPO-P5-A\formal_run_20260914_074500`

The local checkpoint, `result.pkl`, environment, and intermediate outputs are
not uploaded to GitHub.
