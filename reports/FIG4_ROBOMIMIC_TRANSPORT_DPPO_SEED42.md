# Figure 4 ROBOMIMIC Transport DPPO seed 42

## Status

**P5-B FORMAL RUN: PASS — 201/201 iterations completed.**

This is a single-seed state-observation result. It is not a claim of complete
multi-seed ROBOMIMIC reproduction or complete Figure 4 reproduction.

## Run definition and provenance

- Benchmark/task: ROBOMIMIC Transport / TwoArmTransport
- Method: DPPO fine-tuning from the official state checkpoint
- Observation: low-dimensional state, dimension 59; camera observation disabled
- Action: dimension 14
- Robots: two Panda arms, `single-arm-opposed`
- Seed: 42
- Environment: 50 parallel environments, 800 episode horizon, 400 rollout steps, headless MuJoCo
- Official config: `source/dppo_v0.6/cfg/robomimic/finetune/transport/ft_ppo_diffusion_mlp.yaml`
- Denoising: `K=20`, fine-tuning `K'=10`, horizon 8, action steps 8
- Pretrained policy: `transport_pre_diffusion_mlp_ta8_td20/.../checkpoint/state_8000.pt`
- Checkpoint SHA256: `141E0178B2B31D4C0FCE505F0E62AB3E59EFE7E924FF3D492A9F59EA2AFB7485`
- Normalization: `data/robomimic/transport/normalization.npz`
- Normalization SHA256: `F6B893BDEE28AB798C1D4293D7DE565DB61680C57D8909D5C046771A027E5296`
- Source submodule: `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`

The resolved Hydra configuration contains only path/logdir overrides relative
to the official YAML; the scientific settings remain unchanged, including
`n_train_itr=201`, `n_envs=50`, `n_steps=400`, `batch_size=10000`,
`update_epochs=5`, `val_freq=10`, `seed=42`, and `device=cuda:0`. The actor
uses the official `[1024, 1024, 1024]` MLP and `time_dim=32`.

The pre-start Transport environment gate passed an 800-step state-only episode
with finite observations and rewards. The 2-env/3-iteration DPPO smoke passed
with finite Ta=8 rollout, PPO, GAE, critic, and KL values.

## Completion and timing

- Completed rows: 201 (`itr=0` through `itr=200`)
- Automatic evaluation points: 21 (`itr=0, 10, ..., 200`)
- Final raw environment steps: 28,800,000
- Final high-level decision steps: 3,600,000 (`environment_steps / 8`)
- Run-log timestamp span: 2026-09-14 10:59:13.016 to 21:17:51.238
- Actual wall-clock: 37,118.222 s = 10.3106 h
- First 10 result rows: 1,865.643 s; average 186.564 s/row
- First-10 full-run projection: 37,499.420 s = 10.4165 h

Evaluation was performed automatically by the original training loop at
iterations `0, 10, ..., 200` (`val_freq=10`). The final evaluation at
iteration 200 was written after `state_200.pt` was saved.

## Success and reward results

| Quantity | Value |
|---|---:|
| Initial eval success rate (itr 0) | 0.1800 |
| Final eval success rate (itr 200) | 0.9550 |
| Best eval success rate | 0.9950 |
| First best-success iteration | 180 |
| Best-success environment steps | 25,920,000 |
| Initial eval reward | 38.4700 |
| Final eval reward (itr 200) | 412.4000 |
| Best eval reward | 418.7800 at itr 180 |

First evaluation points reaching the requested thresholds were: 0.25 at
iteration 10 / 1,440,000 steps; 0.50 at iteration 30 / 4,320,000 steps; 0.70
at iteration 50 / 7,200,000 steps; 0.80 at iteration 60 / 8,640,000 steps;
and 0.90 at iteration 80 / 11,520,000 steps. These are observed evaluation
points; no interpolation was used.

## Health and implementation audit

- NaN/Inf: none detected; all 846 numeric values in `result.pkl` are finite
- Python exception: none during the formal run
- CUDA OOM: none
- 50-env dual-arm workers: stable through all 201 result rows
- Checkpoint saving: `state_0.pt`, `state_100.pt`, and `state_200.pt` saved
- Performance patch: active; submodule diff is only the approved CUDA denoising-discount vectorization in `model/diffusion/diffusion_ppo.py`
- Benchmark instrumentation: absent from the formal training source
- Peak RAM / VRAM: not instrumented in this formal run; no peak values are claimed
- Reward/success definition: unchanged; original DPPO evaluation path used

## Raw-data audit

`reports/P5B_RAW_DATA_AUDIT.json` is **PASS**. The first evaluation point
(itr 0), middle point (itr 100), and final point (itr 200) match between
`result.pkl` and the corresponding `run.log` values for success, reward, and
environment steps.

## Artifacts

- CSV: `results/fig4_robomimic_transport_dppo_seed42.csv`
- Success curve: `plots/fig4_robomimic_transport_dppo_seed42_success.png`
- Reward curve: `plots/fig4_robomimic_transport_dppo_seed42_reward.png`
- Raw audit: `reports/P5B_RAW_DATA_AUDIT.json`
- Public evidence: `reports/evidence/fig4_robomimic_transport_dppo_seed42/`
- Local run, result pickle, and checkpoints: `D:\AgentData\DPPO-P5-B\formal_run_20260914_110100`

The local checkpoint, `result.pkl`, environment, and intermediate outputs are
not uploaded to GitHub.
