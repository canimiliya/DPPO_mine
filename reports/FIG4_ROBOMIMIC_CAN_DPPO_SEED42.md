# Figure 4 ROBOMIMIC Can DPPO seed 42

## Status

**P4-A FORMAL RUN: PASS — 151/151 iterations completed.**

This is a single-seed state-observation result. It is not a claim of complete
multi-seed ROBOMIMIC reproduction or complete Figure 4 reproduction.

## Run definition and provenance

- Benchmark/task: ROBOMIMIC Can / PickPlaceCan
- Method: DPPO fine-tuning from the official state checkpoint
- Observation: low-dimensional state, dimension 23
- Action: dimension 7
- Seed: 42
- Environment: 50 parallel environments, 300 policy rollout steps
- Official config: `source/dppo_v0.6/cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp.yaml`
- Denoising: `K=20`, fine-tuning `K'=10`, horizon 4, action steps 4
- Pretrained policy: `can_pre_diffusion_mlp_ta4_td20/.../checkpoint/state_5000.pt`
- Checkpoint SHA256: `61851045e6b516807826e3bda4270c9e4a086023f2dd85af00eadb59d9b98a1b`
- Normalization: `data/robomimic/can/normalization.npz`
- State-only environment: object observations enabled; camera and video disabled
- Source submodule: `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`

The resolved Hydra configuration contains only path/logdir overrides relative
to the official YAML; the scientific settings remain unchanged, including
`n_train_itr=151`, `n_steps=300`, `batch_size=7500`, `update_epochs=10`,
`val_freq=10`, `seed=42`, and `device=cuda:0`.

## Completion and timing

- Completed rows: 151 (`itr=0` through `itr=150`)
- Training iterations: 135; automatic evaluation iterations: 16
- Final raw environment steps: 8,100,000
- Final high-level decision steps: 2,025,000 (`environment_steps / 4`)
- Run-log timestamp span: 2026-09-13 23:02:39.512 to 2026-09-14 00:19:41.070
- Actual wall-clock: 4,621.558 s = 1.2838 h
- P4-PREP estimate: 1.34 h
- Absolute difference: 202.442 s = 0.0562 h faster than estimated
- Relative error: -4.20%

Evaluation was performed automatically by the original training loop at
iterations `0, 10, ..., 150` (`val_freq=10`). The final evaluation at
iteration 150 was written after `state_150.pt` was saved.

## Success and reward results

| Quantity | Value |
|---|---:|
| Initial eval success rate (itr 0) | 0.6150 |
| Final eval success rate (itr 150) | 0.9900 |
| Best eval success rate | 1.0000 |
| First best-success iteration | 120 |
| Best-success environment steps | 6,480,000 |
| Initial eval reward | 73.4300 |
| Final eval reward | 207.9450 |
| Best eval reward | 208.2700 at itr 140 |

The single-seed evaluation trajectory improves from 0.6150 to approximately
0.99–1.00 and then fluctuates slightly near the ceiling. This is a cautious
single-seed learning trend qualitatively compatible with the reported Can
DPPO trend; it does not establish statistical or full-paper reproduction.

## Health and implementation audit

- NaN/Inf: none detected in the corrected formal `run.log`
- Python exception: none during the corrected formal run
- CUDA OOM: none
- 50-env workers: stable through all 151 result rows
- Checkpoint saving: `state_0.pt`, `state_100.pt`, and final `state_150.pt` saved
- Performance patch: active; submodule diff is only the approved CUDA denoising-discount vectorization in `model/diffusion/diffusion_ppo.py`
- Benchmark instrumentation: absent from the formal training source; the formal log contains no benchmark timing markers
- Reward/success definition: unchanged; success uses the original `episode_best_reward >= best_reward_threshold_for_success` path with threshold 1

One pre-start launch attempt exited before environment construction because a
PowerShell wrapper lost `LD_LIBRARY_PATH`. The corrected launch used the same
scientific configuration in a fresh run directory; this pre-start shell issue
produced no iteration and is not counted as a formal training failure.

## Raw-data audit

`reports/P4A_RAW_DATA_AUDIT.json` is **PASS**. The first evaluation point
(itr 0), middle point (itr 80), and final point (itr 150) match between
`result.pkl` and the four-decimal values printed in `run.log` for both success
rate and evaluation reward.

## Artifacts

- CSV: `results/fig4_robomimic_can_dppo_seed42.csv`
- Success curve: `plots/fig4_robomimic_can_dppo_seed42_success.png`
- Reward curve: `plots/fig4_robomimic_can_dppo_seed42_reward.png`
- Raw audit: `reports/P4A_RAW_DATA_AUDIT.json`
- Public evidence: `reports/evidence/fig4_robomimic_can_dppo_seed42/`
- Local run, result pickle, and final checkpoint: `D:\AgentData\DPPO-P4-A\formal_run_20260913_2308`

The local checkpoint, `result.pkl`, environment, and intermediate outputs are
not uploaded to GitHub.
