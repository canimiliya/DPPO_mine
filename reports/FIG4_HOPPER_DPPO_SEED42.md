# Figure 4 Hopper DPPO seed=42

## Status

**P3-A FORMAL RUN: PASS**

This is the first complete formal Figure 4 single-seed run in this project. The run completed all 1000 configured iterations. It is not a multi-seed reproduction and does not complete the IDQL/DIPO or three-environment comparison.

## Pre-flight

- Remote base before run: `2a3212bdb4b8c2b6a4f69e19a44a83ef7d077ca6` (`fix: isolate clean DPPO performance patch`). Local `main` and `origin/main` matched.
- Official submodule HEAD: `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`.
- Only clean performance patch present: **YES**. The only submodule source modification was `model/diffusion/diffusion_ppo.py`, matching `reports/PERFORMANCE_PATCH.diff`.
- Resolved configuration: `logs/gym-finetune/hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-09-11_20-11-30_42/.hydra/config.yaml`.
- Scientific config changed: **NO**. Resolved values were `n_train_itr=1000`, `env.n_envs=40`, `train.n_steps=500`, `train.batch_size=50000`, `train.update_epochs=5`, `model.ft_denoising_steps=10`, `seed=42`, and `device=cuda:0`.
- Base policy: `checkpoints/official/hopper_medium_pre_diffusion_mlp_ta4_td20_state_3000.pt`.

## Run

- Environment: `hopper-medium-v2` / Hopper.
- Method: DPPO.
- Seed: 42.
- Iterations completed: 1000 (`itr=0` through `itr=999`).
- Final environment steps: 72,000,000.
- Run-log timestamp span: 4:09:03.064, from `20:12:02.116` to `00:21:05.180`; the run directory was created at `20:11:29`.
- Final local checkpoint: `logs/gym-finetune/hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-09-11_20-11-30_42/checkpoint/state_999.pt`.

The evaluation schedule is `val_freq=10`, so evaluation was performed at iterations 0, 10, ..., 990. Training reward is kept separate from evaluation reward in the CSV; no interpolation or replacement was used.

## Result

- Initial evaluation reward: `1436.6171779257947` at iteration 0 / 0 environment steps.
- Final evaluation reward: `3092.054468005484` at iteration 990 / 71,280,000 environment steps.
- Highest evaluation reward: `3092.054468005484` at iteration 990 / 71,280,000 environment steps.
- Evaluation points: 100.
- Curve: `plots/fig4_hopper_dppo_seed42.png`.
- Raw table: `results/fig4_hopper_dppo_seed42.csv`.

The single-seed curve is noisy but shows an overall improvement from the initial evaluation to the final evaluation. The cautious qualitative conclusion is: **single-seed trend basically agrees with the learning/improvement direction of the paper's Hopper DPPO curve**. This does not establish full Figure 4 reproduction because the paper reports multi-seed statistics.

## Health and provenance

- NaN/Inf: **none detected** in `run.log` or the parsed result records.
- Exception: **none detected**.
- CUDA OOM: **none detected**.
- Performance patch active: **YES**; the local submodule diff is the denoising-discount CUDA vectorization recorded in `reports/PERFORMANCE_PATCH.diff`.
- Benchmark instrumentation present: **NO** in the formal performance patch or formal source files.
- Checkpoint integrity: `state_0.pt` through `state_900.pt` and final `state_999.pt` were written; the final checkpoint remains local-only.

### RAW DATA AUDIT: PASS

The CSV was generated directly from the 1000-record `result.pkl`. Three evaluation points were checked against both sources:

| Point | Iteration | Environment steps | `result.pkl` | `run.log` display | Absolute difference |
|---|---:|---:|---:|---:|---:|
| First | 0 | 0 | 1436.6171779257947 | 1436.6172 | 0.0000220742 |
| Middle | 500 | 36,000,000 | 2932.6670658034504 | 2932.6671 | 0.0000341965 |
| Last | 990 | 71,280,000 | 3092.054468005484 | 3092.0545 | 0.0000319945 |

The differences are only due to the four-decimal formatting in `run.log`; rounding the full-precision values to four decimals gives exact agreement. Machine-readable evidence is in `reports/P3A_RAW_DATA_AUDIT.json`.

## Evidence locations

- Original local run directory: `logs/gym-finetune/hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-09-11_20-11-30_42`.
- Public run log copy: `reports/evidence/fig4_hopper_dppo_seed42/run.log`.
- Public resolved config copy: `reports/evidence/fig4_hopper_dppo_seed42/config.yaml`.
- Public Hydra overrides copy: `reports/evidence/fig4_hopper_dppo_seed42/overrides.yaml`.
- Final checkpoint: local-only under the original run directory; it is not uploaded to GitHub.
