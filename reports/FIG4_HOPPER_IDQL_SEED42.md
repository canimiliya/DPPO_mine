# Figure 4 Hopper IDQL seed=42

## Status

**P3-B FORMAL RUN: PASS**

The official Hopper IDQL run completed all 1000 configured iterations. This is a single-seed result; it does not complete the three-method or multi-environment Figure 4 campaign.

## Pre-flight

- Remote base before run: `806e91923f50e13fa00117350938aa563a6140c7`; local `main` and `origin/main` matched.
- Official submodule HEAD: `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`.
- IDQL source: official v0.6; benchmark instrumentation present: **NO**.
- DPPO performance patch remained isolated to `model/diffusion/diffusion_ppo.py`.
- Resolved config: `/mnt/d/Desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/hopper-medium-v2_idql_diffusion_mlp_ta4_td20/2026-09-12_11-03-24_42/.hydra/config.yaml`.
- Scientific config changed: **NO**; only the requested seed=42 and local official checkpoint path were supplied by the wrapper.

## Run

- Environment: `hopper-medium-v2` / Hopper.
- Method: IDQL.
- Seed: 42.
- Iterations completed: 1000 (`itr=0` through `itr=999`).
- Final environment steps: 72,000,000.
- Wall-clock log span: `8:49:36.603000` (2026-09-12 11:03:58.935000 to 2026-09-12 19:53:35.538000).
- Local final checkpoint: `/mnt/d/Desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/hopper-medium-v2_idql_diffusion_mlp_ta4_td20/2026-09-12_11-03-24_42/checkpoint/state_999.pt`.

## Result

- Initial evaluation reward: `1348.94679600305` at iteration 0 / 0 environment steps.
- Final evaluation reward: `3150.27635638692` at iteration 990 / 71,280,000 environment steps.
- Best evaluation reward: `3150.27635638692` at iteration 990 / 71,280,000 environment steps.
- Evaluation points: 100.
- Curve: `plots/fig4_hopper_idql_seed42.png`.
- Raw table: `results/fig4_hopper_idql_seed42.csv`.

The curve preserves the raw official evaluation points. Any comparison with the paper's Figure 4 IDQL Hopper curve is qualitative only: this single seed is not sufficient to claim complete Figure 4 reproduction or statistical agreement.

## Health and provenance

- NaN/Inf: **none detected** in run.log/result records.
- Exception: **none detected**.
- CUDA OOM: **none detected**.
- Benchmark instrumentation in IDQL agent: **NO**.
- Shared pretrained checkpoint: `checkpoints/official/hopper_medium_pre_diffusion_mlp_ta4_td20_state_3000.pt`.
- Environment-step convention: `step` is the agent's internal count; each training iteration adds `n_envs * act_steps = 40 * 4 = 160` environment steps per rollout step, hence 80,000 per iteration. The paper's high-level decision-step axis should not be conflated with this internal count.

## RAW DATA AUDIT

**PASS**. The first, middle, and last evaluation points were compared between `result.pkl` and `run.log`; four-decimal log formatting accounts for the small absolute differences.

See `reports/P3B_RAW_DATA_AUDIT.json` for machine-readable evidence.

## Evidence locations

- Original local run: `/mnt/d/Desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/hopper-medium-v2_idql_diffusion_mlp_ta4_td20/2026-09-12_11-03-24_42`.
- Public evidence copies: `reports/evidence/fig4_hopper_idql_seed42/` (`run.log`, `config.yaml`, `overrides.yaml`).
- Final checkpoint is local-only and is not uploaded to GitHub.
