# Figure 4 Hopper DIPO seed=42

## Status

**P3-C FORMAL RUN: PASS**

The official Hopper DIPO run completed all 1000 configured iterations. This is a single-seed result and is not a complete multi-seed or three-environment Figure 4 reproduction.

## Pre-flight

- Remote base before run: `42ea08bce55982db225cfc15b705bff65f7192f9`; local `main` and `origin/main` matched.
- Official submodule HEAD: `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`.
- DIPO source: official v0.6; benchmark instrumentation present: **NO**.
- DPPO performance patch remained isolated to `model/diffusion/diffusion_ppo.py`.
- Resolved config: `/mnt/d/Desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/hopper-medium-v2_dipo_diffusion_mlp_ta4_td20/2026-09-12_21-32-02_42/.hydra/config.yaml`.
- Scientific config changed: **NO**; only seed=42 and the local shared checkpoint path were supplied by the wrapper.

## Run

- Environment: `hopper-medium-v2` / Hopper.
- Method: DIPO.
- Seed: 42.
- Iterations completed: 1000 (`itr=0` through `itr=999`).
- Final environment steps: 72,000,000.
- Wall-clock log span: `20:59:46.328000` (2026-09-12 21:32:38.673000 to 2026-09-13 18:32:25.001000).
- Local final checkpoint: `/mnt/d/Desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/hopper-medium-v2_dipo_diffusion_mlp_ta4_td20/2026-09-12_21-32-02_42/checkpoint/state_999.pt`.

## Result

- Initial evaluation reward: `1434.91988092463` at iteration 0 / 0 environment steps.
- Final evaluation reward: `2671.39960911151` at iteration 990 / 71,280,000 environment steps.
- Best evaluation reward: `2938.96184236001` at iteration 180 / 12,960,000 environment steps.
- Evaluation points: 100.
- Curve: `plots/fig4_hopper_dipo_seed42.png`.
- Raw table: `results/fig4_hopper_dipo_seed42.csv`.

The single-seed curve is compared with the paper's DIPO Hopper trend only qualitatively. It does not support a claim that Figure 4 has been completely reproduced.

## Health and provenance

- NaN/Inf: **none detected** in run.log/result records.
- Exception: **none detected**.
- CUDA OOM: **none detected**.
- Benchmark instrumentation in DIPO agent: **NO**.
- Shared pretrained checkpoint: `checkpoints/official/hopper_medium_pre_diffusion_mlp_ta4_td20_state_3000.pt`.
- DIPO-specific official configuration was retained: replay ratio 64, critic warmup 5, action gradient steps 10, actor/critic learning rates, reward scaling, target EMA and network sizes from `ft_dipo_diffusion_mlp.yaml`.
- Environment-step convention: current internal `environment_steps` includes primitive actions emitted by the action chunk; with Ta=4, high-level decision steps are approximately `environment_steps / 4`.

## RAW DATA AUDIT

**PASS**. The first, middle, and last evaluation points were compared between `result.pkl` and `run.log`; four-decimal log formatting accounts for the small differences.

See `reports/P3C_RAW_DATA_AUDIT.json` for machine-readable evidence.

## Evidence locations

- Original local run: `/mnt/d/Desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/hopper-medium-v2_dipo_diffusion_mlp_ta4_td20/2026-09-12_21-32-02_42`.
- Public evidence copies: `reports/evidence/fig4_hopper_dipo_seed42/` (`run.log`, `config.yaml`, `overrides.yaml`).
- Final checkpoint is local-only and is not uploaded to GitHub.
