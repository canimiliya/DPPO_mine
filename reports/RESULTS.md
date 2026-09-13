# Results status

## Hopper Figure 4 DPPO, seed=42

The requested single-seed formal run is complete: 1000/1000 iterations were executed with the official Hopper configuration and the clean CUDA performance patch. It is a single-seed result and must not be described as a complete multi-seed Figure 4 reproduction.

- Final environment steps: 72,000,000.
- Evaluation points: 100 (iteration 0 through 990, every 10 iterations).
- Initial evaluation reward: 1436.6171779257947.
- Final evaluation reward: 3092.054468005484 at iteration 990 / 71,280,000 environment steps.
- Best evaluation reward: 3092.054468005484 at iteration 990 / 71,280,000 environment steps.
- Run-log timestamp span: 4:09:03.064; the run directory was created at 20:11:29 and the final log/checkpoint write completed at 00:21:05.
- Health: no NaN/Inf, exception, or CUDA OOM detected in `run.log`.
- RAW DATA AUDIT: PASS; first, middle, and final evaluation points agree between `result.pkl` and the four-decimal values printed in `run.log`.

Artifacts:

- CSV: `results/fig4_hopper_dppo_seed42.csv`
- Curve: `plots/fig4_hopper_dppo_seed42.png`
- Detailed report: `reports/FIG4_HOPPER_DPPO_SEED42.md`
- Raw audit: `reports/P3A_RAW_DATA_AUDIT.json`
- Public raw evidence: `reports/evidence/fig4_hopper_dppo_seed42/`
- Original local run: `logs/gym-finetune/hopper-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-09-11_20-11-30_42`

The curve shows a rising but noisy single-seed Hopper evaluation trajectory. It is qualitatively compatible with a learning/improvement trend, but the paper reports multi-seed statistics; therefore this result is only a single-seed trend check, not a claim that Figure 4 has been fully reproduced.

## P3-C formal status

- Hopper DPPO seed42: **COMPLETE**; final evaluation reward is recorded in `reports/FIG4_HOPPER_DPPO_SEED42.md`.
- Hopper IDQL seed42: **COMPLETE**; final evaluation reward is recorded in `reports/FIG4_HOPPER_IDQL_SEED42.md`.
- Hopper DIPO seed42: **COMPLETE**; final evaluation reward is recorded in `reports/FIG4_HOPPER_DIPO_SEED42.md`.
- Hopper three-method single-seed comparison: **COMPLETE**; see `reports/FIG4_HOPPER_THREE_METHODS_SEED42.md`.

The combined CSV preserves raw `environment_steps` and adds only the derived `high_level_decision_steps = environment_steps / 4` column. All method comparisons are single-seed observations only. Figure 4 three-environment reproduction, Figure 18 and 5-seed statistical reproduction remain **NOT COMPLETE**.


## P3-C formal status

- Hopper DPPO seed42: **COMPLETE**; final evaluation reward is recorded in `reports/FIG4_HOPPER_DPPO_SEED42.md`.
- Hopper IDQL seed42: **COMPLETE**; final evaluation reward is recorded in `reports/FIG4_HOPPER_IDQL_SEED42.md`.
- Hopper DIPO seed42: **COMPLETE**; final evaluation reward is recorded in `reports/FIG4_HOPPER_DIPO_SEED42.md`.
- Hopper three-method single-seed comparison: **COMPLETE**; see `reports/FIG4_HOPPER_THREE_METHODS_SEED42.md`.

The combined CSV preserves raw `environment_steps` and adds only the derived `high_level_decision_steps = environment_steps / 4` column. All method comparisons are single-seed observations only. Figure 4 three-environment reproduction, Figure 18 and 5-seed statistical reproduction remain **NOT COMPLETE**.

