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

## P5-A ROBOMIMIC Square DPPO, seed=42

The official state-observation Square fine-tuning run is complete for one
seed: 201/201 result rows, 21 automatic evaluation points, and final raw
`environment_steps=14,400,000`. Initial, final, and best evaluation success
rates are 0.4400, 0.9950, and 1.0000, respectively; the first best-success
point is iteration 150. The observed run-log wall-clock was 2.6800 h, while
the first-ten-row projection was 2.7767 h. No NaN/Inf, exception, CUDA OOM,
worker failure, or checkpoint-save failure was observed. **RAW DATA AUDIT:
PASS.** See `reports/FIG4_ROBOMIMIC_SQUARE_DPPO_SEED42.md`,
`results/fig4_robomimic_square_dppo_seed42.csv`, and the corresponding plots.

This is a single-seed Square result, not a complete multi-seed or full Figure
4 reproduction. Transport DPPO remains **NOT COMPLETE**; Square/Transport
IDQL and DIPO, new seeds, mechanism experiments, and pixel input experiments
were not started.

## P4-A ROBOMIMIC Can DPPO, seed=42

The official state-observation Can fine-tuning run is complete for one seed:
151/151 result rows, 16 automatic evaluation points, and final raw
`environment_steps=8,100,000`. The initial, final, and best evaluation success
rates are 0.6150, 0.9900, and 1.0000, respectively; the first best-success
point is iteration 120. The actual run-log wall-clock was 1.2838 h, compared
with the P4-PREP estimate of 1.34 h (relative error -4.20%). No NaN/Inf,
exception, CUDA OOM, worker failure, or checkpoint-save failure was observed.

The result is a single-seed trend check, not a complete multi-seed or full-task
Figure 4 reproduction. The raw-data audit is PASS. See
`reports/FIG4_ROBOMIMIC_CAN_DPPO_SEED42.md`,
`results/fig4_robomimic_can_dppo_seed42.csv`, and the corresponding plots.

## P4-B ROBOMIMIC Can IDQL, seed=42

The official IDQL state-observation Can fine-tuning run is complete for one
seed: 151/151 result rows, 16 automatic evaluation points, and final raw
`environment_steps=8,100,000`. Initial/final/best evaluation success rates are
0.0450, 0.9750, and 1.0000, respectively; the first best-success point is
iteration 110. The run-log wall-clock was 1.7080 h, with a first-ten-row
projection of 1.7038 h. No NaN/Inf, exception, CUDA OOM, worker failure, or
checkpoint-save failure was observed in the corrected run.

The result is a single-seed trend check, not a complete multi-seed or full-task
Figure 4 reproduction. **RAW DATA AUDIT: PASS.** See
`reports/FIG4_ROBOMIMIC_CAN_IDQL_SEED42.md`,
`results/fig4_robomimic_can_idql_seed42.csv`, and the corresponding plots.

## P4-C ROBOMIMIC Can DIPO, seed=42

The official DIPO state-observation Can fine-tuning run is complete for one
seed: 300/300 result rows, 30 automatic evaluation points, and final training
`environment_steps=16,200,000`. Initial/final/best evaluation success rates are
0.6150, 0.9400, and 0.9550, respectively; the first best-success point is
iteration 280. The run-log wall-clock was 3.1856 h. No NaN/Inf, exception,
CUDA OOM, or worker failure was observed. **RAW DATA AUDIT: PASS.** See
`reports/FIG4_ROBOMIMIC_CAN_DIPO_SEED42.md`,
`results/fig4_robomimic_can_dipo_seed42.csv`, and the corresponding plots.

## P4-B/C ROBOMIMIC Can three-method summary

The Can DPPO/IDQL/DIPO single-seed comparison is complete. The unified
long-format CSV contains 62 real evaluation rows and uses `environment_steps`
as the main x-axis. First-threshold points, runtime ratios, per-million-step
indicators, and qualitative paper comparison are reported in
`reports/FIG4_ROBOMIMIC_CAN_THREE_METHODS_SEED42.md`. DPPO and IDQL use 151
official iterations; DIPO uses its official 300 iterations. All method-level
raw audits are PASS. This remains a single-seed observation; multi-seed
ROBOMIMIC and complete Figure 4 reproduction remain **NOT COMPLETE**.


## P3-C formal status

- Hopper DPPO seed42: **COMPLETE**; final evaluation reward is recorded in `reports/FIG4_HOPPER_DPPO_SEED42.md`.
- Hopper IDQL seed42: **COMPLETE**; final evaluation reward is recorded in `reports/FIG4_HOPPER_IDQL_SEED42.md`.
- Hopper DIPO seed42: **COMPLETE**; final evaluation reward is recorded in `reports/FIG4_HOPPER_DIPO_SEED42.md`.
- Hopper three-method single-seed comparison: **COMPLETE**; see `reports/FIG4_HOPPER_THREE_METHODS_SEED42.md`.

The combined CSV preserves raw `environment_steps` and adds only the derived `high_level_decision_steps = environment_steps / 4` column. All method comparisons are single-seed observations only. Figure 4 three-environment reproduction, Figure 18 and 5-seed statistical reproduction remain **NOT COMPLETE**.
