# ROBOMIMIC Can: DPPO vs IDQL vs DIPO, seed=42

## Status

**P4-B/C: COMPLETE.** DPPO, IDQL, and DIPO all completed their official
single-seed Can state-observation runs, and the raw audits and unified outputs
are complete. This is a single-seed comparison, not a multi-seed or complete
paper Figure 4 reproduction.

All three methods use the same official Can `state_5000.pt`, the same official
Can normalization file, `PickPlaceCan`, state observations, `n_envs=50`,
`n_steps=300`, `horizon_steps=4`, `act_steps=4`, `denoising_steps=20`, and
seed 42. IDQL and DIPO use their own official v0.6 YAML configurations; DIPO
retains the official `n_train_itr=300` while DPPO and IDQL use `151`.

## Core comparison

| Method | Official iterations | Final env steps | Initial success | Final success | Best success | Best iteration | Wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| DPPO | 151 | 8,100,000 | 0.6150 | 0.9900 | 1.0000 | 120 | 1.2838 h |
| IDQL | 151 | 8,100,000 | 0.0450 | 0.9750 | 1.0000 | 110 | 1.7080 h |
| DIPO | 300 | 16,200,000 | 0.6150 | 0.9400 | 0.9550 | 280 | 3.1856 h |

The DIPO final success is its last real evaluation point, itr 290, because the
official 300-iteration loop ends after training itr 299. DIPO's best reward was
113.9800 at itr 220; its success rate was not monotonic and declined slightly
from its best point to the final evaluation.

## First real evaluation point reaching each threshold

No interpolation is used. Values are the first actual evaluation point at or
above the threshold, with environment steps as the primary comparison axis.

| Threshold | DPPO | IDQL | DIPO |
|---:|---|---|---|
| 0.80 | itr 10 / 540,000 | itr 30 / 1,620,000 | itr 120 / 6,480,000 |
| 0.90 | itr 20 / 1,080,000 | itr 30 / 1,620,000 | itr 220 / 11,880,000 |
| 0.95 | itr 20 / 1,080,000 | itr 50 / 2,700,000 | itr 230 / 12,420,000 |
| 0.99 | itr 50 / 2,700,000 | itr 90 / 4,860,000 | NOT REACHED |

## Runtime and engineering indicators

| Method | Wall-clock / DPPO | Wall-clock per million env steps |
|---|---:|---:|
| DPPO | 1.0000x | 9.51 min/M |
| IDQL | 1.3304x | 12.65 min/M |
| DIPO | 2.4813x | 11.80 min/M |

The runtime ratios must be read together with the official iteration counts:
DPPO and IDQL run 151 iterations, while DIPO runs 300. The per-million-step
figures are auxiliary engineering indicators, not algorithmic claims.

DIPO's normal existing timer windows (no benchmark instrumentation added) were
36.7563 s/row for itr 1–10, 37.8076 s/row for itr 50–59, 37.8281 s/row for
itr 100–109, and 38.7040 s/row for itr 200–209. This indicates modest cost
growth rather than runaway replay/action-gradient cost.

## Health and audit

- DPPO, IDQL, and DIPO result values are finite.
- No NaN/Inf, exception, CUDA OOM, or worker crash occurred in the corrected
  formal runs.
- All three methods used the same official initialization checkpoint; no
  `state_8000.pt` was used.
- IDQL raw audit: **PASS**, `reports/P4B_RAW_DATA_AUDIT.json`.
- DIPO raw audit: **PASS**, `reports/P4C_RAW_DATA_AUDIT.json`.
- The audits compare first, middle, and final real evaluation points between
  `result.pkl` and `run.log`, including environment steps.

## Paper qualitative comparison

On this Can seed, DPPO rises fastest and reaches high success with the fewest
environment steps. IDQL is competitive after a slower start and reaches a
best success of 1.0. DIPO improves substantially but is weaker and somewhat
less stable near the end, with a best success of 0.955 and no observed 0.99
evaluation point. These are qualitative single-seed observations only; no
precise values are inferred from the paper figure and no population-level
claim is made.

## Public artifacts

- Combined CSV: `results/fig4_robomimic_can_three_methods_seed42.csv`
- Combined success curve: `plots/fig4_robomimic_can_three_methods_seed42_success.png`
- Combined reward curve: `plots/fig4_robomimic_can_three_methods_seed42_reward.png`
- Summary report: this file.
- DPPO report/CSV/curves: `reports/FIG4_ROBOMIMIC_CAN_DPPO_SEED42.md`.
- IDQL report/CSV/curves/evidence: `reports/FIG4_ROBOMIMIC_CAN_IDQL_SEED42.md` and `reports/evidence/fig4_robomimic_can_idql_seed42/`.
- DIPO report/CSV/curves/evidence: `reports/FIG4_ROBOMIMIC_CAN_DIPO_SEED42.md` and `reports/evidence/fig4_robomimic_can_dipo_seed42/`.

The local result pickles, checkpoints, official checkpoint, data, `.runtime`,
and caches remain excluded from GitHub.

## Scope boundary

ROBOMIMIC Can DPPO/IDQL/DIPO seed42 and this three-method single-seed summary
are complete. Square, Transport, new seeds, mechanism experiments, and pixel
input experiments were not started. ROBOMIMIC multi-seed reproduction remains
not complete.
