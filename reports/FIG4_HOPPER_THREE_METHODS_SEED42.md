# Figure 4 Hopper: DPPO / IDQL / DIPO, seed=42

## Status

**Hopper three-method single-seed comparison: COMPLETE**

All three Hopper methods completed their 1000-iteration formal seed-42 runs. This is a single-seed observation; it is not a multi-seed statistical reproduction and does not complete the three-environment Figure 4 campaign.

## Unified comparison

| Method | Initial reward | Final reward | Best reward | Best iteration | Evaluation points | Wall-clock | Stable completion |
|---|---:|---:|---:|---:|---:|---:|---|
| DPPO | 1436.617178 | 3092.054468 | 3092.054468 | 990 | 100 | 4h 09m 03.06s | YES |
| IDQL | 1348.946796 | 3150.276356 | 3150.276356 | 990 | 100 | 8h 49m 36.60s | YES |
| DIPO | 1434.919881 | 2671.399609 | 2938.961842 | 180 | 100 | 20h 59m 46.33s | YES |

- DPPO / IDQL / DIPO use the same Hopper environment, seed=42, and shared pretrained diffusion-policy checkpoint; each method retains its own official Figure 4 configuration.
- `environment_steps` is preserved from each raw CSV. It includes primitive actions emitted by Ta=4 action chunks; the derived `high_level_decision_steps` column is `environment_steps / 4`.
- The plotted curves use only real evaluation points; no interpolation or train-reward substitution was performed.

## Runtime ratios

- IDQL / DPPO wall-clock ratio: **2.127x**.
- DIPO / DPPO wall-clock ratio: **5.058x**.

## Scientific interpretation

In this seed=42 Hopper run, the three methods all show an overall learning/improvement direction, with the listed final rewards and runtimes. These are single-seed observations only; the data do not justify claims of statistical significance, universal superiority, stable superiority, or complete reproduction of the paper's conclusions.

## Files

- Combined raw table: `results/fig4_hopper_three_methods_seed42.csv`.
- Combined curve: `plots/fig4_hopper_three_methods_seed42.png`.
- Individual reports and curves remain available for DPPO, IDQL and DIPO.
