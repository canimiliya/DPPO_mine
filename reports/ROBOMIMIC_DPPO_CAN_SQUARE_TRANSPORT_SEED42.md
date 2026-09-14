# ROBOMIMIC DPPO Can / Square / Transport seed-42 overview

All three entries below are official state-observation DPPO runs with seed 42.
This is a compact single-seed overview across tasks, not a strict cross-task
algorithm benchmark: the tasks have different horizons, observation/action
dimensions, action chunks, and environment-step accounting.

| Task | Official iterations | Horizon / act steps | Obs / action dim | Initial success | Final success | Best success | Final env steps | Wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Can | 151 | 4 / 4 | 23 / 7 | 0.6150 | 0.9900 | 1.0000 | 8,100,000 | 1.2838 h |
| Square | 201 | 4 / 4 | 23 / 7 | 0.4400 | 0.9950 | 1.0000 | 14,400,000 | 2.6800 h |
| Transport | 201 | 8 / 8 | 59 / 14 | 0.1800 | 0.9550 | 0.9950 | 28,800,000 | 10.3106 h |

The Can, Square, and Transport raw-data audits are all PASS. Evaluation is
automatic in the original `val_freq=10` training loop. The corresponding
single-task reports, CSV files, and curves retain the actual evaluation points.

This summary does not claim statistical significance, stable superiority, or
complete Figure 4 reproduction. Square and Transport IDQL/DIPO, new seeds,
mechanism/ablation experiments, and pixel experiments remain not started.
