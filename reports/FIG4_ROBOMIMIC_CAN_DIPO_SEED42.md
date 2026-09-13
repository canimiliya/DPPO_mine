# ROBOMIMIC Can DIPO, seed=42

## Status

**P4-C: PASS.** The corrected formal run completed the official DIPO
configuration for `300/300` iterations. The result is a single seed-42
observation and is not a multi-seed or full Figure 4 reproduction.

## Provenance and configuration

- Benchmark/task: ROBOMIMIC Can (`PickPlaceCan`), state/low-dimensional input.
- Algorithm: official v0.6 `ft_dipo_diffusion_mlp.yaml`.
- Seed/device: `42` / `cuda:0`.
- Official initialization: the same Can `state_5000.pt` used by DPPO and IDQL,
  SHA256 `61851045e6b516807826e3bda4270c9e4a086023f2dd85af00eadb59d9b98a1b`.
- Normalization: the same official Can `normalization.npz` used by DPPO and
  IDQL, SHA256 `A4BB04C498625BFAD0EE6FAAE21674C0A06879CD9F9614BBD2DA41A7D0DC1C1A`.
- Environment: `n_envs=50`, `n_steps=300`, `horizon_steps=4`,
  `act_steps=4`, `denoising_steps=20`, camera/video disabled.
- DIPO-specific settings retained from the YAML: `n_train_itr=300`,
  `n_critic_warmup_itr=2`, `actor_lr=1e-5`, `critic_lr=1e-3`,
  `scale_reward_factor=1`, `target_ema_rate=0.005`, `buffer_size=1000000`,
  `action_lr=1e-4`, `action_gradient_steps=10`, `replay_ratio=16`, and
  `batch_size=1000`.
- No benchmark instrumentation was added to the DIPO source. Resolved Hydra
  config and overrides are preserved in the public evidence directory.

## Completion, timing, and cost windows

- Result rows: `300` (`itr=0..299`).
- Automatic evaluation points: `30` (`itr=0,10,...,290`).
- Final training environment steps: `16,200,000` (`4,050,000` high-level
  decision steps). The final real evaluation point is itr 290 because the
  official loop ends after training itr 299.
- First-ten-row observed time: `369.0450 s`; mean `36.9045 s/row`.
- First-ten projection: `11,071.3503 s` (`3.0754 h`).
- Run-log timestamp span: `11,467.998 s` (`3.1856 h`).

Normal existing per-iteration timings were used to check the DIPO cost near
the requested windows; no source instrumentation was added:

| Window | Mean seconds/row | Min | Max |
|---|---:|---:|---:|
| itr 1–10 | 36.7563 | 24.8548 | 40.6533 |
| itr 50–59 | 37.8076 | 25.3000 | 41.1930 |
| itr 100–109 | 37.8281 | 25.4663 | 41.4668 |
| itr 200–209 | 38.7040 | 22.8891 | 43.2166 |

The windows show a modest increase but no runaway iteration-cost growth.

## Evaluation results

| Metric | Value |
|---|---:|
| Initial success rate | 0.6150 |
| Final success rate | 0.9400 at itr 290 / 15,660,000 environment steps |
| Best success rate | 0.9550 at itr 280 / 15,120,000 environment steps |
| Initial evaluation reward | 73.4300 |
| Final evaluation reward | 105.8800 |
| Best evaluation reward | 113.9800 at itr 220 / 11,880,000 environment steps |

First real evaluation point reaching each success threshold:

| Threshold | Iteration | Environment steps |
|---:|---:|---:|
| 0.80 | 120 | 6,480,000 |
| 0.90 | 220 | 11,880,000 |
| 0.95 | 230 | 12,420,000 |
| 0.99 | NOT REACHED | NOT REACHED |

## Health and raw-data audit

- `result.pkl` numeric values: finite.
- `run.log`: no traceback, NaN/Inf, CUDA OOM, worker crash, or exception in
  the formal run.
- 50 environments: created and remained operational through completion.
- Checkpoint: `state_0.pt`, `state_100.pt`, `state_200.pt`, and `state_299.pt`
  saved locally.
- **RAW DATA AUDIT: PASS.** The first, middle, and final evaluation points
  agree between `result.pkl` and `run.log` for success rate, reward, and
  environment steps; see `reports/P4C_RAW_DATA_AUDIT.json`.

## Public artifacts

- CSV: `results/fig4_robomimic_can_dipo_seed42.csv`
- Success curve: `plots/fig4_robomimic_can_dipo_seed42_success.png`
- Reward curve: `plots/fig4_robomimic_can_dipo_seed42_reward.png`
- Raw audit: `reports/P4C_RAW_DATA_AUDIT.json`
- Evidence: `reports/evidence/fig4_robomimic_can_dipo_seed42/`
- Local run: `D:\AgentData\DPPO-P4-C\formal_run_20260914_0310`

The local `result.pkl`, checkpoints, official checkpoint, data, `.runtime`,
and caches are intentionally not uploaded.
