# ROBOMIMIC Can IDQL, seed=42

## Status

**P4-B: PASS.** The corrected formal run completed `151/151` iterations with
the official IDQL Can state-observation configuration. The result is a single
seed-42 observation and is not a multi-seed or full Figure 4 reproduction.

## Provenance and configuration

- Benchmark/task: ROBOMIMIC Can (`PickPlaceCan`), state/low-dimensional input.
- Algorithm: official v0.6 `ft_idql_diffusion_mlp.yaml`.
- Seed/device: `42` / `cuda:0`.
- Official initialization: Can `state_5000.pt`, SHA256
  `61851045e6b516807826e3bda4270c9e4a086023f2dd85af00eadb59d9b98a1b`.
- Normalization: official Can `normalization.npz`, SHA256
  `A4BB04C498625BFAD0EE6FAAE21674C0A06879CD9F9614BBD2DA41A7D0DC1C1A`.
- Environment: `n_envs=50`, `n_steps=300`, `horizon_steps=4`,
  `act_steps=4`, `denoising_steps=20`, camera/video disabled.
- IDQL-specific settings retained from the YAML: `n_critic_warmup_itr=5`,
  `actor_lr=1e-5`, `critic_lr=1e-3`, `scale_reward_factor=1`,
  `eval_deterministic=true`, `eval_sample_num=10`, `critic_tau=0.001`,
  expectile exploration enabled, `buffer_size=5000`, `replay_ratio=128`,
  and `batch_size=1000`.
- Resolved Hydra config and overrides are preserved in the public evidence
  directory. Only checkpoint, normalization, and external log-directory paths
  were overridden; no smoke or benchmark override was used.

## Completion and timing

- Result rows: `151` (`itr=0..150`).
- Automatic evaluation points: `16` (`itr=0,10,...,150`).
- Final environment steps: `8,100,000` (`2,025,000` high-level decision steps).
- First-ten-row observed time: `406.2021 s`; mean `40.6202 s/row`.
- First-ten projection: `6,133.6515 s` (`1.7038 h`).
- Run-log timestamp span: `6,148.975 s` (`1.7080 h`).

Two pre-start Windows launcher attempts exited before environment construction
because shell argument/path handling did not reach the Python entry point. They
produced zero iterations and are retained outside the scientific result; the
corrected run used the same configuration in a fresh directory.

## Evaluation results

| Metric | Value |
|---|---:|
| Initial success rate | 0.0450 |
| Final success rate | 0.9750 |
| Best success rate | 1.0000 at itr 110 / 5,940,000 environment steps |
| Initial evaluation reward | 5.8850 |
| Final evaluation reward | 174.4150 |
| Best evaluation reward | 179.9700 at itr 90 / 4,860,000 environment steps |

First real evaluation point reaching each success threshold:

| Threshold | Iteration | Environment steps |
|---:|---:|---:|
| 0.80 | 30 | 1,620,000 |
| 0.90 | 30 | 1,620,000 |
| 0.95 | 50 | 2,700,000 |
| 0.99 | 90 | 4,860,000 |

## Health and raw-data audit

- `result.pkl` numeric values: finite.
- `run.log`: no traceback, NaN/Inf, CUDA OOM, worker crash, or exception in
  the corrected formal run.
- 50 environments: created and remained operational through completion.
- Checkpoints: `state_0.pt`, `state_100.pt`, and `state_150.pt` saved locally.
- **RAW DATA AUDIT: PASS.** The first, middle, and final evaluation points
  agree between `result.pkl` and `run.log` for success rate, reward, and
  environment steps; see `reports/P4B_RAW_DATA_AUDIT.json`.

## Public artifacts

- CSV: `results/fig4_robomimic_can_idql_seed42.csv`
- Success curve: `plots/fig4_robomimic_can_idql_seed42_success.png`
- Reward curve: `plots/fig4_robomimic_can_idql_seed42_reward.png`
- Raw audit: `reports/P4B_RAW_DATA_AUDIT.json`
- Evidence: `reports/evidence/fig4_robomimic_can_idql_seed42/`
- Local run: `D:\AgentData\DPPO-P4-B\formal_run_20260914_0100`

The local `result.pkl`, checkpoints, official checkpoint, data, `.runtime`,
and caches are intentionally not uploaded.
