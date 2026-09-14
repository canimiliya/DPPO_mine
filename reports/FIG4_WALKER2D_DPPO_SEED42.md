# Figure 4 Walker2D DPPO seed=42

## Status

**P6-A FORMAL RUN: PASS**

This is a complete formal Walker2D DPPO single-seed run. It is not a five-seed statistical reproduction and does not complete the full Figure 4 campaign.

## Pre-flight

- Remote base before run: `80829284fdbd50fa3205dfa1483c3beb27217b90`; local `main` and `origin/main` matched.
- Official submodule HEAD: `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`.
- Official checkpoint: `checkpoints/official/walker2d_medium_pre_diffusion_mlp_ta4_td20_state_3000.pt` (4565995 bytes), SHA256 `1a650d5d882c78cfef1d753c532c0bdbced054738b0076efe6ce29d088681fa0`.
- Normalization: `data/gym/walker2d-medium-v2/normalization.npz` (1202 bytes), SHA256 `7d103b793b591081947fca807ae13c2752bd6964621374101f5a89a19346ae16`.
- Official resources: torch.load PASS; normalization finite; observation shape 17; action shape 6.
- Resolved configuration: `/mnt/d/desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/walker2d-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-09-14_21-46-12_42/.hydra/config.yaml`.
- Scientific config changed: **NO**. The formal run used the official Walker2D YAML values: `n_train_itr=1000`, `env.n_envs=40`, `train.n_steps=500`, `batch_size=50000`, `update_epochs=5`, `K=20`, `K'=10`, `horizon_steps=4`, `act_steps=4`, `seed=42`, `device=cuda:0`.

## Run

- Environment: `walker2d-medium-v2` / Walker2D.
- Method: DPPO.
- Seed: 42.
- Iterations completed: 1000 (`itr=0` through `itr=999`).
- Final environment steps: 72,000,000.
- Run-log timestamp span: `2026-09-14 21:46:35,586` to `2026-09-15 01:59:19,504`; observed wall-clock `4.2122 h` (15163.918 s).
- First-20-row estimate: `15.2351 s/row`, projected full run `4.2320 h` (15235.121 s).
- Final local checkpoint: `/mnt/d/desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/walker2d-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-09-14_21-46-12_42/checkpoint/state_999.pt`.

The official `val_freq=10` schedule produced 100 real evaluation points at iterations 0, 10, ..., 990. The CSV retains all 1000 result records, with training reward and evaluation reward kept separate. No interpolation was used.

## Result

- Initial evaluation reward: `2772.202863878287189` at iteration 0 / 0 environment steps.
- Final evaluation reward: `3943.777933631889937` at iteration 990 / 71,280,000 environment steps.
- Best evaluation reward: `3979.244093043152134` at iteration 970 / 69,840,000 environment steps.
- Evaluation points: 100.
- Curve: `plots/fig4_walker2d_dppo_seed42.png`.
- Raw table: `results/fig4_walker2d_dppo_seed42.csv`.

The single-seed evaluation trajectory improves overall from the initial to the final point, with expected noise during training. The cautious qualitative comparison with the paper's Figure 4 Walker2D DPPO trend is that the learning direction is consistent; no precise values were digitized from the paper figure.

## Health and provenance

- NaN/Inf in parsed numeric records: **none detected**.
- NaN/Inf tokens in `run.log`: **none detected**.
- Exception/traceback: **none detected**.
- CUDA OOM: **none detected**.
- Performance patch active: **YES**; the only submodule source modification is the approved denoising-discount CUDA vectorization in `model/diffusion/diffusion_ppo.py`.
- Benchmark instrumentation present in the formal training entrypoint: **NO**.
- Checkpoint integrity: `state_0.pt` through `state_900.pt` and final `state_999.pt` were written; checkpoints remain local-only.

### RAW DATA AUDIT: PASS

The CSV was generated directly from the 1000-record `result.pkl`. The following evaluation points were independently compared with `run.log`; the log displays four decimal places, so the allowed difference is formatting-only.

| Point | Iteration | Environment steps | `result.pkl` | `run.log` display | Absolute difference |
|---|---:|---:|---:|---:|---:|
| First | 0 | 0 | 2772.202863878287189 | 2772.2029 | 0.0000361217 |
| Middle | 500 | 36,000,000 | 3621.591239613764628 | 3621.5912 | 0.0000396138 |
| Last | 990 | 71,280,000 | 3943.777933631889937 | 3943.7779 | 0.0000336319 |

All three checks match after rounding. Machine-readable evidence is in `reports/P6A_RAW_DATA_AUDIT.json`.

## Evidence locations

- Original local run directory: `/mnt/d/desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/walker2d-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-09-14_21-46-12_42`.
- Public run log copy: `reports/evidence/fig4_walker2d_dppo_seed42/run.log`.
- Public resolved config copy: `reports/evidence/fig4_walker2d_dppo_seed42/config.yaml`.
- Public Hydra overrides copy: `reports/evidence/fig4_walker2d_dppo_seed42/overrides.yaml`.
- Final checkpoint: local-only under the original run directory; it is not uploaded to GitHub.
