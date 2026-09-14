# Figure 4 HalfCheetah DPPO seed=42

## Status

**P6-B FORMAL RUN: PASS**

This is a complete formal HalfCheetah DPPO single-seed run. It is not a five-seed statistical reproduction and does not by itself complete the full Figure 4 reproduction.

## Pre-flight

- Remote base before run: `656fac4fbc17c145715371dc1c98d27a0a665dcf` (P6-A Walker2D complete); local `main` and `origin/main` matched.
- Official submodule HEAD: `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`.
- Official checkpoint: `checkpoints/official/halfcheetah_medium_pre_diffusion_mlp_ta4_td20_state_3000.pt` (4565995 bytes), SHA256 `ec1169d29e2d0e006f0c2477ee8bc976b2f9b04b2c209df5bce9f36dd42596b8`.
- Normalization: `data/gym/halfcheetah-medium-v2/normalization.npz` (1202 bytes), SHA256 `1bc71a75b5cde71bb699cdbb4d169de9309a714b87c08bf88bd9c6c10a963e2a`.
- Official resources: torch.load PASS; normalization finite; observation shape 17; action shape 6.
- Resolved configuration: `/mnt/d/desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-09-15_02-53-10_42/.hydra/config.yaml`.
- Scientific config changed: **NO**. The formal run used official values: `n_train_itr=1000`, `env.n_envs=40`, `train.n_steps=500`, `batch_size=50000`, `update_epochs=5`, `K=20`, `K'=10`, `horizon_steps=4`, `act_steps=4`, `seed=42`, `device=cuda:0`.

## Run

- Environment: `halfcheetah-medium-v2` / HalfCheetah.
- Method: DPPO.
- Seed: 42.
- Iterations completed: 1000 (`itr=0` through `itr=999`).
- Final environment steps: 72,000,000.
- Run-log timestamp span: `2026-09-15 02:53:35,135` to `2026-09-15 06:59:57,037`; observed wall-clock `4.1061 h` (14781.902 s).
- First-20-row estimate: `14.8240 s/row`, projected full run `4.1178 h` (14824.003 s).
- Final local checkpoint: `/mnt/d/desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-09-15_02-53-10_42/checkpoint/state_999.pt`.

The official `val_freq=10` schedule produced 100 real evaluation points at iterations 0, 10, ..., 990. The CSV retains all 1000 result records, with training reward and evaluation reward kept separate. No interpolation was used.

## Result

- Initial evaluation reward: `4170.441033292450811` at iteration 0 / 0 environment steps.
- Final evaluation reward: `5133.426464062357809` at iteration 990 / 71,280,000 environment steps.
- Best evaluation reward: `5164.536209644036717` at iteration 980 / 70,560,000 environment steps.
- Evaluation points: 100.
- Curve: `plots/fig4_halfcheetah_dppo_seed42.png`.
- Raw table: `results/fig4_halfcheetah_dppo_seed42.csv`.

The single-seed evaluation trajectory improves overall from the initial to the final point. The cautious qualitative comparison with the paper's Figure 4 HalfCheetah DPPO trend is directional only; no precise values were digitized from the paper figure.

## Health and provenance

- NaN/Inf in parsed numeric records: **none detected**.
- NaN/Inf tokens in `run.log`: **none detected**.
- Exception/traceback: **none detected**.
- CUDA OOM: **none detected**.
- Performance patch active: **YES**; the only submodule source modification is the approved denoising-discount CUDA vectorization in `model/diffusion/diffusion_ppo.py`.
- Benchmark instrumentation present in the formal training entrypoint: **NO**.
- Checkpoint integrity: `state_0.pt` through `state_900.pt` and final `state_999.pt` were written; checkpoints remain local-only.

### RAW DATA AUDIT: PASS

The CSV was generated directly from the 1000-record `result.pkl`. The following evaluation points were independently compared with `run.log`; the log displays four decimal places, so the differences are formatting-only.

| Point | Iteration | Environment steps | `result.pkl` | `run.log` display | Absolute difference |
|---|---:|---:|---:|---:|---:|
| First | 0 | 0 | 4170.441033292450811 | 4170.4410 | 0.0000332925 |
| Middle | 500 | 36,000,000 | 5129.917194332928375 | 5129.9172 | 0.0000056671 |
| Last | 990 | 71,280,000 | 5133.426464062357809 | 5133.4265 | 0.0000359376 |

All three checks match after rounding. Machine-readable evidence is in `reports/P6B_RAW_DATA_AUDIT.json`.

## Evidence locations

- Original local run directory: `/mnt/d/desktop/my_project/paper_reproduction/DPPO/logs/gym-finetune/halfcheetah-medium-v2_ppo_diffusion_mlp_ta4_td20_tdf10/2026-09-15_02-53-10_42`.
- Public run log copy: `reports/evidence/fig4_halfcheetah_dppo_seed42/run.log`.
- Public resolved config copy: `reports/evidence/fig4_halfcheetah_dppo_seed42/config.yaml`.
- Public Hydra overrides copy: `reports/evidence/fig4_halfcheetah_dppo_seed42/overrides.yaml`.
- Final checkpoint: local-only under the original run directory; it is not uploaded to GitHub.
