# GitHub publication boundary

Target repository: `https://github.com/canimiliya/DPPO_mine.git`

Publication date: 2026-09-11.

## Included

- Reproduction scripts under `scripts/`.
- Environment, provenance, compatibility, configuration, run, status, and resource reports under `reports/`.
- Text logs, Hydra-resolved configurations, and result pickles from completed smoke/benchmark runs.
- Key locally produced checkpoints:
  - Hopper Figure 4 DPPO benchmark `state_4.pt` (4,977,987 bytes).
  - Hopper Figure 18 Gaussian-MLP smoke `state_1.pt` (4,831,875 bytes).
- Official DPPO v0.6 source as a submodule fixed to commit `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`.

## Intentionally local-only

| Local path | Approximate size | Reason |
|---|---:|---|
| `.runtime/` | 6.71 GiB | WSL virtual environment, CUDA libraries, MuJoCo runtime and helper sources |
| `.cache/` | 3.79 GiB | Re-downloadable package cache |
| `.tmp/` | 4.18 MiB | Retained download archive |
| `data/` | 233.86 MiB | Downloaded D4RL datasets; hashes and shapes are recorded locally |
| `checkpoints/official/` | 12.95 MiB | Upstream pretrained dependencies; hashes are recorded locally |
| `DPPO.pdf` | 17.68 MiB | User-supplied paper reference |
| `图4.png` and `图18.png` | 555,444 bytes total | User-supplied paper screenshots |

## Scientific boundary

The uploaded results are smoke tests and a five-iteration throughput benchmark. They are not the requested complete single-seed Figure 4/Figure 18 experiments. No final reproduction PNG or performance claim is included.
