# GitHub publication boundary

Target repository: `https://github.com/canimiliya/DPPO_mine.git`

Publication date: 2026-09-11.

## Included

- Reproduction scripts under `scripts/`.
- Environment, provenance, compatibility, configuration, run, status, and resource reports under `reports/`.
- Text logs, Hydra-resolved configurations, and result pickles from completed smoke/benchmark runs.
- Hopper DPPO/IDQL/DIPO official-scale 5-iteration engineering benchmark reports,
  telemetry CSVs, raw stdout, and resolved configs.
- Key locally produced checkpoints:
  - Hopper Figure 4 DPPO benchmark `state_4.pt` (4,977,987 bytes).
  - Hopper Figure 18 Gaussian-MLP smoke `state_1.pt` (4,831,875 bytes).
- Official DPPO v0.6 source as a submodule fixed to commit `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`.
- ROBOMIMIC Can DPPO/IDQL/DIPO seed-42 formal reports, raw audits, CSVs,
  success/reward curves, unified comparison CSV/curves/report, and the
  corresponding `run.log`, resolved `config.yaml`, and `overrides.yaml` evidence.

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

The newly generated official-scale formal checkpoints and result pickles remain
local-only under their run directories; only text logs, resolved configs, raw
audits, derived CSVs, and plots are uploaded. Historical engineering benchmark
result pickles already present in the repository predate P4-B/C and are not
formal Can result artifacts.

## Scientific boundary

The uploaded Can results are complete only for the single seed-42 DPPO/IDQL/DIPO
comparison. Square, Transport, new seeds, mechanism experiments, pixel input,
and multi-seed/full Figure 4 reproduction remain outside scope.
