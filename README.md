# DPPO Gym reproduction workspace

This repository contains the reproducibility code, configuration records, smoke-test evidence, and a short full-configuration throughput benchmark for **Diffusion Policy Policy Optimization (DPPO)** on the requested Gym/MuJoCo tasks.

导师线上审阅请首先阅读：[`导师线上审阅说明.md`](导师线上审阅说明.md)。该文件说明了本地目录、环境、数据/checkpoint、GitHub 上传边界、运行证据和当前尚未完成的正式实验。

## Current scientific status

- Environment setup and CUDA/MuJoCo/D4RL/DPPO smoke tests: **PASS**.
- Hopper Figure 4 DPPO/IDQL/DIPO official-size short benchmark: **ENGINEERING EVIDENCE ONLY**.
- Full seed-42 Figure 4 and Figure 18 campaigns: **NOT COMPLETE**.
- Final reproduction curves and side-by-side figures: **NOT GENERATED**.

The repository must not be cited as a completed reproduction of Figure 4 or Figure 18. Short smoke runs and the 5-iteration benchmark are engineering evidence only.

The three-method Hopper benchmark is documented in
`reports/FIG4_HOPPER_METHOD_BENCHMARK.md`; it reports runtime and health only,
not final reward or convergence claims.

The official source remains fixed at the upstream DPPO v0.6 commit shown below. The
local performance run additionally applies one mathematically equivalent CUDA
vectorization patch, recorded in `reports/PERFORMANCE_PATCH.diff` and applied by
`scripts/setup/apply_performance_patch.ps1` (or `.sh`). The formal patch now contains
only the denoising-discount vectorization in `diffusion_ppo.py`; an older mixed patch
had accidentally included benchmark timing instrumentation and was corrected in P3-PREP.
The timing history is preserved separately in
`reports/DPPO_BENCHMARK_TIMING_INSTRUMENTATION.diff` and
`reports/BENCHMARK_TIMING_INSTRUMENTATION.diff`. Neither timing archive is used for
formal scientific training. The patch does not change the DPPO mathematical definition
or paper hyperparameters.

## Source provenance

The upstream implementation is tracked as a Git submodule at `source/dppo_v0.6`, fixed to the official `v0.6` commit:

```text
dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc
```

Clone with:

```bash
git clone --recurse-submodules https://github.com/canimiliya/DPPO_mine.git
```

## Repository contents

- `scripts/`: one-seed and five-seed launchers, plotting utilities, and resource inventory tooling.
- `reports/`: environment, provenance, compatibility, experiment definitions, run evidence, result status, and local resource manifest.
- `logs/`: Hydra-resolved configs, text logs, small result pickles, and two selected key checkpoints.
- `source/dppo_v0.6`: fixed upstream source submodule.
- `plots/` and `results/`: reserved output directories; currently empty because formal training is unfinished.

See [`reports/GITHUB_RELEASE_CONTENTS.md`](reports/GITHUB_RELEASE_CONTENTS.md) for the exact public/local-only boundary and [`reports/RESOURCE_MANIFEST.md`](reports/RESOURCE_MANIFEST.md) for local paths and sizes.

## Local-only dependencies

The following are intentionally excluded from Git because they are large, reproducible, or user-supplied:

- project virtual environment and CUDA Python packages;
- pip/Torch caches and temporary archives;
- D4RL datasets;
- official pretrained checkpoints;
- the supplied paper PDF and paper screenshots.

Their locations, byte sizes, and hashes are recorded in the resource manifest. Compatibility details are in [`reports/COMPATIBILITY_PATCHES.md`](reports/COMPATIBILITY_PATCHES.md).
