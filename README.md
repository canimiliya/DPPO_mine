# DPPO Gym reproduction workspace

This repository contains the reproducibility code, configuration records, smoke-test evidence, and a short full-configuration throughput benchmark for **Diffusion Policy Policy Optimization (DPPO)** on the requested Gym/MuJoCo tasks.

导师线上审阅请首先阅读：[`导师线上审阅说明.md`](导师线上审阅说明.md)。该文件说明了本地目录、环境、数据/checkpoint、GitHub 上传边界、运行证据和当前尚未完成的正式实验。

## Current scientific status

- Environment setup and CUDA/MuJoCo/D4RL/DPPO smoke tests: **PASS**.
- Hopper Figure 4 DPPO/IDQL/DIPO official-size short benchmark: **ENGINEERING EVIDENCE ONLY**.
- Hopper Figure 4 DPPO seed=42 full run: **COMPLETE** (single-seed only; results and curve are uploaded).
- Hopper Figure 4 DPPO/IDQL/DIPO full runs: **COMPLETE** (single-seed only; results and curves are uploaded).
- Hopper three-method single-seed comparison: **COMPLETE**; Figure 4 three-environment reproduction and Figure 18 campaign: **NOT COMPLETE**.

The repository must not be cited as a completed multi-seed reproduction of Figure 4 or Figure 18. The Hopper DPPO result is one formal seed-42 run; short smoke runs and the 5-iteration benchmarks remain engineering evidence only.

The three formal Hopper seed-42 results are documented in `reports/FIG4_HOPPER_DPPO_SEED42.md`, `reports/FIG4_HOPPER_IDQL_SEED42.md`, and `reports/FIG4_HOPPER_DIPO_SEED42.md`. The unified table and curve are `results/fig4_hopper_three_methods_seed42.csv` and `plots/fig4_hopper_three_methods_seed42.png`; the summary is `reports/FIG4_HOPPER_THREE_METHODS_SEED42.md`.

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
- `logs/`: local Hydra-resolved configs, text logs, result pickles, and checkpoints; formal checkpoints remain local-only.
- `source/dppo_v0.6`: fixed upstream source submodule.
- `plots/` and `results/`: formal Hopper DPPO seed-42 curve and CSV, plus reserved output space for later methods/environments.

See [`reports/GITHUB_RELEASE_CONTENTS.md`](reports/GITHUB_RELEASE_CONTENTS.md) for the exact public/local-only boundary and [`reports/RESOURCE_MANIFEST.md`](reports/RESOURCE_MANIFEST.md) for local paths and sizes.

## Local-only dependencies

The following are intentionally excluded from Git because they are large, reproducible, or user-supplied:

- project virtual environment and CUDA Python packages;
- pip/Torch caches and temporary archives;
- D4RL datasets;
- official pretrained checkpoints;
- the supplied paper PDF and paper screenshots.

Their locations, byte sizes, and hashes are recorded in the resource manifest. Compatibility details are in [`reports/COMPATIBILITY_PATCHES.md`](reports/COMPATIBILITY_PATCHES.md).

## P3-C formal status

- Hopper DPPO seed42 full run: **COMPLETE**.
- Hopper IDQL seed42 full run: **COMPLETE**.
- Hopper DIPO seed42 full run: **COMPLETE**.
- Hopper three-method single-seed comparison: **COMPLETE**.
- Figure 4 three-environment reproduction: **NOT COMPLETE**.
- 5-seed statistical reproduction: **NOT COMPLETE**.

These are single-seed observations and must not be described as a complete multi-seed reproduction.

## P4-PREP engineering status

- ROBOMIMIC Can state environment preparation: **PASS**; robomimic import, exact `robosuite==1.4.1`, official normalization, official `state_5000.pt`, single-environment smoke, DPPO update smoke, and official-size 50-env/6-iteration benchmark are documented in `reports/ROBOMIMIC_CAN_PREP.md`.
- ROBOMIMIC Can DPPO seed42 formal fine-tuning: **COMPLETE** for the single-seed state-observation run; see `reports/FIG4_ROBOMIMIC_CAN_DPPO_SEED42.md` and the uploaded CSV/curves.
- ROBOMIMIC Can IDQL seed42 formal fine-tuning: **COMPLETE**; see `reports/FIG4_ROBOMIMIC_CAN_IDQL_SEED42.md` and the uploaded CSV/curves.
- ROBOMIMIC Can DIPO seed42 formal fine-tuning: **NOT COMPLETE**; it is gated on the P4-B commit and push.
- The current robosuite package is the exact PyPI 1.4.1 release because the current ARISE-Initiative GitHub tag listing did not expose a resolvable v1.4.1 tag; this provenance caveat is recorded in the environment report.

The ROBOMIMIC Can result is a single-seed observation only. Multi-seed ROBOMIMIC
reproduction, the full task suite, and complete Figure 4 reproduction remain
**NOT COMPLETE**.

The Can DPPO and IDQL results are single-seed observations. The Can three-method
comparison remains **NOT COMPLETE** until DIPO is completed and its raw audit,
unified curves, and summary report are committed.


## P3-C formal status

- Hopper DPPO seed42 full run: **COMPLETE**.
- Hopper IDQL seed42 full run: **COMPLETE**.
- Hopper DIPO seed42 full run: **COMPLETE**.
- Hopper three-method single-seed comparison: **COMPLETE**.
- Figure 4 three-environment reproduction: **NOT COMPLETE**.
- 5-seed statistical reproduction: **NOT COMPLETE**.

These are single-seed observations and must not be described as a complete multi-seed reproduction.
