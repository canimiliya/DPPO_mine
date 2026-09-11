# DPPO reproduction provenance

Status: setup and smoke-test phase complete; formal seed-42 campaigns are not yet complete.

## Official source

- Repository: https://github.com/irom-princeton/dppo
- Release/tag: `v0.6`
- Local checkout: `D:\Desktop\my_project\paper_reproduction\DPPO\source\dppo_v0.6`
- Commit: `dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc`
- Commit timestamp: `2024-10-30T19:58:06-04:00`
- Local source modifications: none (`git status --short` was empty at report creation).

The local checkout is retained as a provenance-preserving copy. Compatibility is handled by the external runtime and wrapper scripts; the official algorithm source and YAML definitions are not silently rewritten.

## Scope frozen for this round

- Gym/MuJoCo only: Hopper-v2, Walker2D-v2, HalfCheetah-v2.
- Figure 4 top row: DRWR, DAWR, DIPO, IDQL, DQL, QSM, DPPO.
- Figure 18 from scratch: DPPO and Gaussian-MLP.
- Current formal seed: `42` only.
- No Robomimic, D3IL, Kitchen, Furniture, Isaac, quadrotor, or new algorithm implementation.
