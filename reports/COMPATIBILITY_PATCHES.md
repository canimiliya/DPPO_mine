# Compatibility record

These are environment-level compatibility decisions, not changes to the scientific method.

| Item | Official / paper reference | Local choice | Effect |
|---|---|---|---|
| Python | `>=3.8` | WSL Ubuntu 24.04, Python 3.12.3, project venv | Runtime only |
| PyTorch | official `2.4.0` | `2.7.1+cu128` | Selected because the RTX 5060 Ti reports compute capability 12.0; no algorithm/config source change |
| CUDA | official package unspecified in project metadata | PyTorch CUDA runtime 12.8 | GPU runtime only |
| MuJoCo | `mujoco-py==2.1.2.14` | official MuJoCo 2.1.0 Linux runtime in project path | Required by the legacy Gym/D4RL stack |
| Rendering | legacy MuJoCo build | `MUJOCO_GL=osmesa`, system `libosmesa6-dev`, GLFW and GLEW | Headless smoke/training compatibility |
| D4RL download | official `gdown` path | Windows-side gdown helper under `.runtime\gdown_tools` | WSL could not reach Google Drive; downloaded files and hashes were verified locally |
| `mjrl` | D4RL dependency | editable third-party checkout at `.runtime\mjrl_source`, commit `3871d93763d3b49c4741e6daeaebbc605fe140dc` | Dependency only |

The launcher interprets `DPPO_DATA_DIR` as the data root and appends `gym`. The wrappers therefore export the project `data` directory, not `data\gym`.

An interrupted first benchmark created the empty directory `data\gym\gym\hopper-medium-v2` while this variable semantics was being corrected. It contains no downloaded files and is retained in the manifest as an accidental empty diagnostic directory; existing valid data under `data\gym\hopper-medium-v2` is untouched.

## Scientific invariants kept

- Official v0.6 YAML configurations are used as the base.
- Environment names, original dense rewards, dataset normalization, action chunking, denoising steps, network widths, PPO/critic settings, and seed semantics are not translated or relaxed.
- Figure 4 uses the official pretrained diffusion checkpoints and the official fine-tuning method configurations.
- Figure 18 uses the official from-scratch DPPO and Gaussian-MLP configurations.

## Out-of-scope optional packages

Optional D4RL integrations for Flow, Kitchen/dm_control, CARLA, and GymBullet were not installed. Their import warnings do not affect the three requested Gym locomotion tasks.
