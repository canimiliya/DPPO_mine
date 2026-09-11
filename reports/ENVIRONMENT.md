# Reproduction environment

Verified on 2026-09-08.

| Component | Verified value |
|---|---|
| Host | Windows, WSL2 Ubuntu-24.04 |
| Python | 3.12.3 |
| Virtual environment | `D:\Desktop\my_project\paper_reproduction\DPPO\.runtime\venv` |
| PyTorch | `2.7.1+cu128` |
| PyTorch CUDA runtime | `12.8` |
| GPU | NVIDIA GeForce RTX 5060 Ti, compute capability `(12, 0)` |
| Gym | `0.22.0` |
| D4RL | `1.1` |
| mujoco-py | `2.1.2.14` |
| MuJoCo runtime | `2.1.0`, under `.runtime\mujoco210` |
| Hydra/OmegaConf | `1.3.2` / `2.3.0` |
| NumPy | `1.26.4` |
| Matplotlib | `3.7.5` |

## Environment variables used by official runs

```text
DPPO_DATA_DIR=/mnt/d/Desktop/my_project/paper_reproduction/DPPO/data
DPPO_LOG_DIR=/mnt/d/Desktop/my_project/paper_reproduction/DPPO/logs
MUJOCO_PY_MUJOCO_PATH=/mnt/d/Desktop/my_project/paper_reproduction/DPPO/.runtime/mujoco210
MUJOCO_GL=osmesa
LD_LIBRARY_PATH=/mnt/d/Desktop/my_project/paper_reproduction/DPPO/.runtime/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib
```

The complete file-level inventory, including exact byte sizes and selected SHA256 hashes, is maintained in `RESOURCE_MANIFEST.md` and `RESOURCE_MANIFEST.csv`.
