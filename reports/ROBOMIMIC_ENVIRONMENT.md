# ROBOMIMIC Can environment audit

Date: 2026-09-13

## Runtime

The existing WSL runtime was reused at `.runtime/venv`; it was not deleted or
recreated. The stable Gym runtime remained intact.

| Component | Observed value |
|---|---|
| WSL | Ubuntu 24.04 |
| Python | 3.12.3 |
| PyTorch | 2.7.1+cu128 |
| CUDA | 12.8; CUDA available; NVIDIA GeForce RTX 5060 Ti |
| Gym | 0.22.0 |
| numpy | 1.26.4 |
| mujoco-py | 2.1.2.14 |
| MuJoCo library used by the DPPO/robomimic path | existing `.runtime/mujoco210` |
| robomimic | 0.5.0, source commit `d309eaecc18acf4152a830a895a6984b8ac71b05` |
| robosuite | 1.4.1 |
| Additional Python MuJoCo package | 3.1.6, installed for robosuite package compatibility; the tested robomimic path still imports `mujoco_py` |

## Installation decision

The official v0.6 optional dependency declares unconstrained `torch`,
`torchvision`, and other dependencies, while this machine has an already
validated `torch 2.7.1+cu128` runtime. Therefore the full optional install was
not run. The following minimal additions were installed with `--no-deps`, so
pip could not downgrade or replace the protected core packages:

- official robomimic source tarball at commit `d309eae...`;
- PyPI `robosuite==1.4.1`;
- `opencv-python-headless==4.10.0.84`, `mujoco==3.1.6`, `PyOpenGL==3.1.7`,
  `absl-py==2.1.0`, `etils==1.9.2`;
- `llvmlite==0.43.0`, `numba==0.60.0`, `scipy==1.14.1`.

The pyproject Git URL for `robosuite@v1.4.1` was also checked. The current
ARISE-Initiative GitHub tag listing exposes v1.4.0 and v1.5.x but no resolvable
v1.4.1 tag, while the exact 1.4.1 PyPI release exists. The tested package is
therefore pinned to the required version, but its provenance is explicitly
recorded as PyPI fallback rather than being misreported as a Git tag checkout.

The WSL pip cache is `/root/.cache/pip`; downloaded temporary build content was
kept outside the repository. The existing core versions were rechecked after
installation and remained unchanged. `pip check` still reports the expected
project metadata conflict (`dppo` declares torch 2.4.0) plus unused
pretraining/offline dependencies not installed by this minimal state-only
runtime. Those packages are not needed by the verified Can state rollout and
DPPO smoke.

## Result

`robomimic` and `robosuite` imports pass. The official Can metadata constructs
successfully with the existing mujoco-py 2.1 backend and headless OSMesa.
