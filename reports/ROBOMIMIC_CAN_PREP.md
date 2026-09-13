# P4-PREP: ROBOMIMIC Can preparation

Date: 2026-09-13

## Overall result

**P4-PREP: PASS — preparation snapshot complete.** The formal Can runs were a
separate subsequent milestone; P4-A DPPO, P4-B IDQL, and P4-C DIPO are now
documented in their respective reports.

The official v0.6 configuration remains the scientific source of truth. The
tested target is Can state observation, DPPO-MLP, seed 42, with `K=20`,
`K'=10`, `horizon_steps=4`, `act_steps=4`, `obs_dim=23`, and `action_dim=7`.

## Environment and resources

The existing WSL `.runtime/venv` was reused. PyTorch 2.7.1+cu128, CUDA 12.8,
Gym 0.22, numpy 1.26.4, and mujoco-py 2.1.2.14 were preserved. The exact
versions are in [ROBOMIMIC_ENVIRONMENT.md](ROBOMIMIC_ENVIRONMENT.md), and
resource hashes are in [ROBOMIMIC_CAN_RESOURCE_AUDIT.md](ROBOMIMIC_CAN_RESOURCE_AUDIT.md).
The package import checks for robomimic and robosuite pass. The required
robosuite version is exactly 1.4.1. Its current GitHub v1.4.1 tag was not
resolvable, so the exact PyPI 1.4.1 release was used and this provenance caveat
is retained for review.

## Official metadata and single-environment smoke

`cfg/robomimic/env_meta/can.json` was verified as `PickPlaceCan`, Panda,
`OSC_POSE`, control frequency 20, object observations enabled, camera
observations disabled, reward shaping disabled, and no renderer/offscreen
renderer. A direct robosuite environment ran 300 zero-action steps with finite
observations and rewards. The DPPO state wrapper then ran 300 environment steps
through 75 four-action chunks:

- vector environment count: 1;
- wrapped observation: `state`, shape `(1, 1, 23)`;
- action space: `(1, 4, 7)`;
- observations and rewards: finite;
- termination/truncation path: exercised normally at the 300-step limit.

No camera image, video recording, or offscreen rendering was enabled.

## DPPO algorithm smoke

The official `ft_ppo_diffusion_mlp.yaml` was used with engineering-only
overrides `n_envs=2`, `n_steps=8`, `n_train_itr=3`, `batch_size=16`, and
`update_epochs=1`. The official state_5000 checkpoint loaded successfully.
The log shows itr 0 evaluation, itr 1 critic warmup, and itr 2 actor PPO
updates. Diffusion actor forward, critic forward, GAE, PPO loss, backward, and
optimizer steps completed with no NaN, Inf, OOM, exception, or device/shape
failure. This smoke reward is not a scientific result.

## Official-size short benchmark

Only `train.n_train_itr=6` was overridden. The tested scale was exactly
`env.n_envs=50`, `train.n_steps=300`, seed 42, with the official denoising,
action, batch, and update parameters. All 50 environments were created and
all six iterations completed; the final log reached `step=300000` and itr 5
completed 10 PPO epochs with 20 batches per epoch.

The normal per-iteration times emitted by the existing timer were:

| Iteration | Mode | Seconds |
|---:|---|---:|
| 0 | evaluation | 30.9 wall-clock (from normal log timestamps) |
| 1 | training / critic warmup | 32.9292 |
| 2 | training | 29.2716 |
| 3 | training | 32.5878 |
| 4 | training | 32.3934 |
| 5 | training | 32.7043 |

Training itr 1–5 summary: mean `31.9773 s`, median `32.5878 s`, minimum
`29.2716 s`, maximum `32.9292 s`.

Independent telemetry used 196 one-second samples and did not modify training
code. Values below are host-level WSL samples during the second identical
official-size run:

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| CPU utilization | 49.92% | 60.95% | 2.35% | 68.71% |
| RAM used | 10,559.5 MiB | 11,797.1 MiB | 858.6 MiB | 11,891.6 MiB |
| RAM visible to WSL | 23,715.7 MiB | 23,715.7 MiB | 23,715.7 MiB | 23,715.7 MiB |
| GPU utilization | 10.85% | 7% | 2% | 78% |
| VRAM used | 3,042 MiB | 3,266 MiB | 2,236 MiB | 3,592 MiB |

There was no process failure, NaN/Inf, OOM, or exception. CPU was busy but not
pathologically oversubscribed, and RAM/VRAM remained well below the visible
limits. The raw evidence is under
`reports/evidence/robomimic_can_prep/`.

## Full-run estimate

For 151 iterations, the official `val_freq=10` schedule implies 16 evaluation
iterations (`0, 10, ..., 150`) and 135 training iterations. Using the measured
values:

`16 × 30.9 + 135 × 31.9773 = 4,811 s = 1.34 h`.

Best estimate: **1.34 h**. Reasonable range: **1.25–1.50 h**, allowing normal
launch, checkpoint, and runtime variation. This is materially faster than the
paper reference of approximately 42 s per training iteration on 50 CPU threads
plus an NVIDIA L40, but the hardware and visible WSL memory/CPU allocation are
different, and this local timing includes the observed implementation/runtime
profile rather than copying the paper number.

## Recommendation and boundary

**Recommended for formal run: YES, after upper-level review of the robosuite
source-provenance caveat.** The preparation gate passed before the separate
formal runs. This report intentionally preserves the preparation-time state
and timing estimate.
