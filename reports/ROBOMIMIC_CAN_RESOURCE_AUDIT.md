# ROBOMIMIC Can resource audit

Date: 2026-09-13

## Official sources

The normalization URL and checkpoint URL were taken from
`source/dppo_v0.6/script/download_url.py`, which is the v0.6 project download
mechanism. No locally trained substitute was used.

| Resource | Path | Size | SHA256 | Validation |
|---|---|---:|---|---|
| Can state normalization | `data/robomimic/can/normalization.npz` | 1498 bytes | `a4bb04c498625bfad0ee6faae21674c0a06879cd9f9614bbd2da41a7d0dc1c1a` | `np.load` succeeds; fields `obs_min/max (23,)`, `action_min/max (7,)`; all finite |
| Official Can pre-trained Diffusion Policy | `logs/robomimic-pretrain/can/can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt` | 4,613,419 bytes | `61851045e6b516807826e3bda4270c9e4a086023f2dd85af00eadb59d9b98a1b` | `torch.load(..., map_location=cpu, weights_only=False)` succeeds; keys `epoch`, `model`, `ema`; tensor values finite |

The configuration's `base_policy_path` was checked against this exact
`state_5000.pt` path. `state_8000.pt` was not used. The checkpoint is local-only
and remains excluded by `.gitignore`.
