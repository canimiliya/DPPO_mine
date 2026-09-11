# Experiment configuration

The source of truth is the untouched YAML in `source\dppo_v0.6\cfg\gym`.

## Paper-aligned Gym settings

- Tasks: `hopper-medium-v2`, `walker2d-medium-v2`, `halfcheetah-medium-v2`.
- Dataset: official D4RL medium dataset; state input; original dense reward.
- Pretraining: `denoising_steps=20`, `horizon_steps=4`, 3000 epochs, batch 128, learning rate `1e-3` cosine-decayed to `1e-4`, weight decay `1e-6`, EMA `0.995`.
- Figure 4 fine-tuning: action chunk `Ta=4`, denoising horizon `Td=20`, fine-tuning denoising steps `10`, `n_envs=40`, `n_steps=500`, `n_train_itr=1000`, `gamma=0.99`, GAE lambda `0.95`, PPO clip `0.01`, batch size `50000`, update epochs `5`, actor learning rate `1e-4`, critic learning rate `1e-3`.
- Figure 18 from scratch: `horizon_steps=1`, `act_steps=1`, `n_envs=10`, `n_steps=1000`, `n_train_itr=1000`, `gamma=0.99`; DPPO uses batch size `10000` and diffusion denoising, Gaussian-MLP uses batch size `1000`.
- Actor MLP: `[512, 512, 512]`; critic MLP: `[256, 256, 256]` where present; minimum sampling/log-probability standard deviation `0.1`.
- Seed: `42` for the current run; five-seed scripts are prepared but not launched.

## Smoke overrides (not paper results)

The completed smoke runs used `n_train_itr=2`, `n_envs=2`, `n_steps=4`, small batch sizes, and checkpoint saving every iteration. They verify import, environment stepping, forward/loss/backward, optimizer update, and checkpoint writing only. No curve or performance claim is derived from them.
