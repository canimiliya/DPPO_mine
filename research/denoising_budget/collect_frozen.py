"""Collect small, frozen-policy diagnostic rollouts.

The collector is intentionally separate from the official training source.
It never calls an optimizer or ``save_model``.  The resulting buffers belong
under D:\\AgentData and are not intended for GitHub.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_frozen_model(task: str, policy: str, device: str = "cuda:0"):
    root = _repo_root()
    source = root / "source" / "dppo_v0.6"
    os.environ.setdefault("DPPO_LOG_DIR", str(root / "logs"))
    os.environ.setdefault("DPPO_DATA_DIR", str(root / "data"))
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
    cfg_path = source / "cfg" / "robomimic" / "finetune" / task / "ft_ppo_diffusion_mlp.yaml"
    cfg = OmegaConf.load(cfg_path)
    bc = root / "logs" / "robomimic-pretrain" / task / (
        f"{task}_pre_diffusion_mlp_ta{'8' if task == 'transport' else '4'}_td20"
    ) / ("2024-07-08_11-18-59" if task == "transport" else "2024-07-10_01-46-16") / "checkpoint" / "state_8000.pt"
    cfg.base_policy_path = str(bc)
    cfg.model.network_path = str(bc)
    cfg.model.device = device
    cfg.name = "s0b-frozen-diagnostic"
    cfg.logdir = "/tmp/s0b-frozen-diagnostic"
    cfg.wandb = None
    OmegaConf.resolve(cfg)
    model = hydra.utils.instantiate(cfg.model)
    if policy == "ft":
        if task == "transport":
            p = Path("/mnt/d/AgentData/DPPO-P5-B/formal_run_20260914_110100/robomimic-finetune/transport_ft_diffusion_mlp_ta8_td20_tdf10/2026-09-14_10-59-02_42/checkpoint/state_100.pt")
        else:
            p = Path("/mnt/d/AgentData/DPPO-P5-A/formal_run_20260914_074500/robomimic-finetune/square_ft_diffusion_mlp_ta4_td20_tdf10/2026-09-14_07-43-43_42/checkpoint/state_200.pt")
        state = torch.load(p, map_location=device, weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        checkpoint_path = str(p)
    else:
        checkpoint_path = str(bc)
    model.eval()
    return model, cfg, checkpoint_path


def collect(task: str, policy: str, output: str, seed: int = 42, device: str = "cuda:0"):
    root = _repo_root()
    source = root / "source" / "dppo_v0.6"
    os.environ.setdefault("MUJOCO_GL", "egl")
    model, cfg, checkpoint_path = load_frozen_model(task, policy, device)
    from env.gym_utils import make_async

    n_envs = 16
    cfg.env.n_envs = n_envs
    cfg.env.name = task
    cfg.robomimic_env_cfg_path = f"cfg/robomimic/env_meta/{task}.json"
    # Keep exactly the official wrappers and semantics; only use 16 workers
    # because this is a bounded diagnostic collection, not formal training.
    venv = make_async(
        task,
        env_type=None,
        num_envs=n_envs,
        asynchronous=True,
        max_episode_steps=int(cfg.env.max_episode_steps),
        wrappers=cfg.env.wrappers,
        robomimic_env_cfg_path=cfg.robomimic_env_cfg_path,
        shape_meta=None,
        use_image_obs=False,
        render=False,
        render_offscreen=False,
        obs_dim=int(cfg.obs_dim),
        action_dim=int(cfg.action_dim),
    )
    boundary_steps = np.array([20, 40, 60, 80], dtype=np.int64)
    horizon = int(cfg.horizon_steps)
    act_dim = int(cfg.action_dim)
    obs_dim = int(cfg.obs_dim)
    n_steps = int(cfg.env.max_episode_steps) // int(cfg.act_steps)
    states, chains, actions, ep_ids, chunk_ids = [], [], [], [], []
    rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
    terminated = np.zeros((n_steps, n_envs), dtype=np.bool_)
    truncated = np.zeros((n_steps, n_envs), dtype=np.bool_)
    firsts = np.zeros((n_steps + 1, n_envs), dtype=np.bool_)
    firsts[0] = True
    try:
        venv.seed([seed + i for i in range(n_envs)])
        obs = venv.reset_arg(options_list=[{"seed": seed + i} for i in range(n_envs)])
        if isinstance(obs, list):
            obs = {k: np.stack([x[k] for x in obs]) for k in obs[0]}
        torch.manual_seed(seed)
        for step in range(n_steps):
            cond = {"state": torch.from_numpy(obs["state"]).float().to(device)}
            with torch.no_grad():
                sample = model(cond=cond, deterministic=False, return_chain=True)
            chain = sample.chains.detach().cpu().numpy()
            traj = sample.trajectories.detach().cpu().numpy()
            assert chain.ndim == 4 and chain.shape[1] == int(cfg.ft_denoising_steps) + 1
            assert tuple(chain.shape[2:]) == (horizon, act_dim)
            if step in set(boundary_steps.tolist()):
                states.append(obs["state"][:, -int(cfg.cond_steps):].copy())
                chains.append(chain.copy())
                actions.append(traj.copy())
                ep_ids.extend(range(n_envs))
                chunk_ids.extend([int(np.where(boundary_steps == step)[0][0])] * n_envs)
            obs, rew, term, trunc, _info = venv.step(traj[:, : int(cfg.act_steps)])
            rewards[step] = rew
            terminated[step] = term
            truncated[step] = trunc
            firsts[step + 1] = np.asarray(term) | np.asarray(trunc)
        states = np.concatenate(states, axis=0)
        chains = np.concatenate(chains, axis=0)
        actions = np.concatenate(actions, axis=0)
        assert states.shape == (n_envs * len(boundary_steps), int(cfg.cond_steps), obs_dim)
        assert chains.shape == (states.shape[0], int(cfg.ft_denoising_steps) + 1, horizon, act_dim)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            normalized_observation=states.astype(np.float32),
            denoising_chain=chains.astype(np.float32),
            executed_action=actions.astype(np.float32),
            raw_reward=rewards,
            terminated=terminated,
            truncated=truncated,
            firsts=firsts,
            episode_id=np.asarray(ep_ids, dtype=np.int64),
            chunk_boundary_id=np.asarray(chunk_ids, dtype=np.int64),
            boundary_steps=boundary_steps,
            task=np.asarray(task),
            policy=np.asarray(policy),
            checkpoint_path=np.asarray(checkpoint_path),
            seed=np.asarray(seed, dtype=np.int64),
            sampling_rule=np.asarray("official VPGDiffusion DDPM forward; deterministic=False"),
        )
        return {"output": str(out), "states": int(states.shape[0]), "chain_shape": list(chains.shape), "primitive_steps": int(n_envs * n_steps * int(cfg.act_steps))}
    finally:
        venv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["square", "transport"], required=True)
    ap.add_argument("--policy", choices=["bc", "ft"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    print(json.dumps(collect(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
