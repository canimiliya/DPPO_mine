"""Final read-only dependency, GPU, resource, and Gym smoke audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import gym
import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    import d4rl
    import mujoco_py

    env_results = []
    for env_name in ("hopper-medium-v2", "walker2d-medium-v2", "halfcheetah-medium-v2"):
        env = gym.make(env_name)
        obs = env.reset()
        action = env.action_space.sample()
        step_result = env.step(action)
        env.close()
        observation = np.asarray(obs)
        env_results.append(
            {
                "env": env_name,
                "reset_shape": list(observation.shape),
                "reset_finite": bool(np.isfinite(observation).all()),
                "step_reward_finite": bool(np.isfinite(float(step_result[1]))),
                "step_observation_finite": bool(np.isfinite(np.asarray(step_result[0])).all()),
            }
        )

    required = {}
    for env_name in ("hopper", "walker2d", "halfcheetah"):
        data_dir = root / "data" / "gym" / f"{env_name}-medium-v2"
        data_path = data_dir / "train.npz"
        norm_path = data_dir / "normalization.npz"
        checkpoint = root / "checkpoints" / "official" / f"{env_name}_medium_pre_diffusion_mlp_ta4_td20_state_3000.pt"
        required[env_name] = {
            "train_npz": str(data_path.resolve()),
            "train_npz_bytes": data_path.stat().st_size,
            "train_npz_sha256": sha256(data_path),
            "normalization_bytes": norm_path.stat().st_size,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256(checkpoint),
        }

    result = {
        "device": str(torch.device("cuda:0")),
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "gym": gym.__version__,
        "d4rl": importlib.metadata.version("d4rl"),
        "mujoco_py": importlib.metadata.version("mujoco-py"),
        "mujoco_path": os.environ.get("MUJOCO_PY_MUJOCO_PATH"),
        "env_results": env_results,
        "required_resources": required,
        "all_env_smokes_pass": all(
            item["reset_finite"] and item["step_reward_finite"] and item["step_observation_finite"]
            for item in env_results
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
