"""Read-only S0a asset and semantic inventory for the DPPO ROBOMIMIC runs.

The script does not create environments, run training, alter checkpoints, or
modify scientific configuration.  It inventories the selected local assets,
checkpoint state dictionaries, normalization files, runtime versions, and the
installed robosuite Transport success semantics.

Run from the WSL runtime used by this project, for example::

    MUJOCO_PY_MUJOCO_PATH=/mnt/d/Desktop/my_project/paper_reproduction/DPPO/.runtime/mujoco210 \
    MUJOCO_GL=egl \
    LD_LIBRARY_PATH=/mnt/d/Desktop/my_project/paper_reproduction/DPPO/.runtime/mujoco210/bin \
    .runtime/venv/bin/python research/denoising_budget/inventory.py

The output is JSON and is intended to be committed as an audit manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def external_path(linux_path: str, windows_path: str) -> Path:
    """Choose the existing WSL or Windows spelling of an external asset."""
    candidates = (Path(linux_path), Path(windows_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


TRANSPORT_RUN = external_path(
    "/mnt/d/AgentData/DPPO-P5-B/formal_run_20260914_110100/robomimic-finetune/"
    "transport_ft_diffusion_mlp_ta8_td20_tdf10/2026-09-14_10-59-02_42",
    "D:/AgentData/DPPO-P5-B/formal_run_20260914_110100/robomimic-finetune/"
    "transport_ft_diffusion_mlp_ta8_td20_tdf10/2026-09-14_10-59-02_42",
)
SQUARE_RUN = external_path(
    "/mnt/d/AgentData/DPPO-P5-A/formal_run_20260914_074500/robomimic-finetune/"
    "square_ft_diffusion_mlp_ta4_td20_tdf10/2026-09-14_07-43-43_42",
    "D:/AgentData/DPPO-P5-A/formal_run_20260914_074500/robomimic-finetune/"
    "square_ft_diffusion_mlp_ta4_td20_tdf10/2026-09-14_07-43-43_42",
)
TRANSPORT_BC = external_path(
    "/mnt/d/Desktop/my_project/paper_reproduction/DPPO/logs/robomimic-pretrain/"
    "transport/transport_pre_diffusion_mlp_ta8_td20/2024-07-08_11-18-59/"
    "checkpoint/state_8000.pt",
    "D:/Desktop/my_project/paper_reproduction/DPPO/logs/robomimic-pretrain/"
    "transport/transport_pre_diffusion_mlp_ta8_td20/2024-07-08_11-18-59/"
    "checkpoint/state_8000.pt",
)
SQUARE_BC = external_path(
    "/mnt/d/Desktop/my_project/paper_reproduction/DPPO/logs/robomimic-pretrain/"
    "square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/"
    "checkpoint/state_8000.pt",
    "D:/Desktop/my_project/paper_reproduction/DPPO/logs/robomimic-pretrain/"
    "square/square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/"
    "checkpoint/state_8000.pt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str) -> dict[str, Any]:
    record: dict[str, Any] = {"role": role, "path": str(path)}
    if not path.exists():
        record.update({"status": "MISSING", "bytes": None, "sha256": None})
        return record
    record.update(
        {
            "status": "PRESENT",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )
    return record


def shape(value: Any) -> list[int] | str:
    return list(value.shape) if hasattr(value, "shape") else type(value).__name__


def max_abs_diff(left: Any, right: Any) -> float:
    import torch

    return float(torch.max(torch.abs(left - right)).item())


def checkpoint_record(path: Path, torch_module: Any) -> dict[str, Any]:
    record = file_record(path, "checkpoint")
    if record["status"] == "MISSING":
        return record
    data = torch_module.load(path, map_location="cpu", weights_only=True)
    record["top_level_keys"] = list(data.keys())
    record["itr"] = data.get("itr", data.get("epoch"))
    model = data.get("model")
    if not isinstance(model, dict):
        if isinstance(data.get("ema"), dict):
            record["ema_key_count"] = len(data["ema"])
            record["ema_prefixes"] = sorted({key.split(".", 1)[0] for key in data["ema"]})
        return record
    keys = list(model.keys())
    prefixes = ["network.", "actor.", "actor_ft.", "critic.", "ema."]
    record["model_key_count"] = len(keys)
    record["prefix_counts"] = {prefix: sum(key.startswith(prefix) for key in keys) for prefix in prefixes}
    record["first_keys"] = keys[:20]
    record["actor_shapes"] = {key: shape(model[key]) for key in keys if key.startswith("actor.")}
    record["actor_ft_shapes"] = {key: shape(model[key]) for key in keys if key.startswith("actor_ft.")}
    record["critic_keys"] = {key: shape(model[key]) for key in keys if key.startswith("critic.")}
    record["parameter_counts"] = {
        prefix.rstrip("."): int(sum(model[key].numel() for key in keys if key.startswith(prefix)))
        for prefix in ("network.", "actor.", "actor_ft.", "critic.")
    }
    paired = [
        key
        for key in keys
        if key.startswith("actor.") and key.replace("actor.", "actor_ft.", 1) in model
    ]
    if paired:
        diffs = [max_abs_diff(model[key], model[key.replace("actor.", "actor_ft.", 1)]) for key in paired]
        record["actor_vs_actor_ft"] = {
            "matched_keys": len(paired),
            "exact": all(value == 0.0 for value in diffs),
            "max_abs_diff": max(diffs),
        }
    record["resume_fields_present"] = {
        name: name in data
        for name in ("optimizer", "actor_optimizer", "critic_optimizer", "scheduler", "rng", "numpy_rng", "reward_scaler")
    }
    return record


def state0_bc_compare(state0_path: Path, bc_path: Path, torch_module: Any) -> dict[str, Any]:
    if not state0_path.exists() or not bc_path.exists():
        return {"status": "MISSING_INPUT"}
    state0 = torch_module.load(state0_path, map_location="cpu", weights_only=True)["model"]
    bc = torch_module.load(bc_path, map_location="cpu", weights_only=True)
    result: dict[str, Any] = {}
    for candidate in ("ema", "model"):
        bc_state = bc.get(candidate)
        if not isinstance(bc_state, dict):
            continue
        diffs: list[float] = []
        missing: list[str] = []
        for key, value in state0.items():
            if not key.startswith("actor."):
                continue
            bc_key = "network." + key[len("actor.") :]
            if bc_key not in bc_state or tuple(bc_state[bc_key].shape) != tuple(value.shape):
                missing.append(key)
                continue
            diffs.append(max_abs_diff(value, bc_state[bc_key]))
        result[candidate] = {
            "matched_keys": len(diffs),
            "missing_or_shape_mismatch": len(missing),
            "max_abs_diff": max(diffs) if diffs else None,
            "exact": bool(diffs) and all(value == 0.0 for value in diffs) and not missing,
        }
    result["interpretation"] = "state_0 actor and actor_ft are loaded from BC ema only when ema exact=True"
    return result


def normalization_record(path: Path, numpy_module: Any) -> dict[str, Any]:
    record = file_record(path, "normalization")
    if record["status"] == "MISSING":
        return record
    arrays = numpy_module.load(path, allow_pickle=False)
    record["keys"] = list(arrays.files)
    record["arrays"] = {
        key: {
            "shape": list(arrays[key].shape),
            "dtype": str(arrays[key].dtype),
            "finite": bool(numpy_module.isfinite(arrays[key]).all()),
            "min": float(arrays[key].min()),
            "max": float(arrays[key].max()),
        }
        for key in arrays.files
    }
    return record


def line_values(path: Path, names: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {name: None for name in names}
    text = path.read_text(encoding="utf-8")
    values: dict[str, Any] = {}
    for name in names:
        match = re.search(rf"(?m)^\s*{re.escape(name)}:\s*([^#\n]+)", text)
        if not match:
            values[name] = None
            continue
        raw = match.group(1).strip().strip("'\"")
        try:
            values[name] = int(raw)
        except ValueError:
            try:
                values[name] = float(raw)
            except ValueError:
                values[name] = raw
    return values


def runtime_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "environment": {key: os.environ.get(key) for key in ("MUJOCO_PY_MUJOCO_PATH", "MUJOCO_GL", "LD_LIBRARY_PATH")},
    }
    for distribution in ("torch", "numpy", "gym", "mujoco-py", "mujoco", "robomimic", "robosuite", "hydra-core", "omegaconf"):
        try:
            record.setdefault("distributions", {})[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            record.setdefault("distributions", {})[distribution] = None
    try:
        import gym
        import numpy
        import torch

        record["modules"] = {
            "gym": gym.__file__,
            "numpy": numpy.__file__,
            "torch": torch.__file__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - only used on an incomplete runtime
        record["module_error"] = repr(exc)
    try:
        import mujoco

        record.setdefault("modules", {})["mujoco"] = mujoco.__file__
    except Exception as exc:
        record["mujoco_import_error"] = repr(exc)
    try:
        import mujoco_py

        record.setdefault("modules", {}).update(
            {"mujoco_py": mujoco_py.__file__, "mujoco_py_cymj": mujoco_py.cymj.__file__}
        )
    except Exception as exc:
        record["mujoco_py_import_error"] = repr(exc)
    try:
        import robomimic
        import robosuite
        from robosuite.environments.manipulation.two_arm_transport import TwoArmTransport
        from robosuite.models.objects import TransportGroup

        transport_source = Path(inspect.getsourcefile(TwoArmTransport))
        group_source = Path(inspect.getsourcefile(TransportGroup))
        record.setdefault("modules", {}).update(
            {"robomimic": robomimic.__file__, "robosuite": robosuite.__file__}
        )
        record["robosuite_transport"] = {
            "source_path": str(transport_source),
            "source_sha256": sha256(transport_source),
            "native_success_source": inspect.getsource(TwoArmTransport._check_success).strip(),
            "group_source_path": str(group_source),
            "group_source_sha256": sha256(group_source),
            "payload": "HammerObject(name=payload)",
            "trash": "BoxObject(name=trash, size=[0.02, 0.02, 0.02])",
            "lid": "Lid(name=transport_start_bin_lid)",
            "native_success_semantics": "payload_in_target_bin AND trash_in_trash_bin",
        }
    except Exception as exc:
        record["robosuite_import_error"] = repr(exc)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "research" / "S0A_ASSET_MANIFEST.json",
    )
    args = parser.parse_args()

    try:
        import numpy
        import torch
    except Exception as exc:
        raise SystemExit(f"S0a inventory requires the project runtime with torch/numpy: {exc}") from exc

    source_relpaths = [
        "source/dppo_v0.6/model/diffusion/diffusion_vpg.py",
        "source/dppo_v0.6/model/diffusion/diffusion_ppo.py",
        "source/dppo_v0.6/agent/finetune/train_ppo_diffusion_agent.py",
        "source/dppo_v0.6/agent/finetune/train_agent.py",
        "source/dppo_v0.6/env/gym_utils/wrapper/robomimic_lowdim.py",
        "source/dppo_v0.6/env/gym_utils/wrapper/multi_step.py",
    ]
    official_configs = {
        "transport": ROOT / "source/dppo_v0.6/cfg/robomimic/finetune/transport/ft_ppo_diffusion_mlp.yaml",
        "square": ROOT / "source/dppo_v0.6/cfg/robomimic/finetune/square/ft_ppo_diffusion_mlp.yaml",
    }
    resolved_configs = {
        "transport": ROOT / "reports/evidence/fig4_robomimic_transport_dppo_seed42/config.yaml",
        "square": ROOT / "reports/evidence/fig4_robomimic_square_dppo_seed42/config.yaml",
    }

    manifest: dict[str, Any] = {
        "schema": "DPPO-S0a-asset-manifest-v1",
        "scope": "read-only asset and semantic audit; no training or formal long run",
        "project_root": str(ROOT),
        "runtime": runtime_record(),
        "source_files": [file_record(ROOT / path, "specified_source") for path in source_relpaths],
        "reports_and_configs": [
            file_record(ROOT / "reports/FIG4_ROBOMIMIC_TRANSPORT_DPPO_SEED42.md", "formal_report"),
            file_record(ROOT / "reports/FIG4_ROBOMIMIC_SQUARE_DPPO_SEED42.md", "formal_report"),
            file_record(ROOT / "reports/evidence/fig4_robomimic_transport_dppo_seed42/config.yaml", "resolved_config"),
            file_record(ROOT / "reports/evidence/fig4_robomimic_square_dppo_seed42/config.yaml", "resolved_config"),
            file_record(ROOT / "reports/evidence/fig4_robomimic_transport_dppo_seed42/overrides.yaml", "resolved_overrides"),
            file_record(ROOT / "reports/evidence/fig4_robomimic_square_dppo_seed42/overrides.yaml", "resolved_overrides"),
        ],
        "official_config_values": {
            task: line_values(path, [
                "obs_dim", "action_dim", "denoising_steps", "ft_denoising_steps", "horizon_steps",
                "act_steps", "n_envs", "max_episode_steps", "n_steps", "batch_size", "update_epochs",
                "gamma", "gamma_denoising", "min_sampling_denoising_std", "min_logprob_denoising_std",
                "n_train_itr", "val_freq",
            ])
            for task, path in official_configs.items()
        },
        "resolved_config_values": {
            task: line_values(path, [
                "obs_dim", "action_dim", "denoising_steps", "ft_denoising_steps", "horizon_steps",
                "act_steps", "n_envs", "max_episode_steps", "n_steps", "batch_size", "update_epochs",
                "gamma", "gamma_denoising", "min_sampling_denoising_std", "min_logprob_denoising_std",
                "n_train_itr", "val_freq",
            ])
            for task, path in resolved_configs.items()
        },
        "normalization": [
            normalization_record(ROOT / "data/robomimic/transport/normalization.npz", numpy),
            normalization_record(ROOT / "data/robomimic/square/normalization.npz", numpy),
        ],
        "transport_checkpoints": [
            checkpoint_record(TRANSPORT_RUN / f"checkpoint/state_{itr}.pt", torch)
            for itr in (0, 100, 200)
        ],
        "square_selected_ft_checkpoint": checkpoint_record(SQUARE_RUN / "checkpoint/state_200.pt", torch),
        "state0_parameter_comparison": {
            "transport": state0_bc_compare(TRANSPORT_RUN / "checkpoint/state_0.pt", TRANSPORT_BC, torch),
            "square": state0_bc_compare(SQUARE_RUN / "checkpoint/state_0.pt", SQUARE_BC, torch),
        },
        "bc_checkpoints": [
            checkpoint_record(TRANSPORT_BC, torch),
            checkpoint_record(SQUARE_BC, torch),
        ],
        "unrecorded_resume_state": {
            "optimizer": "MISSING by official save_model format",
            "scheduler": "MISSING by official save_model format",
            "python_rng": "MISSING by official save_model format",
            "numpy_rng": "MISSING by official save_model format",
            "torch_rng": "MISSING by official save_model format",
            "running_reward_scaler": "MISSING by official save_model format; resolved configs enable running reward scaling",
            "strict_resume_claim": False,
        },
        "success_semantics": {
            "dppo_reported": "max(chunk_reward) / act_steps >= 1",
            "native_robosuite": "TwoArmTransport._check_success: payload contact with target-bin base AND trash contact with trash-bin base",
            "native_ever_success_in_stored_formal_artifacts": "MISSING; existing run.log/result.pkl contain reward-derived metrics only",
            "mixed_reporting": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({"transport_run": str(TRANSPORT_RUN), "square_run": str(SQUARE_RUN), "transport_bc": str(TRANSPORT_BC), "square_bc": str(SQUARE_BC)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
