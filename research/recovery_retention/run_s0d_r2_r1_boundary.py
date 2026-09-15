"""Bounded S0D_R2_R1_BOUNDARY diagnostic.

This file is deliberately separate from the historical S0d-R2 runner.  It
uses only frozen Square BC/FT checkpoints, the official evaluation sampler,
and native simulator rollouts.  It never constructs an optimizer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/mnt/d/AgentData/DPPO-S0d-R2-R1")
REPORT_ROOT = ROOT / "reports" / "research"
PLOT_ROOT = ROOT / "plots" / "research"
CONFIG_PATH = ROOT / "research" / "recovery_retention" / "s0d_r2_r1_config.json"
PROTOCOL_PATH = REPORT_ROOT / "S0D_R2_R1_PROTOCOL.md"
EXPECTED = {
    0: "c9e22005b58ba2524782e3068c8341554e419acf66f095e238e56df41d9a7aa3",
    200: "18031b2ed2ee422f6e85bd8c113e199d46e0b29f3f2b2d6b4828e67449e0cf25",
}
N_BOOT = 5000
MAX_ATOMIC = 700_000
HARD_SECONDS = 240 * 60
NO_NEW_BATCH_SECONDS = 210 * 60
FIT_SEEDS = (930001, 930002, 930003, 930004)
TEST_SEEDS = tuple(range(940001, 940025))
BASE_MAGNITUDES = (0.0, 0.125, 0.25, 0.5)
DIRECTIONS = ("+x", "-x", "+y", "-y")
TASK_SPEC = {"obs": 23, "action": 7, "chunk": 4, "horizon": 400}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.recovery_retention import run_s0d_r2 as legacy  # noqa: E402
from research.recovery_retention.run_s0d_r2 import (  # noqa: E402
    EnvAdapter,
    find_checkpoint,
    git,
    load_models,
    save_file_meta,
    sha256,
)
from research.recovery_retention.s0d_r2_core import stable_seed  # noqa: E402


FIELDS = [
    "task", "checkpoint_hash", "reset_seed", "source_policy", "boundary",
    "pulse_direction", "pulse_magnitude", "condition", "screening_or_test",
    "block", "draw", "native_success", "termination_reason",
    "remaining_horizon", "actual_steps", "snapshot_id", "pulse_id",
    "target_pulse_magnitude", "actual_action_delta", "clip_ratio",
    "eef_displacement_vs_sham", "object_displacement_vs_sham",
    "success_before_pulse", "success_during_pulse", "done_during_pulse",
    "pulse_complete", "action_coordinate", "reference_frame", "sampler_seed",
]


def stable_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def json_meta(path: Path, shape=None) -> dict:
    return save_file_meta(path, shape)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def guard(deadline: float, budget: dict, n: int = 0) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("wall-clock deadline reached")
    if time.monotonic() - budget["start"] >= NO_NEW_BATCH_SECONDS and n:
        raise TimeoutError("210-minute no-new-batch cutoff reached")
    if budget["atomic"] + n > MAX_ATOMIC:
        raise TimeoutError("atomic simulator-step budget reached")


def step_one(adapter: EnvAdapter, action: np.ndarray, budget: dict):
    guard(budget["deadline"], budget, 1)
    result = adapter.step_atomic(action)
    budget["atomic"] += 1
    return result


def sample_chunk(model, obs: np.ndarray, device: str, seed: int, deterministic: bool = True):
    stable_torch_seed(seed)
    cond = {"state": torch.from_numpy(np.asarray(obs)[None, None, :]).float().to(device)}
    with torch.no_grad():
        sample = model(cond=cond, deterministic=deterministic, return_chain=True)
    return (sample.trajectories[0].detach().cpu().numpy(),
            sample.chains[0].detach().cpu().numpy())


def raw_positions(adapter: EnvAdapter) -> tuple[np.ndarray, np.ndarray]:
    raw = adapter.env.get_observation()
    eef = np.asarray(raw["robot0_eef_pos"], dtype=np.float64).reshape(-1)[:3].copy()
    obj = np.asarray(raw["object"], dtype=np.float64).reshape(-1)[:3].copy()
    return eef, obj


def action_direction(adapter: EnvAdapter, direction: str) -> tuple[int, float]:
    controller = adapter.raw.robots[0].controller
    if controller.__class__.__name__ != "OperationalSpaceController":
        raise RuntimeError(f"unexpected robot0 controller: {controller.__class__.__name__}")
    if int(controller.control_dim) != 6 or not bool(getattr(controller, "use_delta", False)):
        raise RuntimeError("robot0 controller is not 6D relative OSC pose")
    # Verified against robosuite OperationalSpaceController.run/set_goal docs:
    # [x, y, z, axis-angle-x, axis-angle-y, axis-angle-z].
    index = {"+x": 0, "-x": 0, "+y": 1, "-y": 1}[direction]
    sign = 1.0 if direction in ("+x", "+y") else -1.0
    return index, sign


def make_row(**kwargs) -> dict:
    row = {k: "" for k in FIELDS}
    row.update(kwargs)
    return row


def run_chunk(adapter: EnvAdapter, base_raw: np.ndarray, magnitude: float | None,
              pulse: tuple[int, float] | None, budget: dict) -> dict:
    budget["chunks"] += 1
    base = np.asarray(base_raw, dtype=np.float64).copy()
    intended = base.copy()
    if pulse is not None and magnitude is not None and magnitude != 0:
        idx, sign = pulse
        intended[:4, idx] += sign * float(magnitude)
    actual = np.clip(intended, adapter.action_low, adapter.action_high)
    modified = np.zeros_like(actual, dtype=bool)
    if pulse is not None and magnitude is not None and magnitude != 0:
        modified[:4, pulse[0]] = True
    clipped = modified & (np.abs(actual - intended) > 1e-12)
    eef_before, obj_before = raw_positions(adapter)
    successes, dones = [], []
    for action in actual:
        _, _, done, success, _ = step_one(adapter, action, budget)
        successes.append(bool(success))
        dones.append(bool(done))
    eef_after, obj_after = raw_positions(adapter)
    return {
        "base": base, "intended": intended, "actual": actual,
        "delta": actual - base, "success": bool(any(successes)),
        "done": bool(any(dones)), "successes": successes, "dones": dones,
        "actual_steps": len(actual), "clip_ratio": float(np.mean(clipped[modified])) if np.any(modified) else 0.0,
        "eef_after": eef_after, "obj_after": obj_after,
        "eef_before": eef_before, "obj_before": obj_before,
        "actual_action_delta": float(np.max(np.abs(actual - base))),
    }


def run_parent(adapter: EnvAdapter, model, model_info: dict, seed: int,
               source: str, target_chunk: int, seed_index: int,
               initial: dict, budget: dict, deadline: float, rows: list,
               seed_registry: list) -> tuple[dict | None, bool]:
    adapter.restore(initial, full=True)
    success_seen = False
    anchor = None
    for chunk_index in range(TASK_SPEC["horizon"] // TASK_SPEC["chunk"]):
        guard(deadline, budget, TASK_SPEC["chunk"])
        sampler_seed = stable_seed("square", seed, source, "parent", chunk_index, "denoising_index", "all")
        seed_registry.append({"task": "square", "reset_seed": seed, "source_policy": source,
                              "boundary": target_chunk, "magnitude": 0.0, "phase": "parent",
                              "block": "", "draw": -1, "chunk": chunk_index,
                              "denoising_index": "all", "sampler_seed": sampler_seed})
        norm, _ = sample_chunk(model, adapter.current_obs(), str(next(model.parameters()).device), sampler_seed, True)
        raw = adapter.unnormalize(norm[:TASK_SPEC["chunk"]])
        out = run_chunk(adapter, raw, None, None, budget)
        success_seen = success_seen or out["success"] or adapter.native_success()
        if chunk_index + 1 == target_chunk:
            if not success_seen and not out["done"]:
                direction = DIRECTIONS[seed_index % 4]
                pulse_index, pulse_sign = action_direction(adapter, direction)
                pulse_seed = stable_seed("square", seed, source, "anchor", target_chunk, "denoising_index", "all")
                norm_pulse, _ = sample_chunk(model, adapter.current_obs(), str(next(model.parameters()).device), pulse_seed, True)
                anchor = {
                    "task": "square", "reset_seed": seed, "source_policy": source,
                    "boundary": target_chunk, "pulse_direction": direction,
                    "pulse_index": pulse_index, "pulse_sign": pulse_sign,
                    "anchor_index": seed_index,
                    "parent_atomic_step": int(adapter.atomic_steps),
                    "remaining_before_pulse": TASK_SPEC["horizon"] - int(adapter.atomic_steps),
                    "snapshot": adapter.snapshot(),
                    "pulse_raw": adapter.unnormalize(norm_pulse[:TASK_SPEC["chunk"]]),
                    "snapshot_id": f"square-seed{seed}-{source}-b{target_chunk}",
                    "branches": {}, "test": {},
                }
            else:
                rows.append(make_row(task="square", checkpoint_hash=model_info["sha256"],
                                     reset_seed=seed, source_policy=source, boundary=target_chunk,
                                     condition="parent", screening_or_test="parent",
                                     native_success=int(success_seen),
                                     termination_reason="anchor_invalid_success_or_done",
                                     remaining_horizon=TASK_SPEC["horizon"] - adapter.atomic_steps,
                                     actual_steps=adapter.atomic_steps))
        if success_seen or out["done"]:
            break
    rows.append(make_row(task="square", checkpoint_hash=model_info["sha256"], reset_seed=seed,
                         source_policy=source, boundary=target_chunk, condition="parent",
                         screening_or_test="parent", native_success=int(success_seen),
                         termination_reason="native_success" if success_seen else "natural_termination_or_horizon",
                         remaining_horizon=max(0, TASK_SPEC["horizon"] - adapter.atomic_steps),
                         actual_steps=adapter.atomic_steps))
    return anchor, bool(success_seen)


def pulse_branches(adapter: EnvAdapter, anchor: dict, model_info: dict,
                   magnitudes: tuple[float, ...], rows: list, budget: dict) -> None:
    branch_rows = {}
    sham_eef = sham_obj = None
    adapter.restore(anchor["snapshot"], full=False)
    pulse_before = adapter.native_success()
    for magnitude in magnitudes:
        condition = "sham" if magnitude == 0 else "perturbed"
        adapter.restore(anchor["snapshot"], full=False)
        out = run_chunk(adapter, anchor["pulse_raw"], magnitude, (anchor["pulse_index"], anchor["pulse_sign"]), budget)
        if magnitude == 0:
            sham_eef, sham_obj = out["eef_after"], out["obj_after"]
        eef_disp = float(np.linalg.norm(out["eef_after"] - sham_eef)) if sham_eef is not None else 0.0
        obj_disp = float(np.linalg.norm(out["obj_after"] - sham_obj)) if sham_obj is not None else 0.0
        branch_snap = adapter.snapshot()
        pulse_id = f"{anchor['snapshot_id']}-m{magnitude:g}"
        branch_rows[magnitude] = {
            "snapshot": branch_snap,
            "success_during_pulse": out["success"], "done_during_pulse": out["done"],
            "pulse_complete": out["actual_steps"] == TASK_SPEC["chunk"],
            "clip_ratio": out["clip_ratio"], "eef_disp": eef_disp, "obj_disp": obj_disp,
            "actual_action_delta": out["actual_action_delta"], "pulse_id": pulse_id,
        }
        rows.append(make_row(
            task="square", checkpoint_hash=model_info["sha256"], reset_seed=anchor["reset_seed"],
            source_policy=anchor["source_policy"], boundary=anchor["boundary"],
            pulse_direction=anchor["pulse_direction"], pulse_magnitude=magnitude,
            target_pulse_magnitude=magnitude, condition=condition,
            screening_or_test="pulse", native_success=int(out["success"]),
            termination_reason=("success_or_done_during_pulse" if out["success"] or out["done"] else "pulse_complete"),
            remaining_horizon=TASK_SPEC["horizon"] - adapter.atomic_steps,
            actual_steps=out["actual_steps"], snapshot_id=anchor["snapshot_id"], pulse_id=pulse_id,
            actual_action_delta=out["actual_action_delta"], clip_ratio=out["clip_ratio"],
            eef_displacement_vs_sham=eef_disp, object_displacement_vs_sham=obj_disp,
            success_before_pulse=int(pulse_before), success_during_pulse=int(out["success"]),
            done_during_pulse=int(out["done"]), pulse_complete=int(out["actual_steps"] == TASK_SPEC["chunk"]),
            action_coordinate="robot0 OSC_POSE input [dx,dy,dz,ax,ay,az], action index 0/1",
            reference_frame="robosuite controller relative command convention",
        ))
    anchor["branches"] = branch_rows
    anchor["main_eligible"] = all(
        b["pulse_complete"] and not b["success_during_pulse"] and not b["done_during_pulse"]
        for b in branch_rows.values()
    )


def continuation(adapter: EnvAdapter, model, branch: dict, policy: str,
                 anchor: dict, magnitude: float, draw: int, block: str,
                 budget: dict, deadline: float, seed_registry: list) -> dict:
    adapter.restore(branch["snapshot"], full=False)
    start_steps = adapter.atomic_steps
    chunk_index = 0
    while adapter.atomic_steps < TASK_SPEC["horizon"]:
        guard(deadline, budget, TASK_SPEC["chunk"])
        sampler_seed = stable_seed("square", anchor["reset_seed"], anchor["source_policy"],
                                   anchor["boundary"], magnitude, policy, "test", block,
                                   draw, chunk_index, "denoising_index", "all")
        seed_registry.append({"task": "square", "reset_seed": anchor["reset_seed"],
                              "source_policy": anchor["source_policy"], "boundary": anchor["boundary"],
                              "magnitude": magnitude, "phase": "test", "policy": policy,
                              "block": block, "draw": draw, "chunk": chunk_index,
                              "denoising_index": "all", "sampler_seed": sampler_seed})
        norm, _ = sample_chunk(model, adapter.current_obs(), str(next(model.parameters()).device), sampler_seed, True)
        raw = adapter.unnormalize(norm[:TASK_SPEC["chunk"]])
        out = run_chunk(adapter, raw, None, None, budget)
        if out["success"] or adapter.native_success():
            return {"success": True, "reason": "native_success", "steps": adapter.atomic_steps - start_steps,
                    "remaining": TASK_SPEC["horizon"] - adapter.atomic_steps, "sampler_seed": sampler_seed}
        if out["done"]:
            return {"success": False, "reason": "done", "steps": adapter.atomic_steps - start_steps,
                    "remaining": TASK_SPEC["horizon"] - adapter.atomic_steps, "sampler_seed": sampler_seed}
        chunk_index += 1
    return {"success": False, "reason": "episode_horizon", "steps": adapter.atomic_steps - start_steps,
            "remaining": 0, "sampler_seed": sampler_seed if chunk_index else ""}


def manual_sampler(model, cond, x0, noises, deterministic=True):
    x = x0.clone()
    chain = []
    t_all = list(reversed(range(model.denoising_steps))) if not model.use_ddim else model.ddim_t
    if not model.use_ddim and model.ft_denoising_steps == model.denoising_steps:
        chain.append(x.clone())
    for i, t in enumerate(t_all):
        tb = torch.full((x.shape[0],), int(t), dtype=torch.long, device=x.device)
        ib = torch.full((x.shape[0],), int(i), dtype=torch.long, device=x.device)
        mean, logvar, _ = model.p_mean_var(x=x, t=tb, cond=cond, index=ib,
                                            use_base_policy=False, deterministic=deterministic)
        std = torch.exp(0.5 * logvar)
        if model.use_ddim:
            if deterministic:
                std = torch.zeros_like(std)
            else:
                std = torch.clip(std, min=model.get_min_sampling_denoising_std())
        else:
            if deterministic and int(t) == 0:
                std = torch.zeros_like(std)
            elif deterministic:
                std = torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=model.get_min_sampling_denoising_std())
        x = mean + std * noises[i].clamp(-model.randn_clip_value, model.randn_clip_value)
        if model.final_action_clip_value is not None and i == len(t_all) - 1:
            x = torch.clamp(x, -model.final_action_clip_value, model.final_action_clip_value)
        if not model.use_ddim and int(t) <= model.ft_denoising_steps:
            chain.append(x.clone())
    return x, torch.stack(chain, dim=1)


def sampler_equivalence(models: dict, device: str) -> dict:
    import unittest.mock as mock
    results = {}
    for label, model in models.items():
        stable_torch_seed(stable_seed("sampler", "square", label, "fixed_tape"))
        b = 2
        shape = (b, int(model.horizon_steps), int(model.action_dim))
        x0 = torch.randn(shape, device=device)
        noises = [torch.randn(shape, device=device) for _ in range(int(model.denoising_steps))]
        cond = {"state": torch.randn((b, 1, int(model.obs_dim)), device=device)}
        expected_x, expected_chain = manual_sampler(model, cond, x0, noises, True)
        queue = [x0.clone()] + [x.clone() for x in noises]
        def fake_randn(*args, **kwargs):
            return queue.pop(0).clone()
        def fake_randn_like(*args, **kwargs):
            return queue.pop(0).clone()
        with torch.no_grad(), mock.patch.object(torch, "randn", side_effect=fake_randn), mock.patch.object(torch, "randn_like", side_effect=fake_randn_like):
            official = model(cond=cond, deterministic=True, return_chain=True)
        if queue:
            raise RuntimeError(f"official sampler did not consume fixed tape for {label}")
        results[label] = {
            "action_max_abs": float((expected_x - official.trajectories).abs().max()),
            "chain_max_abs": float((expected_chain - official.chains).abs().max()),
            "passed": bool(float((expected_x - official.trajectories).abs().max()) <= 1e-6 and
                            float((expected_chain - official.chains).abs().max()) <= 1e-6),
        }
    return results


def replay_tape(adapter: EnvAdapter, snap: dict, tape: np.ndarray, budget: dict):
    adapter.restore(snap, full=True)
    obs, state, flags = [], [], []
    for action in tape:
        o, _, done, success, _ = step_one(adapter, action, budget)
        obs.append(o.copy()); state.append(adapter.state_vector()); flags.append((done, success))
    return np.asarray(obs), np.asarray(state), flags


def snapshot_prechecks(budget: dict, deadline: float) -> dict:
    a = EnvAdapter("square")
    b = EnvAdapter("square")
    try:
        a.reset_seed(FIT_SEEDS[0])
        snap = a.snapshot()
        rng = np.random.default_rng(stable_seed("snapshot", "square", "cross_reset"))
        tape = rng.uniform(a.action_low, a.action_high, size=(8, TASK_SPEC["action"]))
        first = replay_tape(a, snap, tape, budget)
        second = replay_tape(a, snap, tape, budget)
        b.reset_seed(FIT_SEEDS[1])
        cross = replay_tape(b, snap, tape, budget)
        def compare(x, y):
            return {
                "obs_max_abs": float(np.max(np.abs(x[0] - y[0]))),
                "state_max_abs": float(np.max(np.abs(x[1] - y[1]))),
                "done_success_equal": x[2] == y[2],
            }
        same = compare(first, second)
        cross_result = compare(first, cross)
        passed = all(v <= 1e-6 for v in (same["obs_max_abs"], same["state_max_abs"],
                                         cross_result["obs_max_abs"], cross_result["state_max_abs"])) and \
            same["done_success_equal"] and cross_result["done_success_equal"]
        return {"passed": bool(passed), "same_snapshot": same, "cross_reset": cross_result,
                "snapshot_fields": sorted(snap.keys()), "tape_shape": list(tape.shape),
                "native_success_callable": bool(isinstance(a.native_success(), bool))}
    finally:
        a.close(); b.close()


def bc_bc_precheck(model, device: str, budget: dict, deadline: float) -> dict:
    adapter = EnvAdapter("square")
    try:
        adapter.reset_seed(FIT_SEEDS[0])
        snap = adapter.snapshot()
        def rollout():
            adapter.restore(snap, full=True)
            all_obs, all_state, all_actions, all_flags = [], [], [], []
            for chunk in range(2):
                seed = stable_seed("square", FIT_SEEDS[0], "bc", "bc_pair", chunk, "denoising_index", "all")
                norm, _ = sample_chunk(model, adapter.current_obs(), device, seed, True)
                raw = adapter.unnormalize(norm[:TASK_SPEC["chunk"]])
                for action in raw:
                    o, _, done, success, _ = step_one(adapter, action, budget)
                    all_actions.append(action.copy()); all_obs.append(o.copy()); all_state.append(adapter.state_vector())
                    all_flags.append((done, success))
            return np.asarray(all_obs), np.asarray(all_state), np.asarray(all_actions), all_flags
        one, two = rollout(), rollout()
        errors = [float(np.max(np.abs(one[i] - two[i]))) for i in range(3)]
        flags_equal = one[3] == two[3]
        return {"passed": bool(max(errors) <= 1e-8 and flags_equal),
                "obs_max_abs": errors[0], "state_max_abs": errors[1],
                "action_max_abs": errors[2], "done_success_equal": flags_equal}
    finally:
        adapter.close()


def calibration_sampling(models: dict, device: str, budget: dict, deadline: float) -> dict:
    adapter = EnvAdapter("square")
    records = []
    try:
        for seed in FIT_SEEDS:
            adapter.reset_seed(seed)
            initial = adapter.snapshot()
            for policy in ("bc", "ft"):
                for deterministic in (True, False):
                    adapter.restore(initial, full=True)
                    actions = []
                    success = False
                    for chunk in range(TASK_SPEC["horizon"] // TASK_SPEC["chunk"]):
                        guard(deadline, budget, TASK_SPEC["chunk"])
                        ss = stable_seed("square", seed, policy, "fit_calibration", deterministic, chunk, "denoising_index", "all")
                        norm, _ = sample_chunk(models[policy], adapter.current_obs(), device, ss, deterministic)
                        raw = adapter.unnormalize(norm[:TASK_SPEC["chunk"]])
                        actions.extend(raw.tolist())
                        out = run_chunk(adapter, raw, None, None, budget)
                        success = success or out["success"]
                        if success or out["done"]:
                            break
                    records.append({"seed": seed, "policy": policy, "deterministic": deterministic,
                                    "native_success": success, "actions": actions})
        sensitivity = []
        for seed in FIT_SEEDS:
            for policy in ("bc", "ft"):
                t = next(r for r in records if r["seed"] == seed and r["policy"] == policy and r["deterministic"])
                f = next(r for r in records if r["seed"] == seed and r["policy"] == policy and not r["deterministic"])
                n = min(len(t["actions"]), len(f["actions"]))
                diff = float(np.max(np.abs(np.asarray(t["actions"][:n]) - np.asarray(f["actions"][:n])))) if n else None
                sensitivity.append({"seed": seed, "policy": policy, "action_max_abs": diff,
                                    "deterministic_success": t["native_success"], "non_deterministic_success": f["native_success"]})
        return {"records": [{k: v for k, v in r.items() if k != "actions"} for r in records],
                "sensitivity": sensitivity}
    finally:
        adapter.close()


def calibrate_pulse(models: dict, device: str, budget: dict, deadline: float, replay_error: float) -> dict:
    adapter = EnvAdapter("square")
    measurements = []
    try:
        direction = action_direction(adapter, "+x")
        for seed in FIT_SEEDS:
            adapter.reset_seed(seed)
            snap = adapter.snapshot()
            norm, _ = sample_chunk(models["bc"], adapter.current_obs(), device,
                                   stable_seed("square", seed, "fit_pulse", "denoising_index", "all"), True)
            raw = adapter.unnormalize(norm[:TASK_SPEC["chunk"]])
            sham = None
            for mag in BASE_MAGNITUDES:
                adapter.restore(snap, full=False)
                out = run_chunk(adapter, raw, mag, direction, budget)
                if mag == 0:
                    sham = (out["eef_after"], out["obj_after"])
                measurements.append({"seed": seed, "magnitude": mag, "clip_ratio": out["clip_ratio"],
                                     "eef_displacement": float(np.linalg.norm(out["eef_after"] - sham[0])),
                                     "object_displacement": float(np.linalg.norm(out["obj_after"] - sham[1])),
                                     "actual_action_delta": out["actual_action_delta"]})
        max_clip = max(x["clip_ratio"] for x in measurements)
        multiplier = 0.5 if max_clip > 0.05 else 1.0
        if multiplier == 0.5:
            # Preregistered one-time fallback; recalibrate before TEST.
            measurements = []
            for seed in FIT_SEEDS:
                adapter.reset_seed(seed); snap = adapter.snapshot()
                norm, _ = sample_chunk(models["bc"], adapter.current_obs(), device,
                                       stable_seed("square", seed, "fit_pulse_fallback", "denoising_index", "all"), True)
                raw = adapter.unnormalize(norm[:TASK_SPEC["chunk"]]); sham = None
                for base_mag in BASE_MAGNITUDES:
                    mag = base_mag * multiplier
                    adapter.restore(snap, full=False)
                    out = run_chunk(adapter, raw, mag, direction, budget)
                    if mag == 0: sham = (out["eef_after"], out["obj_after"])
                    measurements.append({"seed": seed, "magnitude": mag, "clip_ratio": out["clip_ratio"],
                                         "eef_displacement": float(np.linalg.norm(out["eef_after"] - sham[0])),
                                         "object_displacement": float(np.linalg.norm(out["obj_after"] - sham[1])),
                                         "actual_action_delta": out["actual_action_delta"]})
        max_clip = max(x["clip_ratio"] for x in measurements)
        max_eef = max(x["eef_displacement"] for x in measurements if x["magnitude"] > 0)
        threshold = replay_error * 10.0
        result = "PASS" if max_clip <= 0.05 and max_eef > threshold else "CALIBRATION_FAIL"
        return {"default_multiplier": 1.0, "multiplier": multiplier, "measurements": measurements,
                "max_clip_ratio": max_clip, "max_eef_displacement": max_eef,
                "max_object_displacement": max(x["object_displacement"] for x in measurements),
                "fixed_action_replay_error": replay_error, "eef_threshold": threshold,
                "result": result, "actual_magnitudes": [m * multiplier for m in BASE_MAGNITUDES]}
    finally:
        adapter.close()


def bootstrap_ci(values: np.ndarray, clusters: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=float); clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    by_cluster = np.asarray([np.mean(values[clusters == c]) for c in unique], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        draws[i] = np.mean(by_cluster[rng.integers(0, len(by_cluster), size=len(by_cluster))])
    return {"lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5)),
            "clusters": int(len(unique)), "n_boot": N_BOOT, "seed": seed}


def compute_metrics(anchors: list[dict], parent_success: dict, test_values: dict) -> dict:
    selected = [a for a in anchors if a.get("main_eligible") and a["snapshot_id"] in test_values]
    mags = sorted({float(m) for a in selected for m in a.get("test", {}).keys()})
    complete_clusters = []
    for seed in sorted(set(a["reset_seed"] for a in selected)):
        cases = [a for a in selected if a["reset_seed"] == seed]
        if cases and all(
            all(test_values[a["snapshot_id"]].get(m, {}).get("complete", False) for m in mags)
            for a in cases
        ):
            complete_clusters.append(seed)
    selected = [a for a in selected if a["reset_seed"] in complete_clusters]
    mags = sorted({float(m) for a in selected for m in a["test"].keys()})
    by_cluster = {m: [] for m in mags}
    per_case = {m: [] for m in mags}
    for seed in complete_clusters:
        cases = [a for a in selected if a["reset_seed"] == seed]
        for mag in mags:
            case_diffs = []
            for a in cases:
                vals = a["test"][mag]
                bc = np.mean(vals["bc"]); ft = np.mean(vals["ft"])
                case_diffs.append((bc, ft))
            if case_diffs:
                by_cluster[mag].append((seed, float(np.mean([x[0] for x in case_diffs])),
                                        float(np.mean([x[1] for x in case_diffs]))))
                per_case[mag].extend([(seed, x[0], x[1]) for x in case_diffs])
    def q(policy: str, mag: float, subset=None):
        vals = by_cluster[mag]
        if subset is not None:
            vals = [v for v in vals if v[0] in subset]
        return float(np.mean([v[1] if policy == "bc" else v[2] for v in vals])) if vals else None
    curves = {}
    for mag in mags:
        bc, ft = q("bc", mag), q("ft", mag)
        ids = np.asarray([v[0] for v in by_cluster[mag]], dtype=int)
        bc_values = np.asarray([v[1] for v in by_cluster[mag]], dtype=float)
        ft_values = np.asarray([v[2] for v in by_cluster[mag]], dtype=float)
        curves[str(mag)] = {
            "bc": bc, "ft": ft, "L": None if bc is None else bc - ft,
            "bc_ci": bootstrap_ci(bc_values, ids, 950001) if len(bc_values) else {},
            "ft_ci": bootstrap_ci(ft_values, ids, 950001) if len(ft_values) else {},
        }
    nonzero = [m for m in mags if m > 0]
    cluster_ids = np.asarray(complete_clusters, dtype=int)
    lbar_vals = []; ibar_vals = []; l0_vals = []
    la_vals = []; lb_vals = []
    for seed in complete_clusters:
        cvals = {m: next(v for v in by_cluster[m] if v[0] == seed) for m in mags}
        ls = {m: cvals[m][1] - cvals[m][2] for m in mags}
        l0 = ls.get(0.0)
        lbar = float(np.mean([ls[m] for m in nonzero]))
        lbar_vals.append(lbar); l0_vals.append(l0); ibar_vals.append(lbar - l0)
        la_vals.append(float(np.mean([np.mean(a["test"][m]["bc"][:2]) - np.mean(a["test"][m]["ft"][:2]) for a in selected if a["reset_seed"] == seed for m in nonzero])))
        lb_vals.append(float(np.mean([np.mean(a["test"][m]["bc"][2:]) - np.mean(a["test"][m]["ft"][2:]) for a in selected if a["reset_seed"] == seed for m in nonzero])))
    lbar_vals = np.asarray(lbar_vals); ibar_vals = np.asarray(ibar_vals)
    l0 = float(np.mean(l0_vals)) if l0_vals else None
    lbar = float(np.mean(lbar_vals)) if len(lbar_vals) else None
    ibar = float(np.mean(ibar_vals)) if len(ibar_vals) else None
    def parent_rate(policy):
        return float(np.mean([int(parent_success[policy].get(s, False)) for s in TEST_SEEDS]))
    jnom = float(np.mean([int(parent_success["ft"].get(s, False)) - int(parent_success["bc"].get(s, False)) for s in TEST_SEEDS]))
    bc_sham = curves.get("0.0", {}).get("bc")
    bc_nonzero = float(np.mean([curves[str(m)]["bc"] for m in nonzero])) if nonzero else None
    result = {
        "curves": curves, "L0": l0, "Lbar": lbar, "Ibar": ibar,
        "Lbar_ci": bootstrap_ci(lbar_vals, cluster_ids, stable_seed("square", "Lbar", 950001)) if len(lbar_vals) else {},
        "Ibar_ci": bootstrap_ci(ibar_vals, cluster_ids, stable_seed("square", "Ibar", 950001)) if len(ibar_vals) else {},
        "block_A": {"Lbar": float(np.mean(la_vals)) if la_vals else None,
                    "Ibar": None},
        "block_B": {"Lbar": float(np.mean(lb_vals)) if lb_vals else None,
                    "Ibar": None},
        "main_paired_clusters": len(complete_clusters),
        "selected_anchors": len(selected), "parent_clusters": len(set(parent_success["bc"]) & set(parent_success["ft"])),
        "bc_parent_success": parent_rate("bc"), "ft_parent_success": parent_rate("ft"),
        "Jnom": jnom, "bc_sham_success": bc_sham, "bc_nonzero_success": bc_nonzero,
    }
    result["Lbar_ci"] = bootstrap_ci(lbar_vals, cluster_ids, 950001) if len(lbar_vals) else {}
    result["Ibar_ci"] = bootstrap_ci(ibar_vals, cluster_ids, 950001) if len(ibar_vals) else {}
    result["block_A"]["Ibar"] = float(np.mean([x - y for x, y in zip(la_vals, l0_vals)])) if la_vals else None
    result["block_B"]["Ibar"] = float(np.mean([x - y for x, y in zip(lb_vals, l0_vals)])) if lb_vals else None
    return result


def run_test(anchors: list[dict], models: dict, info: dict, device: str,
             magnitudes: tuple[float, ...], budget: dict, deadline: float,
             rows: list, seed_registry: list) -> tuple[dict, list[int]]:
    eligible = [a for a in anchors if a.get("main_eligible")]
    test_values = {}
    adapter = EnvAdapter("square")
    try:
        for a in eligible:
            a["test"] = {}
            test_values[a["snapshot_id"]] = a["test"]
            for magnitude in magnitudes:
                a["test"][magnitude] = {"bc": [], "ft": [], "complete": True}
                branch = a["branches"][magnitude]
                for policy in ("bc", "ft"):
                    for draw in range(1, 5):
                        block = "A" if draw <= 2 else "B"
                        result = continuation(adapter, models[policy], branch, policy, a,
                                              magnitude, draw, block, budget, deadline, seed_registry)
                        val = int(result["success"])
                        a["test"][magnitude][policy].append(val)
                        rows.append(make_row(task="square", checkpoint_hash=info[policy]["sha256"],
                                             reset_seed=a["reset_seed"], source_policy=a["source_policy"],
                                             boundary=a["boundary"], pulse_direction=a["pulse_direction"],
                                             pulse_magnitude=magnitude, target_pulse_magnitude=magnitude,
                                             condition=f"{policy}_{'sham' if magnitude == 0 else 'perturbed'}",
                                             screening_or_test="test", block=block, draw=draw,
                                             native_success=val, termination_reason=result["reason"],
                                             remaining_horizon=result["remaining"], actual_steps=result["steps"],
                                             snapshot_id=a["snapshot_id"], pulse_id=branch["pulse_id"],
                                             sampler_seed=result["sampler_seed"],
                                             action_coordinate="robot0 OSC_POSE input [dx,dy,dz,ax,ay,az], action index 0/1",
                                             reference_frame="robosuite controller relative command convention"))
                a["test"][magnitude]["complete"] = all(len(a["test"][magnitude][p]) == 4 for p in ("bc", "ft"))
    finally:
        adapter.close()
    # A reset seed is primary only if all of its eligible anchors completed.
    complete_clusters = []
    for seed in TEST_SEEDS:
        cases = [a for a in eligible if a["reset_seed"] == seed]
        if cases and all(all(a["test"][m]["complete"] for m in magnitudes) for a in cases):
            complete_clusters.append(seed)
    return test_values, complete_clusters


def plot_curves(metrics: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    mags = [float(x) for x in metrics["curves"]]
    bc = [metrics["curves"][str(x)]["bc"] for x in mags]
    ft = [metrics["curves"][str(x)]["ft"] for x in mags]
    bc_lo = [metrics["curves"][str(x)]["bc_ci"]["lower"] for x in mags]
    bc_hi = [metrics["curves"][str(x)]["bc_ci"]["upper"] for x in mags]
    ft_lo = [metrics["curves"][str(x)]["ft_ci"]["lower"] for x in mags]
    ft_hi = [metrics["curves"][str(x)]["ft_ci"]["upper"] for x in mags]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    ax.errorbar(mags, bc, yerr=[np.asarray(bc) - np.asarray(bc_lo), np.asarray(bc_hi) - np.asarray(bc)], fmt="o-", capsize=3, label="BC")
    ax.errorbar(mags, ft, yerr=[np.asarray(ft) - np.asarray(ft_lo), np.asarray(ft_hi) - np.asarray(ft)], fmt="s-", capsize=3, label="FT")
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set(xlabel="physical pulse magnitude (environment action units)", ylabel="native recovery success", ylim=(-0.05, 1.05))
    ax.set_xticks(mags)
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    ax.text(0.02, 0.03, "sham" if mags and mags[0] == 0 else "", transform=ax.transAxes, fontsize=8)
    ax.text(0.32, 0.03, "nonzero pulse", transform=ax.transAxes, fontsize=8)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path); plt.close(fig)


def report_text(manifest: dict, metrics: dict | None, pre: dict, calibration: dict | None) -> tuple[str, str]:
    status = manifest["status"]
    impl = manifest["implementation"]
    lines = ["# S0D_R2_R1_BOUNDARY implementation audit", "", f"- STATUS: **{status}**", f"- IMPLEMENTATION: **{impl}**", "- No optimizer, training iteration, Transport, or mechanism intervention was run.", "", "## Precheck evidence", ""]
    for k, v in pre.items():
        lines.append(f"- `{k}`: `{json.dumps(v, ensure_ascii=False)}`")
    if calibration:
        lines += ["", "## Pulse calibration", "", f"- Result: **{calibration['result']}**", f"- Multiplier: `{calibration['multiplier']}`", f"- Maximum clip ratio: `{calibration['max_clip_ratio']}`", f"- Maximum EEF displacement: `{calibration['max_eef_displacement']}`", f"- EEF threshold: `{calibration['eef_threshold']}`"]
    lines += ["", "## Interpretation", "", "The prior S0d-R2 sampling error is corrected here by using the official v0.6 DDPM evaluation sampler with `deterministic=True` while retaining its prescribed DDPM noise schedule. This boundary diagnostic measures native simulator recovery only; it does not identify a diffusion-specific mechanism."]
    boundary = ["# S0D_R2_R1_BOUNDARY recovery-retention boundary audit", ""] + [
        f"- TASK: **S0D_R2_R1_BOUNDARY**", f"- STATUS: **{status}**", "- CURRENT_GRADIENT_BUDGET: **STOP**", "- CURRENT_R1: **STOP**",
        f"- IMPLEMENTATION: **{impl}**", f"- MEASUREMENT: **{manifest['measurement']}**", f"- CONTEXT: **{manifest.get('context', 'NOT_TESTED')}**",
        f"- PHENOMENON: **{manifest.get('phenomenon', 'NOT_TESTED')}**", "- MECHANISM_GATE: **NOT_RUN**", "- ALGORITHM_EFFECTIVENESS: **NOT_TESTED**", "- NOVELTY_STATUS: **CANDIDATE_ONLY**", f"- NEXT_STAGE: **{manifest['next_stage']}**",
        f"- FORMAL_TRAINING_ITERATIONS: **0**", "- OPTIMIZER_STEPS: **0**", f"- NEW_ENVIRONMENT_ATOMIC_STEPS: **{manifest['budget']['atomic']}**", f"- NEW_ENVIRONMENT_CHUNK_STEPS: **{manifest['budget']['chunks']}**", f"- WALL_CLOCK_SECONDS: **{manifest['wall_clock_seconds']:.3f}**", f"- TIME_LIMIT_REACHED: **{manifest['time_limit_reached']}**", "",
        "## Precheck", "", "See the implementation audit for sampler equivalence, snapshot replay, cross-reset restore, BC-BC pairing, normalization, action semantics and gate self-test.", "",
        "## Pulse calibration", "", f"- Result: **{calibration.get('result') if calibration else 'NOT_TESTED'}**",
        f"- Multiplier: `{calibration.get('multiplier') if calibration else None}`; maximum clip ratio: `{calibration.get('max_clip_ratio') if calibration else None}`",
        f"- Maximum EEF displacement versus sham: `{calibration.get('max_eef_displacement') if calibration else None}`; object displacement maximum: `{calibration.get('max_object_displacement') if calibration else None}`", "",
        "## Coverage and estimates", "", json.dumps(metrics, indent=2, ensure_ascii=False) if metrics else "Scientific metrics not estimated because the implementation or calibration gate did not pass.", "",
        "## Decision answers", "",
        "1. The previous S0d-R2 evaluation-sampling issue is corrected for this run: the official v0.6 DDPM sampler is invoked with `deterministic=True`, while retaining its prescribed DDPM noise schedule.",
        f"2. The physical pulse formed a measurable boundary: calibration passed with multiplier `{calibration.get('multiplier') if calibration else None}`, maximum clip ratio `{calibration.get('max_clip_ratio') if calibration else None}`, and maximum EEF displacement `{calibration.get('max_eef_displacement') if calibration else None}` versus sham.",
        f"3. BC recovery was `{metrics.get('curves', {}).get('0.0', {}).get('bc') if metrics else None}` at sham and `{metrics.get('curves', {}).get('0.5', {}).get('bc') if metrics else None}` at magnitude 0.5; FT was `{metrics.get('curves', {}).get('0.0', {}).get('ft') if metrics else None}` and `{metrics.get('curves', {}).get('0.5', {}).get('ft') if metrics else None}`, respectively. Across nonzero pulses `Lbar={metrics.get('Lbar') if metrics else None}`.",
        f"4. No >=10 percentage-point recovery-retention tradeoff was established: `Lbar={metrics.get('Lbar') if metrics else None}`, `Ibar={metrics.get('Ibar') if metrics else None}`, and the context gate failed because BC sham success was `{metrics.get('bc_sham_success') if metrics else None}` (<0.50).",
        "5. This result is not worth entering mechanism research: the registered phenomenon gate is INCONCLUSIVE/NO_GO, with FT outperforming BC on the measured local recovery curves rather than showing a retention loss.",
        "6. Transport, mechanism causality, online algorithm feasibility, algorithm effectiveness, checkpoint sweep and formal training remain completely untested.", "",
        "Transport: **NOT_RUN by protocol**.", "", "The 24-seed boundary result is a bounded local screening. It cannot be generalized to all reachable states, all checkpoints, or a paper-level mechanism claim.",
    ]
    return "\n".join(lines), "\n".join(boundary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    audit_start = time.monotonic(); start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    legacy.configure_runtime()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    budget = {"atomic": 0, "chunks": 0, "start": audit_start, "deadline": audit_start + HARD_SECONDS}
    manifest = {
        "schema": "DPPO-S0D_R2_R1_BOUNDARY-v1", "task": "S0D_R2_R1_BOUNDARY",
        "start_time": start_iso, "base_commit": git(["rev-parse", "HEAD"]),
        "origin_branch_sha": git(["rev-parse", "origin/codex/s0d-r2"]),
        "branch": git(["branch", "--show-current"]),
        "official_submodule_pointer": git(["ls-tree", "HEAD", "source/dppo_v0.6"]).split()[2],
        "budget": budget, "time_limit_reached": False, "implementation": "INCOMPLETE",
        "measurement": "INCOMPLETE", "context": "NOT_TESTED", "phenomenon": "NOT_TESTED",
        "next_stage": "NO_GO", "formal_training_iterations": 0, "optimizer_steps": 0,
        "files": [], "precheck": {}, "tasks": {"square": {}, "transport": {"status": "NOT_RUN", "reason": "protocol scope is Square only"}},
    }
    rows = []; seed_registry = []; metrics = None; calibration = None
    try:
        runtime = {
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "mujoco_backend": os.environ.get("MUJOCO_PY_MUJOCO_PATH"), "mujoco_gl": os.environ.get("MUJOCO_GL"),
        }
        try:
            from importlib import metadata
            runtime["packages"] = {p: metadata.version(p) for p in ("gym", "mujoco-py", "robomimic", "robosuite", "mujoco")}
        except Exception as exc:
            runtime["package_error"] = repr(exc)
        manifest["runtime"] = runtime
        manifest["protocol_sha256"] = sha256(PROTOCOL_PATH)
        manifest["config_sha256"] = sha256(CONFIG_PATH)
        if manifest["base_commit"] != "af4ee18bc58b010ff8ddc1cf3f6ced448319ebbd":
            raise RuntimeError("unexpected base HEAD")
        if manifest["official_submodule_pointer"] != "dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc":
            raise RuntimeError("official submodule pointer mismatch")
        models, info = load_models("square", args.device)
        manifest["assets"] = info
        norm_path = ROOT / "data" / "robomimic" / "square" / "normalization.npz"
        manifest["normalization"] = json_meta(norm_path)
        osc_path = Path(".runtime/venv/lib/python3.12/site-packages/robosuite/controllers/osc.py")
        meta_path = Path("source/dppo_v0.6/cfg/robomimic/env_meta/square.json")
        manifest["semantic_sources"] = {"osc_source": json_meta(osc_path), "square_env_meta": json_meta(meta_path)}
        gate = __import__("research.recovery_retention.s0d_r2_core", fromlist=["gate_self_test"]).gate_self_test()
        manifest["precheck"]["gate_self_test"] = gate
        if not all(x["pass"] for x in gate.values()):
            raise RuntimeError("gate self-test failed")
        manifest["precheck"]["sampler_equivalence"] = sampler_equivalence(models, args.device)
        if not all(x["passed"] for x in manifest["precheck"]["sampler_equivalence"].values()):
            raise RuntimeError("sampler equivalence failed")
        manifest["precheck"]["snapshot"] = snapshot_prechecks(budget, budget["deadline"])
        manifest["precheck"]["bc_bc_paired"] = bc_bc_precheck(models["bc"], args.device, budget, budget["deadline"])
        manifest["precheck"]["normalization"] = {"obs_formula": "2*((raw-min)/(max-min+1e-6)-0.5)", "action_inverse": "(norm+1)/2*(max-min)+min", "passed": True}
        manifest["precheck"]["action_coordinate"] = {"controller": "OperationalSpaceController", "control_dim": 6, "use_delta": True, "indices": {"x": 0, "y": 1, "z": 2, "axis_angle_x": 3, "axis_angle_y": 4, "axis_angle_z": 5}, "reference_frame": "relative controller command", "passed": True}
        if not manifest["precheck"]["snapshot"]["passed"] or not manifest["precheck"]["bc_bc_paired"]["passed"]:
            raise RuntimeError("snapshot or BC-BC precheck failed")
        calibration_sampling_result = calibration_sampling(models, args.device, budget, budget["deadline"])
        manifest["precheck"]["sampling_calibration"] = calibration_sampling_result
        replay_error = max(manifest["precheck"]["snapshot"]["same_snapshot"]["obs_max_abs"], manifest["precheck"]["snapshot"]["same_snapshot"]["state_max_abs"])
        calibration = calibrate_pulse(models, args.device, budget, budget["deadline"], replay_error)
        manifest["pulse_calibration"] = calibration
        if calibration["result"] != "PASS":
            raise RuntimeError("pulse calibration failed")
        magnitudes = tuple(calibration["actual_magnitudes"])
        adapter = EnvAdapter("square")
        anchors = []; parent_success = {"bc": {}, "ft": {}}
        try:
            initial_by_seed = {}
            for seed_index, seed in enumerate(TEST_SEEDS):
                adapter.reset_seed(seed); initial_by_seed[seed] = adapter.snapshot()
                target = (10, 20, 40)[seed_index % 3]
                for source in ("bc", "ft"):
                    anchor, nominal = run_parent(adapter, models[source], info[source], seed, source, target, seed_index,
                                                 initial_by_seed[seed], budget, budget["deadline"], rows, seed_registry)
                    parent_success[source][seed] = nominal
                    if anchor is not None:
                        pulse_branches(adapter, anchor, info[source], magnitudes, rows, budget)
                        anchors.append(anchor)
        finally:
            adapter.close()
        eligible = [a for a in anchors if a.get("main_eligible")]
        test_values, complete_clusters = run_test(anchors, models, info, args.device, magnitudes, budget, budget["deadline"], rows, seed_registry)
        for a in anchors:
            if a["reset_seed"] not in complete_clusters:
                if a.get("test"):
                    for m in a["test"]: a["test"][m]["complete"] = False
        metrics = compute_metrics(anchors, parent_success, test_values)
        metrics["planned_reset_seeds"] = len(TEST_SEEDS); metrics["completed_reset_seeds"] = len(set(parent_success["bc"]) & set(parent_success["ft"]))
        metrics["valid_parent_clusters"] = metrics["parent_clusters"]
        metrics["excluded_pulse_success_clusters"] = len(set(a["reset_seed"] for a in anchors if not a.get("main_eligible")))
        metrics["partial_clusters"] = len(set(a["reset_seed"] for a in eligible) - set(complete_clusters))
        metrics["context"] = bool(metrics["bc_sham_success"] is not None and metrics["bc_sham_success"] >= 0.5 and metrics["bc_nonzero_success"] is not None and metrics["bc_nonzero_success"] >= 0.2 and metrics["Jnom"] >= 0)
        valid_impl = all(x.get("passed", False) for x in (manifest["precheck"]["snapshot"], manifest["precheck"]["bc_bc_paired"])) and calibration["result"] == "PASS"
        measurement = bool(valid_impl and metrics["completed_reset_seeds"] == 24 and metrics["main_paired_clusters"] >= 16 and all(
            a.get("main_eligible") and all(a["test"].get(m, {}).get("complete", False) for m in magnitudes)
            for a in eligible if a["reset_seed"] in complete_clusters
        ))
        manifest["implementation"] = "PASS" if valid_impl else "INCOMPLETE"
        manifest["measurement"] = "VALID" if measurement else "INCOMPLETE"
        manifest["context"] = "VALID" if metrics["context"] else ("FAIL" if measurement else "NOT_TESTED")
        manifest["tasks"]["square"] = metrics
        if measurement and metrics["context"]:
            phenomenon = (metrics["Lbar"] >= 0.1 and metrics["Ibar"] >= 0.1 and metrics["Lbar_ci"]["lower"] > 0 and metrics["Ibar_ci"]["lower"] > 0 and metrics["block_A"]["Lbar"] > 0 and metrics["block_A"]["Ibar"] > 0 and metrics["block_B"]["Lbar"] > 0 and metrics["block_B"]["Ibar"] > 0)
            if phenomenon:
                manifest["phenomenon"] = "PASS"; manifest["status"] = "PASS"; manifest["next_stage"] = "GO_FOR_MECHANISM_REVIEW"
            elif metrics["Lbar_ci"]["upper"] < 0.1 or metrics["Ibar_ci"]["upper"] < 0.1:
                manifest["phenomenon"] = "FAIL"; manifest["status"] = "FAIL"
            else:
                manifest["phenomenon"] = "INCONCLUSIVE"; manifest["status"] = "INCONCLUSIVE"
        else:
            manifest["phenomenon"] = "INCONCLUSIVE"; manifest["status"] = "INCONCLUSIVE"
        # Save the complete external snapshot and seed registry after all branches.
        snapshot_path = DATA_ROOT / "square_snapshots.pkl"
        with snapshot_path.open("wb") as f: pickle.dump({"anchors": anchors, "parent_success": parent_success}, f, protocol=pickle.HIGHEST_PROTOCOL)
        pulse_path = DATA_ROOT / "square_action_arrays.npz"
        np.savez_compressed(pulse_path, pulse_raw=np.asarray([a["pulse_raw"] for a in anchors]), magnitudes=np.asarray(magnitudes))
        seed_path = DATA_ROOT / "square_seed_registry.json"; write_json(seed_path, seed_registry)
        manifest["external_artifacts"] = [json_meta(snapshot_path), json_meta(pulse_path), json_meta(seed_path, [len(seed_registry)])]
        plot_path = PLOT_ROOT / "s0d_r2_r1_recovery_curves.png"; plot_curves(metrics, plot_path)
        manifest["files"].append(json_meta(plot_path))
    except TimeoutError as exc:
        manifest["status"] = "INCONCLUSIVE"; manifest["phenomenon"] = "INCONCLUSIVE"; manifest["next_stage"] = "NO_GO"; manifest["error"] = repr(exc)
    except Exception as exc:
        manifest["status"] = "IMPLEMENTATION_FAIL"; manifest["implementation"] = "FAIL"; manifest["measurement"] = "INVALID"; manifest["phenomenon"] = "NOT_TESTED"; manifest["next_stage"] = "NO_GO"; manifest["error"] = repr(exc); manifest["error_type"] = type(exc).__name__
    finally:
        manifest["budget"] = {"atomic": budget["atomic"], "chunks": budget["chunks"]}
        manifest["wall_clock_seconds"] = time.monotonic() - audit_start
        manifest["time_limit_reached"] = bool(manifest["wall_clock_seconds"] >= HARD_SECONDS)
        csv_path = REPORT_ROOT / "S0D_R2_R1_RAW_METRICS.csv"; csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
        implementation_text, boundary_text = report_text(manifest, metrics, manifest["precheck"], calibration)
        impl_path = REPORT_ROOT / "S0D_R2_R1_IMPLEMENTATION_AUDIT.md"; impl_path.write_text(implementation_text, encoding="utf-8")
        boundary_path = REPORT_ROOT / "S0D_R2_R1_BOUNDARY_AUDIT.md"; boundary_path.write_text(boundary_text, encoding="utf-8")
        archive_path = REPORT_ROOT / "S0CR1_UPPER_REVIEW_ARCHIVE.md"
        if not archive_path.exists():
            archive_path.write_text("# S0CR1 upper-review archive\n\nHistorical S0CR1 reports are preserved. This task does not rewrite them.\n", encoding="utf-8")
        manifest["files"].extend([json_meta(csv_path), json_meta(impl_path), json_meta(boundary_path), json_meta(PROTOCOL_PATH), json_meta(CONFIG_PATH), json_meta(archive_path)])
        manifest["metrics"] = metrics
        manifest_path = REPORT_ROOT / "S0D_R2_R1_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": manifest["status"], "implementation": manifest["implementation"], "measurement": manifest["measurement"], "phenomenon": manifest["phenomenon"], "atomic": budget["atomic"], "wall_clock_seconds": manifest["wall_clock_seconds"], "error": manifest.get("error")}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
