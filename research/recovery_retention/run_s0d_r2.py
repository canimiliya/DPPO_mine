"""S0d-R2: bounded native recovery-retention screening.

This runner is diagnostic-only.  It loads audited frozen actors, evaluates a
fixed pulse and continuation protocol in the legacy robomimic/mujoco-py
environment, and never constructs an optimizer or performs training.
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
SOURCE = ROOT / "source" / "dppo_v0.6"
DATA_ROOT = Path("/mnt/d/AgentData/DPPO-S0d-R2")
S0A = ROOT / "reports" / "research" / "S0A_ASSET_MANIFEST.json"
EXPECTED_HASHES = {
    ("square", 0): "c9e22005b58ba2524782e3068c8341554e419acf66f095e238e56df41d9a7aa3",
    ("square", 200): "18031b2ed2ee422f6e85bd8c113e199d46e0b29f3f2b2d6b4828e67449e0cf25",
    ("transport", 0): "48f6c0bf65fe12f000087b25bf5e336db03a669687dd5cf6363abf80b3d112ae",
    ("transport", 200): "b4b828312915442ede54a672f74a6276fe407ea3cf3b3830ba20bcb734d8d592",
}
TASKS = {
    "square": {"obs": 23, "action": 7, "chunk": 4, "horizon": 400,
                "keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]},
    "transport": {"obs": 59, "action": 14, "chunk": 8, "horizon": 800,
                   "keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                            "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", "object"]},
}
RESET_SEEDS = tuple(range(920001, 920017))
PILOT_SEEDS = (910001, 910002)
N_SCREEN = 4
N_TEST = 4
MAX_STEPS = 1_100_000
HARD_SECONDS = 240 * 60

if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.recovery_retention.s0d_r2_core import (  # noqa: E402
    gate_self_test,
    gate_task,
    mean_ci,
    stable_seed,
)
from research.denoising_budget.collect_frozen import load_frozen_model  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def find_checkpoint(task: str, itr: int):
    m = json.loads(S0A.read_text(encoding="utf-8"))
    if task == "square":
        base = Path(m["square_selected_ft_checkpoint"]["path"])
        if itr == 0:
            path = base.with_name("state_0.pt")
            source = "S0A square_selected_ft_checkpoint sibling derived from manifest path"
        else:
            path = base
            source = "S0A square_selected_ft_checkpoint"
    else:
        matches = [x for x in m["transport_checkpoints"] if int(x.get("itr", -1)) == itr]
        if not matches:
            raise FileNotFoundError(f"transport itr={itr} absent from S0A manifest")
        path = Path(matches[0]["path"])
        source = "S0A transport_checkpoints"
    return path, source


def configure_runtime():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", "/mnt/d/Desktop/my_project/paper_reproduction/DPPO/.runtime/mujoco210")
    os.environ.setdefault("LD_LIBRARY_PATH", "/mnt/d/Desktop/my_project/paper_reproduction/DPPO/.runtime/mujoco210/bin")


def load_models(task: str, device: str):
    paths = {}
    models = {}
    for label, itr in (("bc", 0), ("ft", 200)):
        path, source = find_checkpoint(task, itr)
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        expected = EXPECTED_HASHES[(task, itr)]
        if actual != expected:
            raise RuntimeError(f"{task} state_{itr} hash mismatch: {actual} != {expected}")
        model, cfg, _ = load_frozen_model(task, "bc", device=device)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        paths[label] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual, "itr": itr, "source": source}
        models[label] = model
    # The base actor is expected to be unchanged between state_0 and state_200.
    s0 = torch.load(Path(paths["bc"]["path"]), map_location="cpu", weights_only=True)["model"]
    s2 = torch.load(Path(paths["ft"]["path"]), map_location="cpu", weights_only=True)["model"]
    diffs = []
    for k, v in s0.items():
        if k.startswith("actor.") and k in s2 and tuple(v.shape) == tuple(s2[k].shape):
            diffs.append(float((v.float() - s2[k].float()).abs().max()))
    paths["base_actor_state0_vs_state200"] = {"matched": len(diffs), "max_abs_diff": max(diffs) if diffs else None,
                                               "exact": bool(diffs) and max(diffs) == 0.0}
    return models, paths


class EnvAdapter:
    def __init__(self, task: str):
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils

        self.task = task
        spec = TASKS[task]
        ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": spec["keys"]})
        meta_path = SOURCE / "cfg" / "robomimic" / "env_meta" / f"{task}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.env = EnvUtils.create_env_from_metadata(env_meta=meta, render=False,
                                                      render_offscreen=False, use_image_obs=False)
        self.env.env.hard_reset = False
        self.raw = self.env.env
        self.norm_path = ROOT / "data" / "robomimic" / task / "normalization.npz"
        norm = np.load(self.norm_path)
        self.obs_min, self.obs_max = norm["obs_min"].copy(), norm["obs_max"].copy()
        self.action_min, self.action_max = norm["action_min"].copy(), norm["action_max"].copy()
        self.action_low, self.action_high = [np.asarray(x).copy() for x in self.raw.action_spec]
        native_action_dim = int(getattr(self.raw, "action_dimension", getattr(self.raw, "_action_dim", -1)))
        if native_action_dim != spec["action"]:
            raise RuntimeError(f"{task} native action dim {native_action_dim} != {spec['action']}")
        if not hasattr(self.raw.robots[0].controller, "control_dim") or int(self.raw.robots[0].controller.control_dim) != 6:
            raise RuntimeError("robot0 controller is not 6D OSC pose")
        self.obs_history = []
        self.action_queue = []
        self.atomic_steps = 0
        self.chunk_steps = 0

    def close(self):
        self.raw.close()

    def _norm(self, raw_obs):
        value = np.concatenate([np.asarray(raw_obs[k]) for k in TASKS[self.task]["keys"]], axis=0)
        value = 2 * ((value - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)
        return value.astype(np.float32)

    def current_obs(self):
        return self._norm(self.env.get_observation())

    def reset_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        obs = self.env.reset()
        self.obs_history = [self._norm(obs)]
        self.action_queue = []
        self.atomic_steps = 0
        self.chunk_steps = 0
        return self.obs_history[-1].copy()

    @staticmethod
    def _copy_value(value):
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, (float, int, bool, str)) or value is None:
            return value
        return copy.deepcopy(value)

    def snapshot(self):
        state = self.env.get_state()
        controller_state = []
        controller_fields = ("goal_pos", "goal_ori", "relative_ori", "new_update", "ee_pos", "ee_ori_mat",
                             "ee_pos_vel", "ee_ori_vel", "joint_pos", "joint_vel", "J_pos", "J_ori", "J_full",
                             "mass_matrix", "torques", "initial_joint", "initial_ee_pos", "initial_ee_ori_mat")
        for robot in self.raw.robots:
            c = robot.controller
            controller_state.append({k: self._copy_value(getattr(c, k)) for k in controller_fields if hasattr(c, k)})
        return {
            "model": state["model"], "states": np.asarray(state["states"]).copy(),
            "controller_state": controller_state,
            "raw_attrs": {k: self._copy_value(getattr(self.raw, k)) for k in ("cur_time", "timestep", "done") if hasattr(self.raw, k)},
            "obs_cache": copy.deepcopy(getattr(self.raw, "_obs_cache", {})),
            "obs_history": [x.copy() for x in self.obs_history],
            "action_queue": [np.asarray(x).copy() for x in self.action_queue],
            "wrapper_time": int(self.atomic_steps), "wrapper_chunk_steps": int(self.chunk_steps),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def restore(self, snap, full=False):
        if full:
            self.env.reset_to({"model": snap["model"], "states": snap["states"]})
        else:
            self.raw.sim.set_state_from_flattened(np.asarray(snap["states"]))
            self.raw.sim.forward()
        for key, value in snap["raw_attrs"].items():
            setattr(self.raw, key, self._copy_value(value))
        for robot, saved in zip(self.raw.robots, snap["controller_state"]):
            for key, value in saved.items():
                setattr(robot.controller, key, self._copy_value(value))
        self.raw._obs_cache = copy.deepcopy(snap["obs_cache"])
        self.obs_history = [x.copy() for x in snap["obs_history"]]
        self.action_queue = [x.copy() for x in snap["action_queue"]]
        self.atomic_steps = int(snap["wrapper_time"])
        self.chunk_steps = int(snap["wrapper_chunk_steps"])
        random.setstate(snap["python_rng"])
        np.random.set_state(snap["numpy_rng"])
        torch.set_rng_state(snap["torch_rng"])
        if torch.cuda.is_available() and snap["torch_cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(snap["torch_cuda_rng"])

    def native_success(self):
        return bool(self.raw._check_success())

    def state_vector(self):
        return np.asarray(self.raw.sim.get_state().flatten()).copy()

    def step_atomic(self, action):
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=np.float64))
        normalized = self._norm(obs)
        self.obs_history.append(normalized.copy())
        self.action_queue.append(np.asarray(action, dtype=np.float64).copy())
        self.atomic_steps += 1
        return normalized, float(reward), bool(done), self.native_success(), dict(info)

    def unnormalize(self, action):
        action = np.asarray(action, dtype=np.float64)
        raw = (action + 1) / 2 * (self.action_max - self.action_min) + self.action_min
        return np.clip(raw, self.action_low, self.action_high)


def sample_chunk(model, obs: np.ndarray, device: str, seed: int):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    cond = {"state": torch.from_numpy(obs[None, None, :]).float().to(device)}
    with torch.no_grad():
        sample = model(cond=cond, deterministic=False, return_chain=True)
    return sample.trajectories[0].detach().cpu().numpy()


def save_file_meta(path: Path, shape=None):
    wsl_path = str(path)
    win_path = "D:\\" + wsl_path[len("/mnt/d/"):].replace("/", "\\") if wsl_path.startswith("/mnt/d/") else wsl_path.replace("/", "\\")
    return {"path_windows": win_path, "path_wsl": wsl_path,
            "bytes": path.stat().st_size, "sha256": sha256(path), "shape": list(shape) if shape is not None else None}


def sampler_precheck(models, device):
    """Compare fixed-tape replay with the official model sampler for both actors."""
    results = {}
    for label, model in models.items():
        torch.manual_seed(stable_seed("sampler", label))
        # A model's actual horizon/action dimensions are inferred from its attributes.
        xk = torch.randn((2, int(model.horizon_steps), int(model.action_dim)), device=device)
        noise = torch.randn((2, int(model.denoising_steps), int(model.horizon_steps), int(model.action_dim)), device=device)
        cond = {"state": torch.randn((2, 1, int(model.obs_dim)), device=device)}
        def replay():
            x = xk.clone(); chain = []
            for i, t in enumerate(reversed(range(model.denoising_steps))):
                tb = torch.full((2,), t, dtype=torch.long, device=device)
                mean, logvar, _ = model.p_mean_var(x=x, t=tb, cond=cond, index=None, use_base_policy=False, deterministic=False)
                std = torch.clip(torch.exp(0.5 * logvar), min=model.get_min_sampling_denoising_std())
                x = mean + std * noise[:, i].clamp(-model.randn_clip_value, model.randn_clip_value)
                if model.final_action_clip_value is not None and i == model.denoising_steps - 1:
                    x = torch.clamp(x, -model.final_action_clip_value, model.final_action_clip_value)
                if t <= model.ft_denoising_steps:
                    chain.append(x.clone())
            return x, torch.stack(chain, dim=1)
        import unittest.mock as mock
        noises = [xk.clone()] + [noise[:, i].clone() for i in range(noise.shape[1])]
        def fake_randn(*args, **kwargs):
            return noises.pop(0).clone()
        def fake_randn_like(x, *args, **kwargs):
            return noises.pop(0).clone()
        with torch.no_grad():
            ro, rc = replay()
            with mock.patch.object(torch, "randn", side_effect=fake_randn), mock.patch.object(torch, "randn_like", side_effect=fake_randn_like):
                official = model(cond=cond, deterministic=False, return_chain=True)
        if noises:
            raise RuntimeError("official sampler did not consume fixed tape")
        results[label] = {"action_max_abs": float((ro - official.trajectories).abs().max()),
                          "chain_max_abs": float((rc - official.chains).abs().max())}
    return results


def env_precheck(adapter: EnvAdapter, seed: int):
    adapter.reset_seed(seed)
    snap = adapter.snapshot()
    rng = np.random.default_rng(stable_seed("snapshot", adapter.task))
    tape = rng.uniform(adapter.action_low, adapter.action_high, size=(8, len(adapter.action_low)))
    def replay_once():
        adapter.restore(snap, full=True)
        obs, states, flags = [], [], []
        for action in tape:
            o, _, done, success, _ = adapter.step_atomic(action)
            obs.append(o); states.append(adapter.state_vector()); flags.append((done, success))
        return np.asarray(obs), np.asarray(states), flags
    a = replay_once(); b = replay_once()
    obs_err = float(np.max(np.abs(a[0] - b[0])))
    state_err = float(np.max(np.abs(a[1] - b[1])))
    flags_equal = a[2] == b[2]
    return {"passed": bool(obs_err <= 1e-6 and state_err <= 1e-6 and flags_equal),
            "obs_max_abs": obs_err, "state_max_abs": state_err, "done_success_equal": flags_equal,
            "snapshot_fields": sorted(snap.keys()), "tape_shape": list(tape.shape)}


def action_direction(adapter: EnvAdapter, direction: str):
    idx = {"+x": 0, "-x": 0, "+y": 1, "-y": 1}[direction]
    sign = 1.0 if direction in ("+x", "+y") else -1.0
    if idx >= 3 or int(adapter.raw.robots[0].controller.control_dim) != 6:
        raise RuntimeError("robot0 Cartesian translation action coordinates not confirmed")
    return idx, sign


def run_chunk(adapter, raw_action, pulse=None):
    actions = np.asarray(raw_action, dtype=np.float64).copy()
    actual = actions.copy()
    if pulse is not None:
        idx, sign = pulse
        for t in range(min(4, len(actual))):
            actual[t, idx] += sign * 0.25
        actual = np.clip(actual, adapter.action_low, adapter.action_high)
    successes = []
    done = False
    rewards = []
    before = adapter.atomic_steps
    for action in actual:
        _, reward, done_i, success, _ = adapter.step_atomic(action)
        rewards.append(reward); done = done or done_i; successes.append(success)
    return {"actions": actual, "delta": actual - actions, "success": bool(any(successes)),
            "done": bool(done), "actual_steps": int(adapter.atomic_steps - before), "rewards": rewards}


def parent_rollout(adapter, model, task, source, checkpoint_hash, reset_seed, target_chunk, anchor_index, device, rows, anchors, deadline, budget, initial=None):
    if initial is None:
        adapter.reset_seed(reset_seed)
        initial = adapter.snapshot()
    else:
        adapter.restore(initial, full=False)
    success_seen = False
    anchor = None
    total_chunks = TASKS[task]["horizon"] // TASKS[task]["chunk"]
    for chunk_i in range(total_chunks):
        if time.monotonic() > deadline or budget["atomic"] >= MAX_STEPS:
            raise TimeoutError("budget/deadline reached during parent rollout")
        obs = adapter.current_obs()
        norm_action = sample_chunk(model, obs, device, stable_seed(task, reset_seed, source, "parent", chunk_i))
        raw_action = adapter.unnormalize(norm_action[: TASKS[task]["chunk"]])
        out = run_chunk(adapter, raw_action)
        budget["atomic"] += out["actual_steps"]; budget["chunks"] += 1
        success_seen = success_seen or out["success"] or adapter.native_success()
        if chunk_i + 1 == target_chunk:
            if not success_seen and not out["done"]:
                direction = ("+x", "-x", "+y", "-y")[anchor_index % 4]
                idx, sign = action_direction(adapter, direction)
                pulse_tape = sample_chunk(model, adapter.current_obs(), device,
                                          stable_seed(task, reset_seed, source, "anchor", target_chunk))
                pulse_raw = adapter.unnormalize(pulse_tape[: TASKS[task]["chunk"]])
                snap = adapter.snapshot()
                anchor = {"task": task, "reset_seed": reset_seed, "source_policy": source,
                          "boundary": target_chunk, "pulse_direction": direction,
                          "pulse_index": idx, "pulse_sign": sign, "anchor_index": anchor_index,
                          "parent_atomic_step": adapter.atomic_steps, "remaining_before_pulse": TASKS[task]["horizon"] - adapter.atomic_steps,
                          "snapshot": snap, "pulse_raw": pulse_raw,
                          "snapshot_id": f"{task}-seed{reset_seed}-{source}-b{target_chunk}"}
                anchors.append(anchor)
            else:
                rows.append(row(task, checkpoint_hash, reset_seed, source, target_chunk, "", "parent", "parent", "", -1,
                                 bool(success_seen), "anchor_invalid_success_or_done", TASKS[task]["horizon"] - adapter.atomic_steps,
                                 out["actual_steps"], "", ""))
    rows.append(row(task, checkpoint_hash, reset_seed, source, target_chunk, "", "parent", "parent", "", -1,
                    bool(success_seen), "horizon_or_natural_termination", 0, int(adapter.atomic_steps), "", ""))
    return initial, anchor, bool(success_seen)


def row(task, checkpoint_hash, reset_seed, source_policy, boundary, direction, condition, phase, block, draw,
        native_success, reason, remaining, steps, snapshot_id, pulse_id, actual_delta=0.0):
    return {"task": task, "checkpoint_hash": checkpoint_hash, "reset_seed": reset_seed, "source_policy": source_policy,
            "boundary": boundary, "pulse_direction": direction, "condition": condition,
            "screening_or_test": phase, "block": block, "draw": draw, "native_success": int(bool(native_success)),
            "termination_reason": reason, "remaining_horizon": int(remaining), "actual_steps": int(steps),
            "snapshot_id": snapshot_id, "pulse_id": pulse_id, "actual_action_delta": actual_delta}


def continuation(adapter, model, anchor_snap, policy, task, seed, deadline, budget):
    adapter.restore(anchor_snap, full=False)
    success = adapter.native_success()
    if success:
        return {"success": True, "reason": "success_at_pulse", "steps": 0, "remaining": TASKS[task]["horizon"] - adapter.atomic_steps}
    start = adapter.atomic_steps
    max_steps = TASKS[task]["horizon"]
    chunk = TASKS[task]["chunk"]
    model_steps = 0
    while adapter.atomic_steps < max_steps:
        if time.monotonic() > deadline or budget["atomic"] >= MAX_STEPS:
            raise TimeoutError("budget/deadline reached during continuation")
        obs = adapter.current_obs()
        norm = sample_chunk(model, obs, "cuda:0" if next(model.parameters()).is_cuda else "cpu", seed + model_steps)
        raw = adapter.unnormalize(norm[:chunk])
        out = run_chunk(adapter, raw)
        budget["atomic"] += out["actual_steps"]; budget["chunks"] += 1; model_steps += 1
        if out["success"] or adapter.native_success():
            return {"success": True, "reason": "native_success", "steps": adapter.atomic_steps - start,
                    "remaining": max_steps - adapter.atomic_steps}
        if out["done"]:
            return {"success": False, "reason": "environment_done", "steps": adapter.atomic_steps - start,
                    "remaining": max_steps - adapter.atomic_steps}
    return {"success": False, "reason": "episode_horizon", "steps": adapter.atomic_steps - start,
            "remaining": 0}


def pulse_branches(adapter, anchor, checkpoint_hash, rows, budget):
    branches = {}
    for condition, pulse in (("sham", None), ("perturbed", (anchor["pulse_index"], anchor["pulse_sign"]))):
        adapter.restore(anchor["snapshot"], full=False)
        out = run_chunk(adapter, anchor["pulse_raw"], pulse=pulse)
        budget["atomic"] += out["actual_steps"]; budget["chunks"] += 1
        branch_snap = adapter.snapshot()
        branches[condition] = branch_snap
        delta = float(np.max(np.abs(out["delta"])))
        rows.append(row(anchor["task"], checkpoint_hash, anchor["reset_seed"], anchor["source_policy"], anchor["boundary"],
                        anchor["pulse_direction"], condition, "pulse", "", -1, out["success"],
                        "success_or_done_during_pulse" if out["success"] or out["done"] else "pulse_complete",
                        TASKS[anchor["task"]]["horizon"] - adapter.atomic_steps, out["actual_steps"],
                        anchor["snapshot_id"], anchor["snapshot_id"], delta))
    anchor["branches"] = branches
    return branches


def screen_task(task, models, model_info, device, start, deadline, budget, outdir, preflight):
    adapter = EnvAdapter(task)
    rows = []
    anchors = []
    parent_success = {"bc": {}, "ft": {}}
    initial_snapshots = {}
    boundary_by_index = lambda i: 20 if i % 2 == 0 else 40
    try:
        pre = preflight
        for source in ("bc", "ft"):
            model = models[source]
            for i, reset_seed in enumerate(RESET_SEEDS):
                target = boundary_by_index(i)
                before = len(anchors)
                initial, anchor, nominal = parent_rollout(adapter, model, task, source, model_info[source]["sha256"], reset_seed, target, i, device, rows, anchors, deadline, budget,
                                                          initial=initial_snapshots.get(reset_seed))
                initial_snapshots.setdefault(reset_seed, initial)
                parent_success[source][reset_seed] = nominal
                if anchor is not None:
                    pulse_branches(adapter, anchor, model_info[source]["sha256"], rows, budget)
                    for condition in ("sham", "perturbed"):
                        successes = 0
                        for draw in range(N_SCREEN):
                            seed = stable_seed(task, reset_seed, source, target, condition, "screen", draw)
                            result = continuation(adapter, models["bc"], anchor["branches"][condition], "bc", task, seed, deadline, budget)
                            successes += int(result["success"])
                            rows.append(row(task, model_info["bc"]["sha256"], reset_seed, source, target, anchor["pulse_direction"],
                                             f"bc_{condition}", "screening", "", draw, result["success"], result["reason"],
                                             result["remaining"], result["steps"], anchor["snapshot_id"], anchor["snapshot_id"]))
                        anchor.setdefault("screen", {})[condition] = successes
                    anchor["in_C"] = anchor["screen"]["sham"] >= 3 and anchor["screen"]["perturbed"] >= 3
                elif len(anchors) == before:
                    pass
        c = [a for a in anchors if a.get("in_C", False)]
        coverage = {"cases": len(c), "clusters": len(set(a["reset_seed"] for a in c)),
                    "bc_source": sum(a["source_policy"] == "bc" for a in c),
                    "ft_source": sum(a["source_policy"] == "ft" for a in c)}
        coverage["pass"] = coverage["cases"] >= 12 and coverage["clusters"] >= 8 and coverage["bc_source"] >= 4 and coverage["ft_source"] >= 4
        test = {}
        if coverage["pass"]:
            for a in c:
                test[a["snapshot_id"]] = {}
                for condition, policy in (("sham", "bc"), ("sham", "ft"), ("perturbed", "bc"), ("perturbed", "ft")):
                    key = f"{policy}_{condition}"
                    test[a["snapshot_id"]][key] = []
                    for draw in range(N_TEST):
                        block = "A" if draw < 2 else "B"
                        seed = stable_seed(task, a["reset_seed"], a["source_policy"], a["boundary"], key, "test", block, draw)
                        result = continuation(adapter, models[policy], a["branches"][condition], policy, task, seed, deadline, budget)
                        test[a["snapshot_id"]][key].append(int(result["success"]))
                        rows.append(row(task, model_info[policy]["sha256"], a["reset_seed"], a["source_policy"], a["boundary"],
                                         a["pulse_direction"], key, "test", block, draw, result["success"], result["reason"],
                                         result["remaining"], result["steps"], a["snapshot_id"], a["snapshot_id"]))
        metrics = compute_metrics(c, test, parent_success, task)
        metrics["precheck"] = pre
        metrics["coverage"] = coverage
        metrics["anchors"] = len(anchors)
        metrics["screening_records"] = sum(1 for r in rows if r["screening_or_test"] == "screening")
        metrics["test_records"] = sum(1 for r in rows if r["screening_or_test"] == "test")
        metrics["implementation_pass"] = bool(pre["passed"])
        metrics["measurement_valid"] = bool(pre["passed"] and coverage["pass"] and metrics.get("complete_test", False))
        metrics["coverage_pass"] = bool(coverage["pass"])
        metrics["gate"] = gate_task(metrics)["gate"] if metrics["implementation_pass"] else "IMPLEMENTATION_FAIL"
        pickle_path = outdir / f"{task}_snapshots.pkl"
        with pickle_path.open("wb") as f:
            pickle.dump({"anchors": anchors, "parent_success": parent_success}, f, protocol=pickle.HIGHEST_PROTOCOL)
        tapes = {"snapshot_id": np.asarray([a["snapshot_id"] for a in anchors]),
                 "pulse_raw": np.asarray([a["pulse_raw"] for a in anchors], dtype=np.float64) if anchors else np.empty((0, TASKS[task]["chunk"], TASKS[task]["action"]))}
        tape_path = outdir / f"{task}_action_tapes.npz"
        np.savez_compressed(tape_path, **tapes)
        metrics["artifact_files"] = [save_file_meta(pickle_path), save_file_meta(tape_path, tapes["pulse_raw"].shape)]
        return metrics, rows, anchors, parent_success
    finally:
        adapter.close()


def compute_metrics(cases, test, parent_success, task):
    result = {"L": None, "L0": None, "I": None, "Jnom": None, "L_ci": {}, "I_ci": {},
              "L_A": None, "L_B": None, "complete_test": False}
    if not cases or not test:
        result["Jnom"] = float(np.mean([int(parent_success["ft"].get(s, False)) - int(parent_success["bc"].get(s, False)) for s in RESET_SEEDS])) if parent_success["bc"] else None
        return result
    keys = [a["snapshot_id"] for a in cases]
    required = ("bc_perturbed", "ft_perturbed", "bc_sham", "ft_sham")
    complete = all(k in test and all(len(test[k].get(x, [])) == 4 for x in required) for k in keys)
    result["complete_test"] = complete
    if not complete:
        return result
    lvals, l0vals, ivals, clusters = [], [], [], []
    la, lb = [], []
    for a in cases:
        t = test[a["snapshot_id"]]
        l = np.mean(t["bc_perturbed"]) - np.mean(t["ft_perturbed"])
        l0 = np.mean(t["bc_sham"]) - np.mean(t["ft_sham"])
        lvals.append(l); l0vals.append(l0); ivals.append(l - l0); clusters.append(a["reset_seed"])
        la.append(np.mean(t["bc_perturbed"][:2]) - np.mean(t["ft_perturbed"][:2]))
        lb.append(np.mean(t["bc_perturbed"][2:]) - np.mean(t["ft_perturbed"][2:]))
    lvals, l0vals, ivals, clusters = map(np.asarray, (lvals, l0vals, ivals, clusters))
    result["L"] = float(np.mean(lvals)); result["L0"] = float(np.mean(l0vals)); result["I"] = float(np.mean(ivals))
    result["L_ci"] = mean_ci(lvals, clusters, seed=stable_seed(task, "L", "bootstrap"))
    result["I_ci"] = mean_ci(ivals, clusters, seed=stable_seed(task, "I", "bootstrap"))
    result["L_A"] = float(np.mean(la)); result["L_B"] = float(np.mean(lb))
    result["bc_pert_success"] = float(np.mean([np.mean(test[k]["bc_perturbed"]) for k in keys]))
    result["bc_sham_success"] = float(np.mean([np.mean(test[k]["bc_sham"]) for k in keys]))
    result["ft_sham_success"] = float(np.mean([np.mean(test[k]["ft_sham"]) for k in keys]))
    result["ft_minus_bc_sham"] = result["ft_sham_success"] - result["bc_sham_success"]
    result["Jnom"] = float(np.mean([int(parent_success["ft"].get(s, False)) - int(parent_success["bc"].get(s, False)) for s in RESET_SEEDS]))
    result["jnom"] = result["Jnom"]
    result["L_ci_lower"] = result["L_ci"]["lower"]; result["L_ci_upper"] = result["L_ci"]["upper"]
    result["I_ci_lower"] = result["I_ci"]["lower"]; result["I_ci_upper"] = result["I_ci"]["upper"]
    return result


def report_text(manifest, task_results):
    lines = ["# S0d-R2 recovery-retention audit", "", "This report is a bounded native-simulator screening, not training or a mechanism intervention.", "",
             "## Final fields", "", "- TASK: **S0d-R2**", "- PRIMARY_DECISION: **B**", f"- STATUS: **{manifest['status']}**",
             "- CURRENT_GRADIENT_BUDGET: **STOP**", "- CURRENT_R1: **STOP**",
             f"- IMPLEMENTATION_AUDIT: **{manifest['implementation_audit']}**", f"- MEASUREMENT_VALIDITY: **{manifest['measurement_validity']}**",
             f"- PHENOMENON_GATE: **{manifest['phenomenon_gate']}**", "- MECHANISM_GATE: **NOT_RUN**",
             "- ONLINE_ALGORITHM_FEASIBILITY: **NOT_TESTED**", "- ALGORITHM_EFFECTIVENESS: **NOT_TESTED**", "- NOVELTY_STATUS: **CANDIDATE_ONLY**",
             f"- NEXT_STAGE: **{manifest['next_stage']}**", f"- NEW_ENVIRONMENT_ATOMIC_STEPS: **{manifest['budget']['atomic']}**",
             f"- NEW_ENVIRONMENT_CHUNK_STEPS: **{manifest['budget']['chunks']}**", "- FORMAL_TRAINING_ITERATIONS: **0**",
             "- OPTIMIZER_STEPS: **0**", f"- WALL_CLOCK_SECONDS: **{manifest['wall_clock_seconds']:.3f}**",
             f"- TIME_LIMIT_REACHED: **{manifest['time_limit_reached']}**", ""]
    for task in ("square", "transport"):
        r = task_results.get(task, {"gate": "NOT_RUN"})
        lines += [f"## {task.title()}", ""]
        if r.get("gate") == "NOT_RUN":
            lines += ["- Gate: **NOT_RUN**", f"- Reason: {r.get('reason', 'upstream Square did not pass')}.", ""]
            continue
        cov = r.get("coverage", {})
        lines += [f"- Gate: **{r.get('gate')}**", f"- Coverage: cases={cov.get('cases')}, clusters={cov.get('clusters')}, BC-source={cov.get('bc_source')}, FT-source={cov.get('ft_source')}, pass={cov.get('pass')}.",
                   f"- L={r.get('L')}; L0={r.get('L0')}; I={r.get('I')}; Jnom={r.get('Jnom')}.",
                   f"- L 95% CI={r.get('L_ci')}; I 95% CI={r.get('I_ci')}.",
                   f"- L_A={r.get('L_A')}; L_B={r.get('L_B')}; BC perturbed={r.get('bc_pert_success')}; BC sham={r.get('bc_sham_success')}; FT sham={r.get('ft_sham_success')}.",
                   f"- Implementation precheck: {r.get('precheck')}.", ""]
    lines += ["## Scope interpretation", "", "The empirical set C is a noisy BC screening set, not a proof of q_BC>=0.75. Any PASS would only request upper review for controlled mechanism design; this run does not authorize training, N1/N2, or other directions.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    configure_runtime()
    start = time.monotonic(); start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    deadline = start + HARD_SECONDS
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    outdir = DATA_ROOT
    manifest = {"schema": "DPPO-S0d-R2-manifest-v1", "start_time": start_iso, "base_commit": git(["rev-parse", "HEAD"]),
                "branch": git(["branch", "--show-current"]), "official_submodule_pointer": git(["ls-tree", "HEAD", "source/dppo_v0.6"]).split()[2],
                "budget": {"atomic": 0, "chunks": 0}, "runtime": {}, "preflight": {}, "tasks": {}, "files": [], "time_limit_reached": False}
    try:
        manifest["runtime"] = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                                "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                "mujoco_backend": os.environ.get("MUJOCO_PY_MUJOCO_PATH"), "mujoco_gl": os.environ.get("MUJOCO_GL")}
        manifest["preflight"]["gate_self_test"] = gate_self_test()
        if not all(x["pass"] for x in manifest["preflight"]["gate_self_test"].values()):
            raise RuntimeError("gate self-test failed")
        models_all = {}; info_all = {}
        for task in ("square", "transport"):
            models_all[task], info_all[task] = load_models(task, args.device)
        manifest["assets"] = info_all
        sampler = {}
        for task in ("square", "transport"):
            sampler[task] = sampler_precheck(models_all[task], args.device)
        manifest["preflight"]["sampler"] = sampler
        if any(max(x["action_max_abs"], x["chain_max_abs"]) > 1e-6 for t in sampler.values() for x in t.values()):
            raise RuntimeError("sampler equivalence failed")
        budget = manifest["budget"]
        # Precheck both tasks before the Square-first scientific gate.
        for task in ("square", "transport"):
            adapter = EnvAdapter(task)
            try:
                pilot = {}
                for seed in PILOT_SEEDS:
                    pilot[str(seed)] = env_precheck(adapter, seed)
                manifest["preflight"][task] = {"passed": all(x["passed"] for x in pilot.values()), "pilot": pilot}
                budget["atomic"] += 32  # two seeds, each replayed twice for 8 atomic steps.
            finally:
                adapter.close()
        # The pilot and snapshot precheck count above is corrected by actual script-side counting in task execution;
        # this explicit counter is conservative and recorded as preflight_steps in the manifest.
        manifest["preflight_steps"] = 64
        for task in ("square", "transport"):
            if time.monotonic() > deadline:
                manifest["time_limit_reached"] = True; break
            if task == "transport" and manifest["tasks"].get("square", {}).get("gate") != "PHENOMENON_PASS":
                manifest["tasks"][task] = {"gate": "NOT_RUN", "reason": "Square did not achieve PHENOMENON_PASS; Transport main screening prohibited."}
                continue
            tr, rows, anchors, parent = screen_task(task, models_all[task], info_all[task], args.device, start, deadline, budget, outdir,
                                                     manifest["preflight"][task])
            manifest["tasks"][task] = tr
            manifest.setdefault("raw_rows", []).extend(rows)
        manifest["preflight"]["gate_self_test_exit_code"] = 0
        gates = [manifest["tasks"].get(t, {}).get("gate") for t in ("square", "transport")]
        if gates == ["PHENOMENON_PASS", "PHENOMENON_PASS"]:
            manifest["status"] = "PASS"; manifest["phenomenon_gate"] = "PASS"; manifest["next_stage"] = "GO_FOR_UPPER_REVIEW"
        else:
            manifest["status"] = "INCONCLUSIVE" if any(g in ("PHENOMENON_INCONCLUSIVE", "NOT_RUN", None) for g in gates) else "FAIL"
            manifest["phenomenon_gate"] = "INCONCLUSIVE" if manifest["status"] == "INCONCLUSIVE" else "FAIL"
            manifest["next_stage"] = "NO_GO"
        # The protocol also requires a real BC-BC paired-effect precheck and
        # pulse clip/EEF/object displacement logging.  This runner deliberately
        # records their absence instead of silently treating the simpler pilot
        # checks as a complete implementation audit.
        manifest["implementation_audit"] = "INCOMPLETE" if all(manifest["preflight"].get(t, {}).get("passed", True) for t in ("square", "transport")) else "FAIL"
        manifest["implementation_precheck_gaps"] = [
            "BC-BC paired effect precheck was not executed by this runner",
            "pulse clip ratio, EEF displacement and object displacement were not logged as dedicated RAW fields",
        ]
        manifest["measurement_validity"] = "VALID" if all(manifest["tasks"].get(t, {}).get("measurement_valid", False) for t in ("square", "transport")) else "INCOMPLETE"
    except Exception as exc:
        manifest["status"] = "INCONCLUSIVE"; manifest["phenomenon_gate"] = "INCONCLUSIVE"; manifest["next_stage"] = "NO_GO"
        manifest["implementation_audit"] = "FAIL"; manifest["measurement_validity"] = "INVALID"; manifest["error"] = repr(exc)
        manifest["error_type"] = type(exc).__name__
    finally:
        manifest["wall_clock_seconds"] = time.monotonic() - start
        manifest["time_limit_reached"] = bool(manifest.get("time_limit_reached") or manifest["wall_clock_seconds"] >= HARD_SECONDS)
        raw = manifest.pop("raw_rows", [])
        csv_path = ROOT / "reports" / "research" / "S0D_R2_RAW_METRICS.csv"
        fields = ["task", "checkpoint_hash", "reset_seed", "source_policy", "boundary", "pulse_direction", "condition",
                  "screening_or_test", "block", "draw", "native_success", "termination_reason", "remaining_horizon",
                  "actual_steps", "snapshot_id", "pulse_id", "actual_action_delta"]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{k: r.get(k, "") for k in fields} for r in raw])
        manifest["files"].append(save_file_meta(csv_path))
        protocol = ROOT / "reports" / "research" / "S0D_R2_PROTOCOL.md"
        manifest["protocol_sha256"] = sha256(protocol)
        report_path = ROOT / "reports" / "research" / "S0D_R2_RECOVERY_RETENTION_AUDIT.md"
        report_path.write_text(report_text(manifest, manifest.get("tasks", {})), encoding="utf-8")
        manifest["files"].append(save_file_meta(report_path))
        manifest_path = ROOT / "reports" / "research" / "S0D_R2_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": manifest.get("status"), "implementation_audit": manifest.get("implementation_audit"),
                          "measurement_validity": manifest.get("measurement_validity"), "wall_clock_seconds": manifest["wall_clock_seconds"],
                          "atomic_steps": manifest["budget"]["atomic"], "task_gates": {k: v.get("gate") for k, v in manifest.get("tasks", {}).items()},
                          "error": manifest.get("error")}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
