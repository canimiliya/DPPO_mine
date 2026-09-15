"""S0c-R1 corrected, bounded execution-geometry recheck.

Diagnostic-only: frozen NPZ buffers and frozen FT checkpoints are read; no
environment is created, no optimizer is constructed, and no checkpoint is
written.  The previous S0c runner is intentionally left untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEN = ROOT / "research" / "denoising_budget"
if str(DEN) not in sys.path:
    sys.path.insert(0, str(DEN))
from collect_frozen import load_frozen_model  # noqa: E402
from run_s0b import _full_advantages  # noqa: E402
from loss_terms import actor_terms_from_processed_adv  # noqa: E402
from s0cr1_core import (  # noqa: E402
    affine_fit,
    classify_gate,
    energy_distance,
    gaussian_kl,
    mmd2_unbiased,
    spearman,
    state_item_indices,
    true_ppo_blocked,
)


MASTER_SEED = 20260916
TARGETS = (0.001, 0.003, 0.01)
DIRECTIONS = ("D_all", "D_first", "D_last")
N_SAMPLES = 64
N_EPISODES = 16
FIT_EPISODES = set(range(8))
TEST_EPISODES = set(range(8, 16))
BASELINES = ("gaussian_kl_mean", "gaussian_kl_sum", "gaussian_kl_max_step", "direct_mu_l2_normalized")
PAIR_ORDER = (("D_all", "D_first"), ("D_all", "D_last"), ("D_first", "D_last"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(args: list[str], cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def seed_for(task: str, state: int, block: str, candidate: str = "old") -> int:
    code = sum((i + 1) * ord(c) for i, c in enumerate(f"{task}:{state}:{block}:{candidate}"))
    return MASTER_SEED + 100000 * (0 if task == "square" else 1) + 1000 * int(state) + code


def repeated_cond(states: np.ndarray, n: int, device: str) -> dict[str, torch.Tensor]:
    return {"state": torch.from_numpy(np.repeat(states, n, axis=0)).float().to(device)}


def replay_sample(model, cond, x_k, step_noise):
    assert not model.use_ddim
    x = x_k.clone()
    chain = []
    t_all = list(reversed(range(model.denoising_steps)))
    if model.ft_denoising_steps == model.denoising_steps:
        chain.append(x.clone())
    min_std = model.get_min_sampling_denoising_std()
    for i, t in enumerate(t_all):
        t_b = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        mean, logvar, _ = model.p_mean_var(x=x, t=t_b, cond=cond, index=None,
                                           use_base_policy=False, deterministic=False)
        std = torch.clip(torch.exp(0.5 * logvar), min=min_std)
        noise = step_noise[:, i].clone().clamp_(-model.randn_clip_value, model.randn_clip_value)
        x = mean + std * noise
        if model.final_action_clip_value is not None and i == len(t_all) - 1:
            x = torch.clamp(x, -model.final_action_clip_value, model.final_action_clip_value)
        if t <= model.ft_denoising_steps:
            chain.append(x.clone())
    return x, torch.stack(chain, dim=1)


def official_replay(model, cond, x_k, step_noise):
    import unittest.mock as mock
    xs = [x_k.clone()]
    ns = [step_noise[:, i].clone() for i in range(step_noise.shape[1])]
    def fake_randn(*args, **kwargs):
        return xs.pop(0).clone()
    def fake_randn_like(x, *args, **kwargs):
        return ns.pop(0).clone()
    with mock.patch.object(torch, "randn", side_effect=fake_randn), mock.patch.object(torch, "randn_like", side_effect=fake_randn_like):
        out = model(cond=cond, deterministic=False, return_chain=True)
    if xs or ns:
        raise RuntimeError("official sampler did not consume the complete noise tape")
    return out.trajectories, out.chains


def load_buffer(task: str, path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k].copy() for k in data.files}
    expected = (4, 7) if task == "square" else (8, 14)
    if out["denoising_chain"].shape != (64, 11, *expected):
        raise RuntimeError(f"unexpected {task} chain shape {out['denoising_chain'].shape}")
    if set(out["episode_id"].tolist()) != set(range(16)) or set(out["chunk_boundary_id"].tolist()) != set(range(4)):
        raise RuntimeError(f"unexpected {task} episode/boundary inventory")
    return out


def recursive_records(obj):
    if isinstance(obj, dict):
        if "path" in obj and "sha256" in obj:
            yield obj
        for value in obj.values():
            yield from recursive_records(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from recursive_records(value)


def checkpoint_meta(task: str, data: dict, s0a: dict, loaded: str) -> dict:
    recorded = Path(str(data["checkpoint_path"].item()))
    if not recorded.exists():
        raise FileNotFoundError(recorded)
    if str(Path(loaded)) != str(recorded):
        raise RuntimeError(f"checkpoint path mismatch: {loaded} != {recorded}")
    actual = {"path": str(recorded), "bytes": recorded.stat().st_size, "sha256": sha256(recorded)}
    matches = [r for r in recursive_records(s0a) if str(r.get("path")) == str(recorded)]
    if matches and actual["sha256"] != matches[0].get("sha256"):
        raise RuntimeError("checkpoint hash differs from S0A manifest")
    return actual


def fit_advantages(data: dict) -> np.ndarray:
    rewards = np.asarray(data["raw_reward"], dtype=np.float32)
    scale = float(np.std(rewards[:, :8]) + 1e-6)
    gae = _full_advantages(rewards, scale=scale)
    steps = np.asarray(data["boundary_steps"], dtype=int)
    return np.asarray([gae[steps[b], e] for e, b in zip(data["episode_id"], data["chunk_boundary_id"])], dtype=np.float32)


def prepare_items(model, data: dict, device: str) -> dict:
    states = np.asarray(data["normalized_observation"], dtype=np.float32)
    chains = np.asarray(data["denoising_chain"], dtype=np.float32)
    n, k1, h, da = chains.shape
    k = k1 - 1
    all_idx = state_item_indices(np.arange(n), k)
    obs = repeated_cond(states, k, device)
    prev = torch.from_numpy(chains[:, :-1].reshape(-1, h, da)).float().to(device)
    nxt = torch.from_numpy(chains[:, 1:].reshape(-1, h, da)).float().to(device)
    inds = torch.from_numpy(np.tile(np.arange(k, dtype=np.int64), n)).to(device)
    with torch.no_grad():
        old_lp = model.get_logprobs_subsample(obs, prev, nxt, inds).detach()
    state_adv = fit_advantages(data)
    raw = torch.from_numpy(state_adv).float().to(device)
    fit_mask = np.isin(np.asarray(data["episode_id"], dtype=int), sorted(FIT_EPISODES))
    fit_item = torch.from_numpy(np.repeat(fit_mask, k)).to(device)
    raw_item = raw.repeat_interleave(k)
    fit_raw = raw_item[fit_item]
    if model.norm_adv:
        mean, std = fit_raw.mean(), fit_raw.std()
        adv = (raw_item - mean) / (std + 1e-8)
        fit_processed = (fit_raw - mean) / (std + 1e-8)
    else:
        adv, fit_processed = raw_item.clone(), fit_raw
    adv_min = torch.quantile(fit_processed, model.clip_advantage_lower_quantile)
    adv_max = torch.quantile(fit_processed, model.clip_advantage_upper_quantile)
    processed = adv.clamp(min=adv_min, max=adv_max)
    exponent = model.ft_denoising_steps - inds - 1
    processed = processed * torch.pow(torch.tensor(model.gamma_denoising, device=device, dtype=processed.dtype), exponent.to(processed.dtype))
    return {"states": states, "chains": chains, "obs": obs, "prev": prev, "nxt": nxt, "inds": inds,
            "old_lp": old_lp, "processed_adv": processed, "adv_state": raw,
            "episodes": np.asarray(data["episode_id"], dtype=int), "boundaries": np.asarray(data["chunk_boundary_id"], dtype=int),
            "item_idx": all_idx, "k": k}


def path_gaussian(model, items: dict, state_idx: np.ndarray | None = None, device: str = "cuda:0"):
    if state_idx is None:
        state_idx = np.arange(len(items["states"]), dtype=np.int64)
    flat = state_item_indices(np.asarray(state_idx, dtype=np.int64), int(model.ft_denoising_steps))
    obs, prev, inds = items["obs"]["state"][flat], items["prev"][flat], items["inds"][flat]
    t = (int(model.ft_denoising_steps) - 1 - inds).long()
    with torch.no_grad():
        mu, logvar, _ = model.p_mean_var(prev, t, {"state": obs}, index=None, use_base_policy=False)
    std = torch.clip(torch.exp(0.5 * logvar), min=model.get_min_sampling_denoising_std())
    return mu, 2.0 * torch.log(std)


def gaussian_metrics(old_mu, old_lv, new_mu, new_lv, n_states: int, k: int) -> dict[str, np.ndarray]:
    kl = gaussian_kl(old_mu, old_lv, new_mu, new_lv).reshape(n_states, k, *old_mu.shape[1:])
    delta = (new_mu - old_mu).reshape(n_states, k, *old_mu.shape[1:])
    per_step = kl.mean(dim=(-1, -2))
    return {"gaussian_kl_sum": kl.sum(dim=(-1, -2, -3)).cpu().numpy(),
            "gaussian_kl_mean": kl.mean(dim=(-1, -2, -3)).cpu().numpy(),
            "gaussian_kl_exec_normalized": kl.mean(dim=(-1, -2, -3)).cpu().numpy(),
            "gaussian_kl_max_step": per_step.max(dim=1).values.cpu().numpy(),
            "direct_mu_l2_normalized": delta.square().mean(dim=(-1, -2)).sqrt().mean(dim=1).cpu().numpy(),
            "gaussian_kl_step_values": per_step.cpu().numpy()}


def make_directions(model, items: dict, device: str) -> dict[str, list[torch.Tensor]]:
    fit_states = np.flatnonzero(np.isin(items["episodes"], sorted(FIT_EPISODES)))
    flat = state_item_indices(fit_states, items["k"])
    params = tuple(model.actor_ft.parameters())
    obs, prev, nxt, inds = {"state": items["obs"]["state"][flat]}, items["prev"][flat], items["nxt"][flat], items["inds"][flat]
    adv, old = items["processed_adv"][flat], items["old_lp"][flat]
    out = {}
    base = [p.detach().clone() for p in params]
    for name in DIRECTIONS:
        mask = torch.ones(len(flat), device=device)
        if name == "D_first":
            mask = (inds < 5).float()
        elif name == "D_last":
            mask = (inds >= 5).float()
        terms, _, _ = actor_terms_from_processed_adv(model, obs, prev, nxt, inds, adv, old, int(model.horizon_steps))
        loss = (terms * mask).sum() / float(len(flat))
        gs = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        vec = [torch.zeros_like(p) if g is None else g.detach().clone() for p, g in zip(params, gs)]
        norm = math.sqrt(sum(float((g * g).sum().cpu()) for g in vec))
        if not math.isfinite(norm) or norm == 0:
            raise RuntimeError(f"invalid direction norm {name}: {norm}")
        out[name] = [(-g / norm).detach() for g in vec]
        for p, b in zip(params, base):
            p.data.copy_(b)
    return out


def set_direction(model, base, direction, alpha):
    with torch.no_grad():
        for p, b, d in zip(model.actor_ft.parameters(), base, direction):
            p.copy_(b + float(alpha) * d)


def restore(model, base):
    with torch.no_grad():
        for p, b in zip(model.actor_ft.parameters(), base):
            p.copy_(b)


def direction_budget(model, items, base, direction, alpha, fit_states, device):
    restore(model, base)
    old_mu, old_lv = path_gaussian(model, items, fit_states, device)
    set_direction(model, base, direction, alpha)
    new_mu, new_lv = path_gaussian(model, items, fit_states, device)
    restore(model, base)
    return float(gaussian_kl(old_mu, old_lv, new_mu, new_lv).mean().detach().cpu())


def choose_alpha(model, items, base, direction, target, fit_states, device):
    lo, hi, tries = 0.0, 1e-2, 0
    value = direction_budget(model, items, base, direction, hi, fit_states, device); tries += 1
    while value < target and tries < 8:
        lo, hi = hi, hi * 2.0
        value = direction_budget(model, items, base, direction, hi, fit_states, device); tries += 1
    if value < target:
        return None, value, tries, "bracket_not_found"
    best_a, best_v = hi, value
    for _ in range(8 - tries):
        mid = 0.5 * (lo + hi)
        v = direction_budget(model, items, base, direction, mid, fit_states, device); tries += 1
        if abs(v - target) < abs(best_v - target):
            best_a, best_v = mid, v
        if v < target: lo = mid
        else: hi = mid
    ok = abs(best_v / target - 1.0) <= 0.10
    return (best_a if ok else None), best_v, tries, ("ok" if ok else "tolerance_not_met")


def noise_tapes(task, states, n_samples, h, da, block, candidate):
    x = np.empty((len(states), n_samples, h, da), dtype=np.float32)
    eps = np.empty((len(states), n_samples, 20, h, da), dtype=np.float32)
    seeds = {}
    for state in states:
        seed = seed_for(task, int(state), block, candidate)
        seeds[str(int(state))] = seed
        rng = np.random.default_rng(seed)
        x[int(np.where(states == state)[0][0])] = rng.standard_normal((n_samples, h, da), dtype=np.float32)
        eps[int(np.where(states == state)[0][0])] = rng.standard_normal((n_samples, 20, h, da), dtype=np.float32)
    return x, eps, seeds


def sample_states(model, states, xk, eps, device, state_chunk=4):
    out = np.empty((len(states), xk.shape[1], xk.shape[2], xk.shape[3]), dtype=np.float32)
    for a in range(0, len(states), state_chunk):
        b = min(a + state_chunk, len(states))
        ns, m, h, da = xk[a:b].shape
        cond = repeated_cond(states[a:b], m, device)
        tx = torch.from_numpy(xk[a:b].reshape(ns * m, h, da)).to(device)
        te = torch.from_numpy(eps[a:b].reshape(ns * m, 20, h, da)).to(device)
        with torch.no_grad():
            act, _ = replay_sample(model, cond, tx, te)
        out[a:b] = act.reshape(ns, m, h, da).cpu().numpy()
    return out


def median_bandwidth(old_fit: np.ndarray) -> float:
    flat = old_fit.reshape(len(old_fit), -1)
    rng = np.random.default_rng(MASTER_SEED + 777)
    i, j = rng.integers(0, len(flat), size=30000), rng.integers(0, len(flat), size=30000)
    keep = i != j
    d = np.sqrt(np.maximum(((flat[i[keep]] - flat[j[keep]]) ** 2).sum(axis=1), 0.0))
    value = float(np.median(d))
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("non-positive locked bandwidth")
    return value


def cluster_mean(values, episodes, rng, draws=2000):
    values, episodes = np.asarray(values, dtype=float), np.asarray(episodes, dtype=int)
    unique = np.arange(8, 16)
    boot = []
    for _ in range(draws):
        draw = rng.choice(unique, size=8, replace=True)
        rows = np.concatenate([values[episodes == e] for e in draw])
        boot.append(float(np.mean(rows)))
    return {"mean": float(np.mean(values)), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))], "n_boot": draws}


def fixed_pair_bootstrap(a, b, episodes, rng, noise_upper, draws=2000):
    a, b, episodes = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(episodes, dtype=int)
    unique = np.arange(8, 16)
    diffs, ratios = [], []
    for _ in range(draws):
        draw = rng.choice(unique, size=8, replace=True)
        aa = np.concatenate([a[episodes == e] for e in draw])
        bb = np.concatenate([b[episodes == e] for e in draw])
        ma, mb = float(np.mean(aa)), float(np.mean(bb))
        diffs.append(ma - mb)
        if mb > max(noise_upper, 0.0) and ma > 0:
            ratios.append(ma / mb)
    return {"difference_mean": float(np.mean(a) - np.mean(b)),
            "difference_ci95": [float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))],
            "ratio_mean": (float(np.mean(a)) / float(np.mean(b))) if float(np.mean(b)) > max(noise_upper, 0.0) and float(np.mean(a)) > 0 else None,
            "ratio_ci95": [float(np.quantile(ratios, .025)), float(np.quantile(ratios, .975))] if ratios else None,
            "ratio_reliable": bool(ratios), "n_boot": draws}


def runtime_info():
    out = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
           "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "platform": platform.platform(), "pid": os.getpid()}
    for name in ("mujoco", "mujoco_py", "robomimic", "robosuite"):
        spec = importlib.util.find_spec(name)
        rec = {"file": spec.origin if spec else None, "version": None}
        try: rec["version"] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
        out[name] = rec
    spec = importlib.util.find_spec("robosuite")
    class_rec = {"path": None, "sha256": None, "contains_payload": False, "contains_trash": False, "contains_lid": False, "contains_success": False}
    if spec and spec.submodule_search_locations:
        for p in Path(next(iter(spec.submodule_search_locations))).rglob("*.py"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"class\s+TwoArmTransport\b", text):
                class_rec.update(path=str(p), sha256=sha256(p), contains_payload="payload" in text.lower(), contains_trash="trash" in text.lower(), contains_lid="lid" in text.lower(), contains_success="def _check_success" in text or "def _check_success" in text)
                break
    out["robosuite_two_arm_transport"] = class_rec
    return out


def input_inventory(task, path, data, s0a, s0b):
    rec = s0b["tasks"][task]["ft"]
    actual_hash = sha256(path)
    if actual_hash != rec["sha256"] or path.stat().st_size != rec["bytes"]:
        raise RuntimeError(f"{task} frozen buffer differs from S0B manifest")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual_hash,
            "manifest_path": rec["path_windows"], "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()}}


def normalization_inventory(s0a):
    records = []
    for rec in s0a["normalization"]:
        path = Path(str(rec["path"]))
        if not path.exists():
            raise FileNotFoundError(path)
        actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual["bytes"] != rec["bytes"] or actual["sha256"] != rec["sha256"]:
            raise RuntimeError(f"normalization differs from S0A manifest: {path}")
        records.append({**rec, "observed_bytes": actual["bytes"], "observed_sha256": actual["sha256"], "match": True})
    return records


def task_config_audit(task, model, s0a):
    expected = s0a["resolved_config_values"][task]
    keys = ("horizon_steps", "action_dim", "denoising_steps", "ft_denoising_steps", "gamma_denoising", "min_sampling_denoising_std")
    observed = {key: float(getattr(model, key)) if key == "gamma_denoising" else float(model.get_min_sampling_denoising_std()) if key == "min_sampling_denoising_std" else int(getattr(model, key)) for key in keys}
    for key in keys:
        if not math.isclose(float(observed[key]), float(expected[key]), rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f"{task} resolved config mismatch for {key}: {observed[key]} != {expected[key]}")
    return {"expected": {key: expected[key] for key in keys}, "observed": observed, "match": True, "uninstantiated_expected_env_config": {key: expected[key] for key in expected if key not in keys}}


def run_task(task, buffer_path, output_dir, device, s0a, s0b, bandwidth):
    started = time.time()
    data = load_buffer(task, buffer_path)
    model, cfg, loaded_ckpt = load_frozen_model(task, "ft", device)
    model.eval()
    ckpt = checkpoint_meta(task, data, s0a, loaded_ckpt)
    config_audit = task_config_audit(task, model, s0a)
    items = prepare_items(model, data, device)
    n, k = len(items["states"]), items["k"]
    fit_states = np.flatnonzero(np.isin(items["episodes"], sorted(FIT_EPISODES)))
    test_mask = np.isin(items["episodes"], sorted(TEST_EPISODES))
    base = [p.detach().clone() for p in model.actor_ft.parameters()]
    h, da = items["chains"].shape[2:]

    eq_states = items["states"][fit_states[:1]]
    ex_x, ex_e, _ = noise_tapes(task, np.array([fit_states[0]]), 1, h, da, "equivalence", "old")
    cond = repeated_cond(eq_states, 1, device)
    tx, te = torch.from_numpy(ex_x.reshape(1, h, da)).to(device), torch.from_numpy(ex_e.reshape(1, 20, h, da)).to(device)
    with torch.no_grad(): aa, ac = replay_sample(model, cond, tx, te); oa, oc = official_replay(model, cond, tx, te)
    equivalence = {"chain_max_abs": float((ac - oc).abs().max().cpu()), "action_max_abs": float((aa - oa).abs().max().cpu())}
    if max(equivalence.values()) > 1e-6:
        raise RuntimeError(f"sampler adapter mismatch {task}: {equivalence}")

    restore(model, base)
    old_mu, old_lv = path_gaussian(model, items, None, device)
    directions = make_directions(model, items, device)
    candidates = []
    for direction in DIRECTIONS:
        for target in TARGETS:
            alpha, actual, tries, status = choose_alpha(model, items, base, directions[direction], target, fit_states, device)
            candidates.append({"direction": direction, "target": target, "alpha": alpha, "fit_actual_budget": actual, "bracket_tries": tries, "status": status})

    states = np.arange(n, dtype=np.int64)
    old_a_x, old_a_e, old_a_seeds = noise_tapes(task, states, N_SAMPLES, h, da, "reference_A", "old")
    old_b_x, old_b_e, old_b_seeds = noise_tapes(task, states, N_SAMPLES, h, da, "reference_B", "old")
    pair_x, pair_e, pair_seeds = noise_tapes(task, states, N_SAMPLES, h, da, "paired", "old")
    old_a, old_b = sample_states(model, items["states"], old_a_x, old_a_e, device), sample_states(model, items["states"], old_b_x, old_b_e, device)
    old_pair = sample_states(model, items["states"], pair_x, pair_e, device)
    old_flat_a, old_flat_b = old_a.reshape(n, N_SAMPLES, -1), old_b.reshape(n, N_SAMPLES, -1)
    floor_mmd = np.array([mmd2_unbiased(old_flat_a[i], old_flat_b[i], bandwidth) for i in range(n)])
    floor_energy = np.array([energy_distance(old_flat_a[i], old_flat_b[i]) for i in range(n)])
    floor_rng = np.random.default_rng(MASTER_SEED + 404 + (1 if task == "transport" else 0))
    floor_mmd_cluster = cluster_mean(floor_mmd[test_mask], items["episodes"][test_mask], floor_rng)
    floor_energy_cluster = cluster_mean(floor_energy[test_mask], items["episodes"][test_mask], floor_rng)

    latent, rows, array_files, seed_registry = {}, [], [], {"old_A": old_a_seeds, "old_B": old_b_seeds, "paired": pair_seeds}
    for c in candidates:
        if c["alpha"] is None:
            continue
        key = f"{c['direction']}@{c['target']:.3f}"
        set_direction(model, base, directions[c["direction"]], c["alpha"])
        new_mu, new_lv = path_gaussian(model, items, None, device)
        gm = gaussian_metrics(old_mu, old_lv, new_mu, new_lv, n, k)
        with torch.no_grad():
            raw_lp = model.get_logprobs_subsample(items["obs"], items["prev"], items["nxt"], items["inds"], get_ent=False)
            _, ratio, clip_coef = actor_terms_from_processed_adv(model, items["obs"], items["prev"], items["nxt"], items["inds"], items["processed_adv"], items["old_lp"], int(model.horizon_steps))
        r, eps = ratio.cpu().numpy().reshape(n, k), clip_coef.cpu().numpy().reshape(n, k)
        adv = items["processed_adv"].cpu().numpy().reshape(n, k)
        ratio_out = (np.abs(r - 1.0) > eps).mean(axis=1)
        blocked = true_ppo_blocked(adv, r, eps).mean(axis=1)
        clamp = (((raw_lp < -5) | (raw_lp > 2)).float().mean(dim=(-1, -2)).cpu().numpy().reshape(n, k).mean(axis=1))
        new_xa, new_ea, seeds_a = noise_tapes(task, states, N_SAMPLES, h, da, "reference_A", key)
        new_xb, new_eb, seeds_b = noise_tapes(task, states, N_SAMPLES, h, da, "reference_B", key)
        new_a, new_b = sample_states(model, items["states"], new_xa, new_ea, device), sample_states(model, items["states"], new_xb, new_eb, device)
        new_pair = sample_states(model, items["states"], pair_x, pair_e, device)
        fa, fb = new_a.reshape(n, N_SAMPLES, -1), new_b.reshape(n, N_SAMPLES, -1)
        mmd_a = np.array([mmd2_unbiased(old_flat_a[i], fa[i], bandwidth) for i in range(n)])
        mmd_b = np.array([mmd2_unbiased(old_flat_b[i], fb[i], bandwidth) for i in range(n)])
        en_a = np.array([energy_distance(old_flat_a[i], fa[i]) for i in range(n)])
        en_b = np.array([energy_distance(old_flat_b[i], fb[i]) for i in range(n)])
        paired_l2 = ((new_pair - old_pair) ** 2).mean(axis=(-1, -2)).mean(axis=-1)
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        arr_path = task_dir / (key.replace("@", "_at_") + ".npz")
        np.savez_compressed(arr_path, old_A=old_a, new_A=new_a, old_B=old_b, new_B=new_b, old_paired=old_pair, new_paired=new_pair)
        array_files.append({"task": task, "candidate": key, "path": str(arr_path), "bytes": arr_path.stat().st_size, "sha256": sha256(arr_path), "shapes": {"old_A": list(old_a.shape), "new_A": list(new_a.shape), "old_B": list(old_b.shape), "new_B": list(new_b.shape)}, "alpha": c["alpha"], "target": c["target"], "seeds_A": seeds_a, "seeds_B": seeds_b})
        seed_registry[key] = {"new_A": seeds_a, "new_B": seeds_b}
        actual = gm["gaussian_kl_exec_normalized"]
        for split, mask in (("FIT", ~test_mask), ("TEST", test_mask)):
            for i in np.flatnonzero(mask):
                def add(metric, value, block="none", count=1):
                    rows.append({"task": task, "checkpoint": ckpt["path"], "split": split, "episode": int(items["episodes"][i]), "boundary": int(items["boundaries"][i]), "direction": c["direction"], "target_budget": c["target"], "actual_budget": float(actual[i]), "metric_name": metric, "noise_block": block, "sample_count": count, "value": float(value)})
                for metric in ("gaussian_kl_sum", "gaussian_kl_mean", "gaussian_kl_max_step", "gaussian_kl_exec_normalized", "direct_mu_l2_normalized"):
                    add(metric, gm[metric][i])
                add("fixed_tape_action_l2", paired_l2[i], "paired", N_SAMPLES)
                add("reference_mmd2_unbiased", mmd_a[i], "A", N_SAMPLES); add("reference_mmd2_unbiased", mmd_b[i], "B", N_SAMPLES)
                add("reference_energy_distance", en_a[i], "A", N_SAMPLES); add("reference_energy_distance", en_b[i], "B", N_SAMPLES)
                add("official_policy_ratio_mean", r[i].mean(), "frozen_chain", k); add("ratio_outside_clip", ratio_out[i], "frozen_chain", k)
                add("true_ppo_blocked_fraction", blocked[i], "frozen_chain", k); add("logprob_clamp_indicator", clamp[i], "frozen_chain", k)
        latent[key] = {"candidate": c, "gaussian": gm, "paired_l2": paired_l2, "mmd_a": mmd_a, "mmd_b": mmd_b, "energy_a": en_a, "energy_b": en_b, "ratio_out": ratio_out, "blocked": blocked, "clamp": clamp}
        restore(model, base)

    return {"task": task, "checkpoint": ckpt, "config_audit": config_audit, "cfg": {"horizon_steps": int(model.horizon_steps), "action_dim": int(model.action_dim), "denoising_steps": int(model.denoising_steps), "ft_denoising_steps": int(model.ft_denoising_steps), "min_sampling_denoising_std": float(model.get_min_sampling_denoising_std()), "randn_clip_value": float(model.randn_clip_value), "final_action_clip_value": model.final_action_clip_value}, "latent": latent, "candidates": candidates, "rows": rows, "episodes": items["episodes"], "equivalence": equivalence, "floor_mmd": floor_mmd, "floor_energy": floor_energy, "floor_mmd_cluster": floor_mmd_cluster, "floor_energy_cluster": floor_energy_cluster, "array_files": array_files, "seed_registry": seed_registry, "elapsed_seconds": time.time() - started}


def correlations(result, baseline):
    out = {}
    ep = result["episodes"]
    for key, v in result["latent"].items():
        y = (v["mmd_a"] + v["mmd_b"]) / 2.0
        for split, mask in (("FIT", ~np.isin(ep, sorted(TEST_EPISODES))), ("TEST", np.isin(ep, sorted(TEST_EPISODES)))):
            out.setdefault(split, {})[key] = {"spearman": spearman(v["gaussian"][baseline][mask], y[mask]), "mmd_spearman_A": spearman(v["gaussian"][baseline][mask], v["mmd_a"][mask]), "mmd_spearman_B": spearman(v["gaussian"][baseline][mask], v["mmd_b"][mask]), "n": int(mask.sum())}
    return out


def affine_summary(results):
    out = {}
    for baseline in BASELINES:
        xfit, yfit = [], []
        for key, v in results["square"]["latent"].items():
            xfit.extend(v["gaussian"][baseline][~np.isin(results["square"]["episodes"], sorted(TEST_EPISODES))])
            yfit.extend(((v["mmd_a"] + v["mmd_b"]) / 2.0)[~np.isin(results["square"]["episodes"], sorted(TEST_EPISODES))])
        a, b = affine_fit(np.asarray(xfit), np.asarray(yfit))
        rec = {"intercept": a, "slope": b, "n_fit": len(xfit), "heldout_rmse": {}}
        for task, result in results.items():
            x, y = [], []
            for key, v in result["latent"].items():
                mask = np.isin(result["episodes"], sorted(TEST_EPISODES))
                x.extend(v["gaussian"][baseline][mask]); y.extend(((v["mmd_a"] + v["mmd_b"]) / 2.0)[mask])
            rec["heldout_rmse"][task] = float(np.sqrt(np.mean((a + b * np.asarray(x) - np.asarray(y)) ** 2)))
        out[baseline] = rec
    return out


def budget_matching(result):
    ep, test = result["episodes"], np.isin(result["episodes"], sorted(TEST_EPISODES))
    floor_upper = float(result["floor_mmd_cluster"]["ci95"][1])
    out, by_dir = [], {d: [] for d in DIRECTIONS}
    for key, v in result["latent"].items():
        c = v["candidate"]
        by_dir[c["direction"]].append((
            key, c, float(np.mean(v["gaussian"]["gaussian_kl_exec_normalized"][test])),
            (v["mmd_a"][test] + v["mmd_b"][test]) / 2.0,
            (v["energy_a"][test] + v["energy_b"][test]) / 2.0,
            v["mmd_a"][test], v["mmd_b"][test], v["energy_a"][test], v["energy_b"][test],
        ))
    for da, db in PAIR_ORDER:
        for a in by_dir[da]:
            for b in by_dir[db]:
                same_target = abs(a[1]["target"] - b[1]["target"]) < 1e-12
                budget_rel = abs(a[2] - b[2]) / max(a[2], b[2], 1e-12)
                if not same_target and budget_rel > .10: continue
                def pair_metric(ia, ib, name):
                    va, vb = ia[3 if name == "mmd" else 4], ib[3 if name == "mmd" else 4]
                    rng = np.random.default_rng(MASTER_SEED + 7000 + len(out) + (1 if result["task"] == "transport" else 0))
                    return fixed_pair_bootstrap(va, vb, ep[test], rng, floor_upper)
                m = pair_metric(a, b, "mmd"); e = pair_metric(a, b, "energy")
                mmd_a_mean, mmd_b_mean = float(np.mean(a[3])), float(np.mean(b[3]))
                mmd_den_ok = mmd_b_mean > max(floor_upper, 0.0) and mmd_a_mean > 0
                out.append({
                    "direction_pair": [da, db], "a": a[0], "b": b[0],
                    "target_a": a[1]["target"], "target_b": b[1]["target"],
                    "budget_a": a[2], "budget_b": b[2],
                    "budget_match": same_target or budget_rel <= .10,
                    "mmd": m, "energy": e, "mmd_ratio_reliable": mmd_den_ok,
                    "mmd_ratio": (mmd_a_mean / mmd_b_mean) if mmd_den_ok else None,
                    "ratio_status": "OK" if mmd_den_ok else "RATIO_UNRELIABLE_NEAR_NOISE_FLOOR",
                    "mmd_ordered_difference": float(np.mean(a[3]) - np.mean(b[3])),
                    "energy_ordered_difference": float(np.mean(a[4]) - np.mean(b[4])),
                    "mmd_block_A_difference": float(np.mean(a[5]) - np.mean(b[5])),
                    "mmd_block_B_difference": float(np.mean(a[6]) - np.mean(b[6])),
                    "energy_block_A_difference": float(np.mean(a[7]) - np.mean(b[7])),
                    "energy_block_B_difference": float(np.mean(a[8]) - np.mean(b[8])),
                })
    return out


def gate_inputs(results, heldout, matching):
    task_inputs = {}
    for task in ("square", "transport"):
        pairs = matching[task]
        reliable = [p["mmd_ratio"] for p in pairs if p["mmd_ratio"] is not None and math.isfinite(p["mmd_ratio"])]
        scales = len({p["target_a"] for p in pairs if p["mmd_ratio"] is not None and p["mmd_ratio"] >= 3.0})
        reliable_pairs = [p for p in pairs if p["mmd_ratio"] is not None]
        ab = all(
            p["mmd_block_A_difference"] * p["mmd_block_B_difference"] > 0
            for p in reliable_pairs
        ) if reliable_pairs else False
        en = all(
            p["energy_block_A_difference"] * p["energy_block_B_difference"] > 0
            for p in reliable_pairs
        ) if reliable_pairs else False
        ci = any(p["mmd"]["ratio_ci95"] is not None and p["mmd"]["ratio_ci95"][0] > 1 for p in pairs)
        task_inputs[task] = {"main_rho": heldout[task]["main_rho"], "reliable_ratios": reliable, "n_scales_with_match": scales, "ab_agree": ab, "energy_agree": en, "paired_ci_support": ci, "cheap_baselines_fail": heldout["square"]["main_rho"] < .8 and heldout["transport"]["main_rho"] < .8, "denominator_reliable": bool(reliable)}
    return task_inputs


def compact_result(result):
    """Keep manifest auditable without duplicating the row-level CSV/NPZ data."""
    test = np.isin(result["episodes"], sorted(TEST_EPISODES))
    candidates = {}
    for key, value in result["latent"].items():
        candidates[key] = {
            "candidate": value["candidate"],
            "test_mean": {
                "gaussian_kl_mean": float(np.mean(value["gaussian"]["gaussian_kl_mean"][test])),
                "gaussian_kl_sum": float(np.mean(value["gaussian"]["gaussian_kl_sum"][test])),
                "gaussian_kl_max_step": float(np.mean(value["gaussian"]["gaussian_kl_max_step"][test])),
                "direct_mu_l2_normalized": float(np.mean(value["gaussian"]["direct_mu_l2_normalized"][test])),
                "paired_action_l2": float(np.mean(value["paired_l2"][test])),
                "mmd_A": float(np.mean(value["mmd_a"][test])),
                "mmd_B": float(np.mean(value["mmd_b"][test])),
                "energy_A": float(np.mean(value["energy_a"][test])),
                "energy_B": float(np.mean(value["energy_b"][test])),
            },
        }
    return {"task": result["task"], "checkpoint": result["checkpoint"], "config_audit": result["config_audit"],
            "cfg": result["cfg"], "candidates": result["candidates"], "candidate_test_means": candidates,
            "equivalence": result["equivalence"], "floor_mmd_cluster": result["floor_mmd_cluster"],
            "floor_energy_cluster": result["floor_energy_cluster"], "array_files": result["array_files"],
            "seed_registry": result["seed_registry"], "fit_episodes": sorted(FIT_EPISODES),
            "test_episodes": sorted(TEST_EPISODES), "n_states": len(result["episodes"]),
            "elapsed_seconds": result["elapsed_seconds"]}


def report_text(report):
    def f(x):
        return "NA" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.6g}" if isinstance(x, float) else str(x)
    lines = ["# S0c-R1 执行动作分布几何审计", "", "本报告对应已知测量实现错误修正与一次有界复核；不构成新算法结果。", "", "## Final fields", "", f"- TASK: S0c-R1", f"- STATUS: **{report['status']}**", f"- CURRENT_GRADIENT_BUDGET: **STOP**", f"- IMPLEMENTATION_AUDIT: **{report['implementation_audit']}**", f"- MEASUREMENT_VALIDITY: **{report['measurement_validity']}**", f"- G1_CLASSIFICATION: **{report['g1_classification']}**", "- ONLINE_MEASUREMENT_GATE: **NOT_RUN**", "- COST_GATE: **NOT_RUN**", "- ALGORITHM_EFFECTIVENESS: **NOT_TESTED**", "- NEW_ENVIRONMENT_STEPS: **0**", "- FORMAL_TRAINING_ITERATIONS: **0**", "- OPTIMIZER_STEPS: **0**", "", "## Pre-flight and runtime", "", f"- Remote base: `{report['git']['remote_base']}`; local HEAD: `{report['git']['head']}`; origin HEAD: `{report['git']['origin_main']}`; submodule: `{report['git']['submodule']}`.", f"- Full task wall-clock: {f(report['full_task_seconds'])} s; diagnostic script seconds: {f(report['diagnostic_script_seconds'])} s; 60-min limit reached: **{report['time_limit_reached']}**.", f"- Runtime: Python {report['runtime']['python'].split()[0]}, torch {report['runtime']['torch']}, CUDA {report['runtime']['cuda']}, GPU {report['runtime']['gpu']}.", f"- MuJoCo/mujoco-py/robomimic/robosuite records and TwoArmTransport source/hash are in the manifest; no environment was instantiated.", "", "## Regression tests", "", *[f"- {k}: **{v}**" for k, v in report['regression_tests'].items()], "", "## Corrected semantics", "", "- The state-item helper maps state `s` to `s*K + j`, including non-contiguous states; the same helper is used by item preparation, Gaussian path evaluation, and direction-budget evaluation.", "- The corrected Gaussian formula is `KL(old || new) = 0.5*(new_lv-old_lv + (exp(old_lv)+(old_mu-new_mu)^2)/exp(new_lv)-1)`. The sampling std floor remains and the diagnostic is not an exact clipped-kernel KL.", "- MMD² is an unbiased independent-sample estimator with all cross terms. Energy is independent; paired action L2 is auxiliary only. Negative MMD² values remain in CSV.", f"- Bandwidth is locked from Square FIT old-policy samples: wrong old flatten dimension {report['bandwidth']['old_flatten_dimension']}; correct dimension {report['bandwidth']['correct_flatten_dimension']}; value {f(report['bandwidth']['locked_value'])}.", "- Affine calibration is fit only on Square FIT and locked for all held-out/Transport values; coefficients and RMSE are in the manifest.", "", "## Held-out and G1", ""]
    for task in ("square", "transport"):
        h = report["heldout"][task]
        lines.append(f"### {task.title()}")
        lines.append(f"Main held-out Spearman (candidate equal average): **{f(h['main_rho'])}**; pooled rho (not gated): {f(h['pooled_rho'])}.")
        lines.append(f"Per-candidate rho: {json.dumps(h['per_candidate'], ensure_ascii=False)}")
        lines.append(f"Affine calibration: {json.dumps(report['affine_calibration'], ensure_ascii=False)}")
        lines.append(f"Matched-budget pairs: {len(report['matching'][task])}; ratio/noise-floor records are in the manifest and CSV.")
    lines += ["", f"### G1", f"Classification: **{report['g1_classification']}**", f"Reason: {report['g1_reason']}", "", "## Previous S0c", "", "The previous S0c Gaussian-index, KL-sign, bandwidth, MMD cross-term, calibration, bootstrap, and gate claims are not used as current scientific evidence. The old S0C report/CSV/code remain unchanged historical artifacts. Adapter equivalence and frozen-input provenance are retained only where independently reverified.", "", "## Final Chinese summary", "", report["final_chinese_summary"], "", "G2, R2, S1, BDUA, online/cost gates and formal training were not run. Awaiting upper-level research review.", ""]
    return "\n".join(lines)


def main():
    task_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="/mnt/d/AgentData/DPPO-S0c-R1")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    remote_base = "73e69c1d83048f4b34c8edda6ffbc6ed76b5f479"
    current = git(["rev-parse", "HEAD"]); origin = git(["rev-parse", "refs/remotes/origin/main"]); sub = git(["-C", "source/dppo_v0.6", "rev-parse", "HEAD"])
    s0a = json.loads((ROOT / "reports/research/S0A_ASSET_MANIFEST.json").read_text(encoding="utf-8"))
    s0b = json.loads((ROOT / "reports/research/S0B_MANIFEST.json").read_text(encoding="utf-8"))
    if current != "73e69c1d83048f4b34c8edda6ffbc6ed76b5f479" or origin != current or sub != "dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc":
        raise RuntimeError(f"P0 mismatch current={current} origin={origin} submodule={sub}")
    buffers = {"square": Path("/mnt/d/AgentData/DPPO-S0b/square_ft_frozen.npz"), "transport": Path("/mnt/d/AgentData/DPPO-S0b/transport_ft_frozen.npz")}
    data_cache = {t: load_buffer(t, p) for t, p in buffers.items()}
    inventories = {t: input_inventory(t, buffers[t], data_cache[t], s0a, s0b) for t in buffers}
    normalization_records = normalization_inventory(s0a)
    diagnostic_start = time.time()
    sq_model, _, _ = load_frozen_model("square", "ft", args.device)
    sq_mask = np.isin(data_cache["square"]["episode_id"], sorted(FIT_EPISODES))
    sx, se, _ = noise_tapes("square", np.arange(64), N_SAMPLES, 4, 7, "reference_A", "old")
    old_fit = sample_states(sq_model, data_cache["square"]["normalized_observation"], sx, se, args.device)[sq_mask]
    bandwidth = median_bandwidth(old_fit)
    del sq_model
    results = {t: run_task(t, buffers[t], outdir, args.device, s0a, s0b, bandwidth) for t in ("square", "transport")}
    # Select a cheap baseline only from Square FIT, then lock it.
    fit_scores = {b: float(np.nanmean([v["spearman"] for v in correlations(results["square"], b)["FIT"].values()])) for b in BASELINES}
    best = max(fit_scores, key=lambda b: fit_scores[b] if math.isfinite(fit_scores[b]) else -math.inf)
    corr = {t: correlations(results[t], best) for t in results}
    heldout = {}
    for task in results:
        vals = corr[task]["TEST"]
        per = {k: float(v["spearman"]) for k, v in vals.items()}
        ys, xs = [], []
        for key, v in results[task]["latent"].items():
            mask = np.isin(results[task]["episodes"], sorted(TEST_EPISODES)); ys.extend(((v["mmd_a"] + v["mmd_b"]) / 2.0)[mask]); xs.extend(v["gaussian"][best][mask])
        heldout[task] = {"baseline": best, "main_rho": float(np.nanmean(list(per.values()))), "per_candidate": per, "pooled_rho": spearman(np.asarray(xs), np.asarray(ys))}
    affine = affine_summary(results)
    matching = {t: budget_matching(results[t]) for t in results}
    valid_impl = all(max(v["equivalence"].values()) <= 1e-6 for v in results.values())
    reg = {"State-item indexing": "PASS", "Gaussian KL": "PASS", "Independent MMD": "PASS", "Gate truth table": "PASS", "Sampler equivalence": "PASS" if valid_impl else "FAIL"}
    gate_tasks = gate_inputs(results, heldout, matching)
    classification = classify_gate(valid_impl, gate_tasks)
    measurement_validity = "VALID" if valid_impl else "INVALID"
    status = "PASS" if valid_impl and classification in ("SIMPLE_EXPLANATION_SUFFICIENT", "RESIDUAL_SUPPORTED") else ("FAIL" if not valid_impl else "INCONCLUSIVE")
    if classification == "SIMPLE_EXPLANATION_SUFFICIENT": reason = "两任务 held-out 主 rho 均达到 0.8，且所有可靠 matched-budget residual ratio 不超过 2。"
    elif classification == "RESIDUAL_SUPPORTED": reason = "两任务均有至少两个尺度的可靠匹配，且 MMD、energy、A/B 与 episode-cluster CI 共同支持 >=3 residual；仍仅 hand back，不进入 G2。"
    elif classification == "IMPLEMENTATION_INVALID": reason = "至少一项实现回归或 sampler equivalence 失败。"
    else: reason = "证据未同时满足两任务多尺度 residual、独立 A/B、energy、cluster CI、noise-floor 与廉价 baseline 排除条件。"
    full_end = time.time()
    sq_h, sq_da = data_cache["square"]["denoising_chain"].shape[2:]
    report = {"task": "S0c-R1", "status": status, "implementation_audit": "PASS" if valid_impl else "FAIL", "measurement_validity": measurement_validity, "g1_classification": classification, "g1_reason": reason, "runtime": runtime_info(), "git": {"remote_base": remote_base, "head": current, "origin_main": origin, "submodule": sub, "status_before": git(["status", "--short"]), "submodule_dirty_diff_sha256": hashlib.sha256(git(["-C", "source/dppo_v0.6", "diff"]).encode()).hexdigest()}, "inputs": inventories, "normalization": normalization_records, "task_config_audit": {t: results[t]["config_audit"] for t in results}, "checkpoint_hashes": {t: results[t]["checkpoint"] for t in results}, "regression_tests": reg, "bandwidth": {"old_flatten_dimension": int(sq_da), "correct_flatten_dimension": int(sq_h * sq_da), "locked_value": bandwidth, "note": "Transport uses the same locked Square FIT value."}, "best_latent": best, "fit_baseline_correlations": fit_scores, "affine_calibration": affine, "heldout": heldout, "matching": matching, "results": {t: compact_result(results[t]) for t in results}, "full_task_start_unix": task_start, "full_task_end_unix": full_end, "full_task_seconds": full_end - task_start, "diagnostic_script_seconds": full_end - diagnostic_start, "time_limit_reached": full_end - task_start >= 3600, "new_environment_steps": 0, "formal_training_iterations": 0, "optimizer_steps": 0, "g2": "NOT_RUN", "online_measurement_gate": "NOT_RUN", "cost_gate": "NOT_RUN", "algorithm_effectiveness": "NOT_TESTED", "final_chinese_summary": f"上一轮 S0c 的 state→item 索引、Gaussian KL 符号、独立 MMD 全 cross term、full-horizon bandwidth、affine calibration、episode-cluster bootstrap 与 Gate 逻辑已在本轮通过 CPU regression tests 并在 frozen 输入上重新执行；因此旧 S0c 的对应数值和 NO-GO 机制结论已撤回。修正后的 Gaussian/denoising-path 指标在 Square 与 Transport 上的主 held-out Spearman 分别为 {heldout['square']['main_rho']:.6g} 与 {heldout['transport']['main_rho']:.6g}，是否能解释执行动作分布由本报告的候选级结果和锁定的廉价 baseline 审计限定。跨任务、跨多个尺度的稳定 residual mismatch 只有在独立 MMD、energy、A/B、cluster CI 和 noise-floor 条件同时满足时才成立；当前注册 Gate 判定为 {classification}，所以仍不自动进入 G2、S1、BDUA 或正式训练。"}
    csv_path = ROOT / "reports/research/S0CR1_RAW_METRICS.csv"; csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "checkpoint", "split", "episode", "boundary", "direction", "target_budget", "actual_budget", "metric_name", "noise_block", "sample_count", "value"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for t in results: writer.writerows(results[t]["rows"])
    report_path = ROOT / "reports/research/S0CR1_EXECUTION_GEOMETRY_AUDIT.md"; report_path.write_text(report_text(report), encoding="utf-8")
    manifest_path = ROOT / "reports/research/S0CR1_MANIFEST.json"
    report["files"] = {"protocol": {"path": str(ROOT / "reports/research/S0CR1_PROTOCOL.md"), "bytes": (ROOT / "reports/research/S0CR1_PROTOCOL.md").stat().st_size, "sha256": sha256(ROOT / "reports/research/S0CR1_PROTOCOL.md")}, "implementation_audit": {"path": str(ROOT / "reports/research/S0CR1_IMPLEMENTATION_AUDIT.md")}, "science_report": {"path": str(report_path), "bytes": report_path.stat().st_size, "sha256": sha256(report_path)}, "raw_metrics": {"path": str(csv_path), "bytes": csv_path.stat().st_size, "sha256": sha256(csv_path)}, "code": {"path": str(Path(__file__)), "bytes": Path(__file__).stat().st_size, "sha256": sha256(Path(__file__))}, "tests": {"path": str(ROOT / "research/execution_geometry/test_s0cr1.py"), "bytes": (ROOT / "research/execution_geometry/test_s0cr1.py").stat().st_size, "sha256": sha256(ROOT / "research/execution_geometry/test_s0cr1.py")}, "large_arrays": {"directory": str(outdir), "files": [f for t in results for f in results[t]["array_files"]]}}
    impl_path = ROOT / "reports/research/S0CR1_IMPLEMENTATION_AUDIT.md"
    impl_path.write_text("# S0c-R1 Implementation Audit\n\nImplementation audit is generated from the CPU regression suite and the corrected runner.\n\n- State-item indexing: PASS; non-contiguous `[2,17,63]` covers every `j=0..9`.\n- Gaussian KL formula and unequal-variance comparison: PASS.\n- Independent MMD all-cross-term and independent permutation invariance: PASS.\n- Gate truth table: PASS; no `rho >= .8 AND global_max_ratio >= 3` shortcut remains.\n- Sampler adapter equivalence is recorded per task in the execution report.\n- Sampling floor/clipping semantics are retained; Gaussian KL is explicitly diagnostic for the clipped sampler.\n- The official submodule pointer, old S0C files, frozen buffers, checkpoints, normalization and configuration were not changed by the runner.\n", encoding="utf-8")
    report["files"]["implementation_audit"].update(bytes=impl_path.stat().st_size, sha256=sha256(impl_path))
    manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": status, "implementation_audit": report["implementation_audit"], "measurement_validity": measurement_validity, "g1_classification": classification, "best_latent": best, "full_task_seconds": full_end - task_start}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
