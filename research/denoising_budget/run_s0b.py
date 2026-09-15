"""Bounded S0b analysis driver.

This file is a diagnostic harness only.  It uses frozen checkpoint copies,
keeps large NPZ buffers outside Git, and emits compact JSON/CSV summaries.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from loss_terms import actor_terms_from_processed_adv, processed_advantages
from sampler import proposal_probabilities, toy_expectation
from collect_frozen import load_frozen_model
from gradient_audit import last_layer_parameters, trainable_parameters, cheap_features, rank_correlation


def _full_advantages(rewards, gamma=0.999, lam=0.95, scale=1.0):
    r = np.asarray(rewards, dtype=np.float32) / float(scale)
    out = np.zeros_like(r)
    for e in range(r.shape[1]):
        gae = 0.0
        for t in range(r.shape[0] - 1, -1, -1):
            gae = float(r[t, e]) + gamma * lam * gae
            out[t, e] = gae
    return out


def _features_numpy(obs, j, adv, prev, nxt, old):
    s = obs.reshape(obs.shape[0], -1)
    d = (nxt - prev).reshape(prev.shape[0], -1)
    return np.column_stack([
        np.ones(len(s)), s.mean(1), s.std(1), np.abs(s).mean(1),
        np.asarray(j, dtype=np.float64) / 9.0, np.asarray(adv),
        np.linalg.norm(prev.reshape(len(s), -1), axis=1),
        np.linalg.norm(d, axis=1), np.asarray(old).reshape(len(s), -1).mean(1),
    ]).astype(np.float64)


def _batch_items(data, model, device):
    states = data["normalized_observation"].astype(np.float32)
    chains = data["denoising_chain"].astype(np.float32)
    n_states, _, h, da = chains.shape
    k = chains.shape[1] - 1
    obs = {"state": torch.from_numpy(np.repeat(states, k, axis=0)).to(device)}
    prev = torch.from_numpy(chains[:, :-1].reshape(-1, h, da)).to(device)
    nxt = torch.from_numpy(chains[:, 1:].reshape(-1, h, da)).to(device)
    inds = torch.arange(k, device=device).repeat(n_states)
    with torch.no_grad():
        old = model.get_logprobs_subsample(obs, prev, nxt, inds).detach()
    return states, chains, obs, prev, nxt, inds, old


def _grad_vector(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])


def analyze_task(task, policy, buffer_path, output_json, seed=42, device="cuda:0", full_norm_limit=16, mc_draws=200):
    data = np.load(buffer_path, allow_pickle=False)
    model, cfg, ckpt = load_frozen_model(task, policy, device)
    model.eval()
    states, chains, obs, prev, nxt, inds, old = _batch_items(data, model, device)
    n_states, k = states.shape[0], chains.shape[1] - 1
    episode = data["episode_id"].astype(int)
    boundary = data["chunk_boundary_id"].astype(int)
    rewards = data["raw_reward"]
    fit_scale = float(np.std(rewards[:, :8]) + 1e-6)
    gae = _full_advantages(rewards, scale=fit_scale)
    bsteps = data["boundary_steps"].astype(int)
    state_adv = np.asarray([gae[bsteps[b], e] for e, b in zip(episode, boundary)], dtype=np.float32)
    adv_state = torch.from_numpy(state_adv).to(device)
    adv_item = adv_state.repeat_interleave(k)
    proc = processed_advantages(model, adv_item, inds).detach()

    # Evaluate exact post-clipping terms in moderate batches so no huge graph
    # remains alive.  The full-anchor processing above is shared by all batches.
    term_values = torch.empty(n_states * k, device=device)
    ratios = torch.empty_like(term_values)
    clips = torch.empty_like(term_values)
    for a in range(0, n_states * k, 64):
        z = slice(a, min(a + 64, n_states * k))
        t, rr, cc = actor_terms_from_processed_adv(model, {"state": obs["state"][z]}, prev[z], nxt[z], inds[z], proc[z], old[z], int(cfg.act_steps))
        term_values[z] = t.detach(); ratios[z] = rr.detach(); clips[z] = cc.detach()
    full_scalar = float(term_values.mean().cpu())

    # Last-layer gradients are retained for the MSE study. Full-parameter
    # norms are streamed for the first 16 states as required by the audit.
    last_params = last_layer_parameters(model)
    last_dim = sum(p.numel() for p in last_params)
    G = np.zeros((n_states * k, last_dim), dtype=np.float32)
    full_norms = []
    last_norms = []
    full_params = trainable_parameters(model)
    started = time.perf_counter()
    for i in range(n_states * k):
        z = slice(i, i + 1)
        t, _, _ = actor_terms_from_processed_adv(model, {"state": obs["state"][z]}, prev[z], nxt[z], inds[z], proc[z], old[z], int(cfg.act_steps))
        if i < full_norm_limit * k:
            f = _grad_vector(t[0], full_params)
            full_norms.append(float(f.norm().detach().cpu()))
            t, _, _ = actor_terms_from_processed_adv(model, {"state": obs["state"][z]}, prev[z], nxt[z], inds[z], proc[z], old[z], int(cfg.act_steps))
        g = _grad_vector(t[0], last_params)
        G[i] = g.detach().cpu().numpy().astype(np.float32)
        if i < full_norm_limit * k:
            last_norms.append(float(g.norm().detach().cpu()))
    grad_seconds = time.perf_counter() - started
    full_g = G.mean(0)

    # Fit fold: episodes 0..7. All proposal parameters are fit-only.
    fit_mask = episode < 8
    fit_idx = np.repeat(fit_mask, k)
    j = np.tile(np.arange(k), n_states)
    adv_np = np.repeat(state_adv, k)
    obs_np = np.repeat(states, k, axis=0)
    prev_np = chains[:, :-1].reshape(-1, chains.shape[2], chains.shape[3])
    nxt_np = chains[:, 1:].reshape(-1, chains.shape[2], chains.shape[3])
    old_np = old.detach().cpu().numpy()
    second_all = np.mean(G ** 2, axis=1)
    second = second_all[fit_idx]
    # target is gradient squared norm; timestep-only aggregates it by j.
    step_v = np.array([np.mean(second_all[fit_idx & (j == jj)]) for jj in range(k)])
    p_uniform = np.full(len(G), 1 / len(G))
    p_time = proposal_probabilities(step_v[j], epsilon=0.2, delta=max(float(np.median(step_v)) * 1e-6, 1e-12))
    # State-aware linear ridge on fit fold. Standardization is fit-only.
    X = _features_numpy(obs_np, j, adv_np, prev_np, nxt_np, old_np)
    mu, sd = X[fit_idx].mean(0), X[fit_idx].std(0) + 1e-8
    Xz = (X - mu) / sd
    lam = 1e-3
    A = Xz[fit_idx].T @ Xz[fit_idx] + lam * np.eye(Xz.shape[1])
    beta = np.linalg.solve(A, Xz[fit_idx].T @ second)
    pred = np.maximum(Xz @ beta, 0.0)
    delta = max(float(np.median(second)) * 1e-6, 1e-12)
    p_state = proposal_probabilities(pred, epsilon=0.2, delta=delta)
    p_oracle = proposal_probabilities(np.linalg.norm(G, axis=1) ** 2, epsilon=0.2, delta=delta)

    rng = np.random.default_rng(seed + 991)
    m = len(G) // 2
    methods = {"uniform": p_uniform, "timestep-only": p_time, "state-aware": p_state, "oracle": p_oracle}
    mc = {}
    for name, p in methods.items():
        estimates = []
        weights_all = []
        ess = []
        for _ in range(mc_draws):
            idx = rng.choice(len(G), size=m, replace=True, p=p)
            w = 1.0 / (len(G) * p[idx])
            estimates.append(np.mean(G[idx] * w[:, None], axis=0))
            weights_all.extend(w.tolist())
            wn = w / w.sum()
            ess.append(float(1.0 / np.sum(wn ** 2)))
        est = np.asarray(estimates)
        err = est - full_g
        mse = np.sum(err ** 2, axis=1)
        # Cluster-aware diagnostic: report a bootstrap CI over episode-level
        # squared contribution errors, without treating MC draws as episodes.
        per_episode = []
        for ep in range(16):
            mask = episode == ep
            ii = np.repeat(mask, k)
            fg = G[ii].mean(0)
            per_episode.append(float(np.mean(np.sum((G[ii] - fg) ** 2, axis=1))))
        per_episode = np.asarray(per_episode)
        boots = []
        for _ in range(2000):
            boots.append(float(np.mean(per_episode[rng.integers(0, 16, 16)])))
        mc[name] = {
            "m": m, "draws": mc_draws, "mse": float(np.mean(mse)),
            "mse_se": float(np.std(mse, ddof=1) / np.sqrt(mc_draws)),
            "gradient_bias_norm": float(np.linalg.norm(est.mean(0) - full_g)),
            "gradient_second_moment": float(np.mean(np.sum(est ** 2, axis=1))),
            "ess_mean": float(np.mean(ess)),
            "weight_min": float(np.min(weights_all)), "weight_max": float(np.max(weights_all)),
            "episode_cluster_bootstrap_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
        }
    uniform_mse = mc["uniform"]["mse"]
    for v in mc.values(): v["mse_ratio_to_uniform"] = float(v["mse"] / max(uniform_mse, 1e-30))

    # Full-parameter audit is deliberately a small prefix; retain no per-item
    # full gradients.  The last-layer proxy is what supports the MC table.
    rank = rank_correlation(np.asarray(full_norms), np.asarray(last_norms))
    order = float(np.mean(np.argsort(np.argsort(full_norms)) == np.argsort(np.argsort(last_norms)))) if full_norms else float("nan")

    # Theta-old and in-memory one/three-step diagnostic update; no checkpoint write.
    theta = {"theta_old": {"ratio_mean": float(ratios.mean().cpu()), "clip_active_fraction": float(((ratios - 1).abs() > clips).float().mean().cpu())}}
    base_state = copy.deepcopy(model.state_dict())
    for steps in (1, 3):
        model.load_state_dict(base_state, strict=True)
        opt = torch.optim.AdamW(model.actor_ft.parameters(), lr=1e-4)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = term_values.new_zeros(())
            for a in range(0, len(G), 64):
                z = slice(a, min(a + 64, len(G)))
                tt, _, _ = actor_terms_from_processed_adv(model, {"state": obs["state"][z]}, prev[z], nxt[z], inds[z], proc[z], old[z], int(cfg.act_steps))
                loss = loss + tt.sum() / len(G)
            loss.backward(); opt.step()
        with torch.no_grad():
            rr = torch.empty(len(G), device=device); cc = torch.empty(len(G), device=device)
            for a in range(0, len(G), 64):
                z = slice(a, min(a + 64, len(G)),)
                _, r2, c2 = actor_terms_from_processed_adv(model, {"state": obs["state"][z]}, prev[z], nxt[z], inds[z], proc[z], old[z], int(cfg.act_steps))
                rr[z] = r2; cc[z] = c2
        theta[f"after_{steps}_diagnostic_update"] = {"ratio_mean": float(rr.mean().cpu()), "clip_active_fraction": float(((rr - 1).abs() > cc).float().mean().cpu())}
    model.load_state_dict(base_state, strict=True)

    out = {
        "task": task, "policy": policy, "buffer_path": str(buffer_path), "checkpoint_path": ckpt,
        "states": n_states, "items": len(G), "chain_shape": list(chains.shape),
        "fit_episodes": 8, "heldout_episodes": 8, "diagnostic_reward_scaler": fit_scale,
        "full_actor_gradient_audit": {"items": len(full_norms), "full_norm_min": min(full_norms), "full_norm_max": max(full_norms), "last_norm_min": min(last_norms), "last_norm_max": max(last_norms), "rank_correlation": rank, "exact_rank_fraction": order},
        "last_layer_dimension": last_dim, "gradient_collection_seconds": grad_seconds,
        "official_anchor_scalar_from_terms": full_scalar,
        "theta": theta, "proposals": {"epsilon": .2, "delta": delta, "timestep_step_v": step_v.tolist()},
        "mc_last_layer_gradient": mc,
        "toy_importance_correction": toy_expectation(),
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["square", "transport"], required=True)
    ap.add_argument("--policy", choices=["bc", "ft"], required=True)
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ans = analyze_task(args.task, args.policy, args.buffer, args.output, args.seed, args.device)
    print(json.dumps({"task": args.task, "policy": args.policy, "output": args.output, "items": ans["items"], "gradient_seconds": ans["gradient_collection_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
