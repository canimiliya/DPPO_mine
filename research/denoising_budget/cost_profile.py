"""Three-cycle bounded cost profile for the frozen diagnostic workload."""

from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from collect_frozen import load_frozen_model
from run_s0b import _batch_items, _full_advantages
from loss_terms import processed_advantages, actor_terms_from_processed_adv
from sampler import proposal_probabilities


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def profile(task, policy, buffer_path, output, device="cuda:0", cycles=3):
    data = np.load(buffer_path, allow_pickle=False)
    model, cfg, ckpt = load_frozen_model(task, policy, device)
    states, chains, obs, prev, nxt, inds, old = _batch_items(data, model, device)
    n = len(inds); k = chains.shape[1] - 1
    rewards = data["raw_reward"]
    adv_state = _full_advantages(rewards, scale=float(np.std(rewards[:, :8]) + 1e-6))
    bsteps = data["boundary_steps"].astype(int)
    ep = data["episode_id"].astype(int); bid = data["chunk_boundary_id"].astype(int)
    a = torch.from_numpy(np.asarray([adv_state[bsteps[b], e] for e, b in zip(ep, bid)], dtype=np.float32)).to(device).repeat_interleave(k)
    proc = processed_advantages(model, a, inds).detach()

    def actor_forward():
        s = 0
        for z in range(0, n, 64):
            t, _, _ = actor_terms_from_processed_adv(model, {"state": obs["state"][z:z+64]}, prev[z:z+64], nxt[z:z+64], inds[z:z+64], proc[z:z+64], old[z:z+64], int(cfg.act_steps))
            s = s + t.sum()
        return s / n

    with torch.no_grad():
        for _ in range(2): model({"state": torch.from_numpy(states[:16]).to(device)}, deterministic=False, return_chain=True)
    results = {x: [] for x in ["rollout_model_sampling", "old_logprob", "actor_forward", "actor_backward", "critic", "proposal", "io"]}
    for _ in range(cycles):
        sync(device); t=time.perf_counter();
        with torch.no_grad(): model({"state": torch.from_numpy(states[:16]).to(device)}, deterministic=False, return_chain=True)
        sync(device); results["rollout_model_sampling"].append(time.perf_counter()-t)
        sync(device); t=time.perf_counter();
        with torch.no_grad(): model.get_logprobs_subsample(obs, prev, nxt, inds)
        sync(device); results["old_logprob"].append(time.perf_counter()-t)
        sync(device); t=time.perf_counter(); loss=actor_forward(); sync(device); results["actor_forward"].append(time.perf_counter()-t)
        for p in model.actor_ft.parameters(): p.grad=None
        sync(device); t=time.perf_counter(); loss.backward(); sync(device); results["actor_backward"].append(time.perf_counter()-t)
        sync(device); t=time.perf_counter();
        with torch.no_grad(): model.critic(obs).view(-1)
        sync(device); results["critic"].append(time.perf_counter()-t)
        t=time.perf_counter(); proposal_probabilities(np.abs(np.random.default_rng(1).normal(size=n))); results["proposal"].append(time.perf_counter()-t)
        t=time.perf_counter(); np.load(buffer_path, allow_pickle=False); results["io"].append(time.perf_counter()-t)
    ms={k:float(np.mean(v)*1000) for k,v in results.items()}
    baseline=sum(ms.values())
    actor=ms["actor_forward"]+ms["actor_backward"]
    f=actor/baseline if baseline else float("nan")
    s=actor/(0.5*actor) if actor else float("nan")
    h=(ms["proposal"]+ms["io"])/baseline if baseline else float("nan")
    predicted=1/(1-f+f/s+h) if np.isfinite(f+s+h) else float("nan")
    out={"task":task,"policy":policy,"checkpoint_path":ckpt,"cycles":cycles,"timing_ms_mean":ms,"baseline_total_ms":baseline,"actor_reducible_fraction_f":f,"assumed_actor_speedup_s":s,"proposal_io_overhead_h":h,"predicted_total_speedup_with_50pct_actor":predicted,"note":"rollout_model_sampling is model-only on 16 states; full environment wall time is not silently substituted."}
    Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,sort_keys=True))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument('--task',required=True); ap.add_argument('--policy',required=True); ap.add_argument('--buffer',required=True); ap.add_argument('--output',required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--cycles',type=int,default=3); a=ap.parse_args(); profile(a.task, a.policy, a.buffer, a.output, a.device, a.cycles)
