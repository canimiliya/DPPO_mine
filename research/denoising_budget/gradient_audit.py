"""Streaming gradient audit helpers for frozen diagnostic anchors."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def trainable_parameters(model):
    return [p for p in model.actor_ft.parameters() if p.requires_grad]


def last_layer_parameters(model):
    params = trainable_parameters(model)
    # The final linear projection is named ``final`` in DiffusionMLP.  Fall
    # back to the last parameter pair if a future architecture renames it.
    named = [(n, p) for n, p in model.actor_ft.named_parameters() if p.requires_grad]
    selected = [(n, p) for n, p in named if "out" in n or "final" in n]
    if selected:
        return [p for _, p in selected]
    return params[-2:]


def vector_grad(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])


def item_gradient_audit(model, term_fn, limit: int = 16):
    """Compute full and last-layer norms for a small prefix, streaming only."""
    full_params = trainable_parameters(model)
    last_params = last_layer_parameters(model)
    full_norms, last_norms = [], []
    n = min(limit, int(term_fn("n_items")))
    for i in range(n):
        term = term_fn(i)
        full = vector_grad(term, full_params)
        # Recompute the scalar graph for the last-layer vector; no giant tensor
        # is retained after the norm is recorded.
        term_last = term_fn(i)
        last = vector_grad(term_last, last_params)
        full_norms.append(float(full.norm().detach().cpu()))
        last_norms.append(float(last.norm().detach().cpu()))
    return {"n": n, "full_norm": np.asarray(full_norms), "last_norm": np.asarray(last_norms)}


def cheap_features(obs, j, advantage, chain_prev, chain_next, old_logprob):
    state = obs.reshape(obs.shape[0], -1)
    state_summary = torch.stack([state.mean(-1), state.std(-1), state.abs().mean(-1)], -1)
    delta = (chain_next - chain_prev).reshape(chain_prev.shape[0], -1)
    return torch.cat(
        [state_summary, torch.as_tensor(j, device=state.device).reshape(-1, 1).float(),
         torch.as_tensor(advantage, device=state.device).reshape(-1, 1),
         chain_prev.reshape(chain_prev.shape[0], -1).norm(dim=-1, keepdim=True),
         delta.norm(dim=-1, keepdim=True), old_logprob.reshape(old_logprob.shape[0], -1).mean(-1, keepdim=True)],
        -1,
    )


def rank_correlation(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
