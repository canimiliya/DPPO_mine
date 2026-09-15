"""Exact, diagnostic-only decomposition of the official DPPO actor loss.

This module deliberately does not alter ``source/dppo_v0.6``.  It mirrors the
ordering in ``PPODiffusion.loss`` and returns the post-PPO-clipping loss for
each (anchor, denoising-step) item.  Advantage normalization and quantile
clipping are performed on the complete anchor supplied by the caller.
"""

from __future__ import annotations

import math
from typing import Dict

import torch


def actor_loss_terms(
    model,
    obs: Dict[str, torch.Tensor],
    chains_prev: torch.Tensor,
    chains_next: torch.Tensor,
    denoising_inds: torch.Tensor,
    advantages: torch.Tensor,
    oldlogprobs: torch.Tensor,
    reward_horizon: int,
) -> dict[str, torch.Tensor]:
    """Return official-equivalent scalar inputs and an ``[N]`` actor vector.

    Inputs use the same flattened item convention as ``PPODiffusion.loss``:
    one row is one anchor state paired with one denoising index.  The returned
    vector is already averaged over the requested action horizon and action
    dimensions, but not over rows.
    """
    newlogprobs, eta = model.get_logprobs_subsample(
        obs, chains_prev, chains_next, denoising_inds, get_ent=True
    )
    newlogprobs = newlogprobs.clamp(min=-5, max=2)
    oldlogprobs = oldlogprobs.clamp(min=-5, max=2)
    newlogprobs = newlogprobs[:, :reward_horizon, :]
    oldlogprobs = oldlogprobs[:, :reward_horizon, :]
    new_mean = newlogprobs.mean(dim=(-1, -2)).reshape(-1)
    old_mean = oldlogprobs.mean(dim=(-1, -2)).reshape(-1)

    adv = advantages.clone()
    if model.norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv_min = torch.quantile(adv, model.clip_advantage_lower_quantile)
    adv_max = torch.quantile(adv, model.clip_advantage_upper_quantile)
    adv = adv.clamp(min=adv_min, max=adv_max)

    exponent = model.ft_denoising_steps - denoising_inds - 1
    discount = torch.pow(
        torch.tensor(model.gamma_denoising, device=adv.device, dtype=adv.dtype),
        exponent.to(dtype=adv.dtype),
    )
    adv = adv * discount

    logratio = new_mean - old_mean
    ratio = logratio.exp()
    t = (denoising_inds.float() / (model.ft_denoising_steps - 1)).to(model.device)
    if model.ft_denoising_steps > 1:
        clip_coef = model.clip_ploss_coef_base + (
            model.clip_ploss_coef - model.clip_ploss_coef_base
        ) * (torch.exp(model.clip_ploss_coef_rate * t) - 1) / (
            math.exp(model.clip_ploss_coef_rate) - 1
        )
    else:
        clip_coef = t
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    terms = torch.maximum(unclipped, clipped)
    return {
        "terms": terms,
        "advantages_after_processing": adv,
        "new_logprob_mean": new_mean,
        "old_logprob_mean": old_mean,
        "logratio": logratio,
        "ratio": ratio,
        "clip_coef": clip_coef,
        "eta": eta,
    }


def official_actor_scalar(model, *args, **kwargs) -> torch.Tensor:
    """Call the official loss and return its actor scalar only."""
    return model.loss(*args, **kwargs)[0]


def processed_advantages(model, advantages, denoising_inds):
    """Apply the official full-anchor advantage processing exactly once."""
    adv = advantages.clone()
    if model.norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv_min = torch.quantile(adv, model.clip_advantage_lower_quantile)
    adv_max = torch.quantile(adv, model.clip_advantage_upper_quantile)
    adv = adv.clamp(min=adv_min, max=adv_max)
    exponent = model.ft_denoising_steps - denoising_inds - 1
    discount = torch.pow(
        torch.tensor(model.gamma_denoising, device=adv.device, dtype=adv.dtype),
        exponent.to(dtype=adv.dtype),
    )
    return adv * discount


def actor_terms_from_processed_adv(
    model, obs, chains_prev, chains_next, denoising_inds,
    processed_adv, oldlogprobs, reward_horizon,
):
    """Evaluate terms when normalization was already done on the full anchor."""
    newlogprobs = model.get_logprobs_subsample(
        obs, chains_prev, chains_next, denoising_inds, get_ent=False
    ).clamp(min=-5, max=2)
    oldlogprobs = oldlogprobs.clamp(min=-5, max=2)
    new_mean = newlogprobs[:, :reward_horizon, :].mean(dim=(-1, -2)).reshape(-1)
    old_mean = oldlogprobs[:, :reward_horizon, :].mean(dim=(-1, -2)).reshape(-1)
    logratio = new_mean - old_mean
    ratio = logratio.exp()
    t = (denoising_inds.float() / (model.ft_denoising_steps - 1)).to(model.device)
    clip_coef = model.clip_ploss_coef_base + (
        model.clip_ploss_coef - model.clip_ploss_coef_base
    ) * (torch.exp(model.clip_ploss_coef_rate * t) - 1) / (
        math.exp(model.clip_ploss_coef_rate) - 1
    ) if model.ft_denoising_steps > 1 else t
    return torch.maximum(-processed_adv * ratio, -processed_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)), ratio, clip_coef
