"""Importance-correct diagnostic samplers; no training integration."""

from __future__ import annotations

import numpy as np


def proposal_probabilities(v_hat, epsilon: float = 0.2, delta: float = 1e-8):
    """Return the preregistered mixture proposal, with uniform fallback."""
    v = np.asarray(v_hat, dtype=np.float64)
    score = np.sqrt(np.maximum(v, 0.0) + float(delta))
    denom = float(score.sum())
    n = len(score)
    if n == 0 or not np.isfinite(denom) or denom <= 0:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    p = (1.0 - epsilon) * score / denom + epsilon / n
    p = p / p.sum()
    return p


def draw_importance_corrected(loss_terms, p, m, rng):
    """Draw with replacement and return the unbiased loss/gradient weights."""
    loss_terms = np.asarray(loss_terms)
    p = np.asarray(p, dtype=np.float64)
    n = len(loss_terms)
    if n == 0 or len(p) != n or np.any(p <= 0):
        raise ValueError("p must be positive and match loss_terms")
    indices = rng.choice(n, size=int(m), replace=True, p=p)
    weights = 1.0 / (n * p[indices])
    return indices, weights, weights / len(indices)


def toy_expectation(seed: int = 0):
    """Exact-enumeration float64 gate for a tiny finite sampling problem."""
    import itertools

    losses = np.array([0.7, -1.1, 2.4], dtype=np.float64)
    grads = np.array([[1.0, 2.0], [-2.0, 0.5], [0.2, -1.5]], dtype=np.float64)
    p = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    m = 3
    full_loss = losses.mean()
    full_grad = grads.mean(axis=0)
    expected_loss = 0.0
    expected_grad = np.zeros(2, dtype=np.float64)
    for draw in itertools.product(range(len(losses)), repeat=m):
        prob = float(np.prod(p[list(draw)]))
        w = 1.0 / (len(losses) * p[list(draw)])
        expected_loss += prob * float(np.mean(losses[list(draw)] * w))
        expected_grad += prob * np.mean(grads[list(draw)] * w[:, None], axis=0)
    return {
        "full_loss": full_loss,
        "expected_loss": expected_loss,
        "loss_abs_error": abs(expected_loss - full_loss),
        "full_grad": full_grad.tolist(),
        "expected_grad": expected_grad.tolist(),
        "gradient_abs_error": float(np.max(np.abs(expected_grad - full_grad))),
        "m": m,
        "n": len(losses),
        "p": p.tolist(),
    }
