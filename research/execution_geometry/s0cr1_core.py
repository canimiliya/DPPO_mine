"""Pure, testable measurement primitives for S0c-R1.

This file contains no environment, checkpoint, optimizer, or GPU code.  The
runner imports these functions so the scientific definitions used by the
diagnostic and the CPU regression tests are identical.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def state_item_indices(state_idx: np.ndarray, k: int) -> np.ndarray:
    """Return flattened rows for complete [state, denoising-item] blocks."""
    state_idx = np.asarray(state_idx, dtype=np.int64)
    if state_idx.ndim != 1 or k <= 0:
        raise ValueError("state_idx must be 1-D and k positive")
    return (state_idx[:, None] * int(k) + np.arange(int(k), dtype=np.int64)[None, :]).reshape(-1)


def gaussian_kl(old_mu, old_lv, new_mu, new_lv):
    """Elementwise KL(old || new) for diagonal Gaussians in log-variance form."""
    old_var, new_var = old_lv.exp(), new_lv.exp()
    return 0.5 * (new_lv - old_lv + (old_var + (old_mu - new_mu).square()) / new_var - 1.0)


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Unbiased independent-sample RBF MMD^2, retaining negative estimates."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or len(x) < 2 or len(y) < 2:
        raise ValueError("x/y must be [samples, features] with >=2 samples")
    if not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError("sigma must be positive")
    def kernel(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-d2 / (2.0 * sigma * sigma))
    kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(
        kxx.sum() / (len(x) * (len(x) - 1))
        + kyy.sum() / (len(y) * (len(y) - 1))
        - 2.0 * kxy.mean()
    )


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or len(x) < 2 or len(y) < 2:
        raise ValueError("x/y must be [samples, features] with >=2 samples")
    def dist(a, b):
        return np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1), 0.0))
    xx, yy, xy = dist(x, x), dist(y, y), dist(x, y)
    np.fill_diagonal(xx, 0.0)
    np.fill_diagonal(yy, 0.0)
    return float(2.0 * xy.mean() - xx.sum() / (len(x) * (len(x) - 1)) - yy.sum() / (len(y) * (len(y) - 1)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        return np.bincount(inv, weights=r)[inv] / counts[inv]
    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or len(x) != len(y):
        raise ValueError("affine fit needs matching arrays with >=2 rows")
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def true_ppo_blocked(advantages: np.ndarray, ratio: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Correct PPO blocked condition, distinct from ratio-outside-clip."""
    a, r, e = np.asarray(advantages), np.asarray(ratio), np.asarray(eps)
    return ((a > 0) & (r > 1.0 + e)) | ((a < 0) & (r < 1.0 - e))


def classify_gate(implementation_valid: bool, tasks: dict[str, dict[str, Any]]) -> str:
    """Apply the registered G1 truth table without a global max-ratio shortcut."""
    if not implementation_valid:
        return "IMPLEMENTATION_INVALID"
    if not tasks or set(tasks) != {"square", "transport"}:
        return "INCONCLUSIVE"
    rho_ok = all(float(v.get("main_rho", float("nan"))) >= 0.8 for v in tasks.values())
    reliable_ratios = [float(r) for v in tasks.values() for r in v.get("reliable_ratios", [])]
    if rho_ok and reliable_ratios and max(reliable_ratios) <= 2.0:
        return "SIMPLE_EXPLANATION_SUFFICIENT"
    residual_ok = all(
        int(v.get("n_scales_with_match", 0)) >= 2
        and any(float(r) >= 3.0 for r in v.get("reliable_ratios", []))
        and bool(v.get("ab_agree"))
        and bool(v.get("energy_agree"))
        and bool(v.get("paired_ci_support"))
        and bool(v.get("cheap_baselines_fail"))
        and bool(v.get("denominator_reliable"))
        for v in tasks.values()
    )
    return "RESIDUAL_SUPPORTED" if residual_ok else "INCONCLUSIVE"
