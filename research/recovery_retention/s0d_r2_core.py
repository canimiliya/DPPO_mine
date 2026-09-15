"""Small, dependency-light core for the S0d-R2 recovery-retention audit."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def stable_seed(*parts: object) -> int:
    import hashlib

    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**31 - 1)


def mean_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 5000, seed: int = 20260916):
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if len(values) == 0:
        return {"estimate": None, "lower": None, "upper": None, "clusters": 0}
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    groups = {u: values[clusters == u] for u in unique}
    for i in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        boot[i] = np.mean(np.concatenate([groups[u] for u in chosen]))
    return {
        "estimate": float(np.mean(values)),
        "lower": float(np.percentile(boot, 2.5)),
        "upper": float(np.percentile(boot, 97.5)),
        "clusters": int(len(unique)),
    }


def gate_task(metrics: dict) -> dict:
    """The registered task gate; no hidden shortcut or constant PASS."""
    impl = bool(metrics.get("implementation_pass", False))
    valid = bool(metrics.get("measurement_valid", False))
    coverage = bool(metrics.get("coverage_pass", False))
    context = bool(
        metrics.get("bc_pert_success", -math.inf) >= 0.60
        and metrics.get("bc_sham_success", -math.inf) >= 0.60
        and metrics.get("jnom", -math.inf) >= 0.10
        and metrics.get("ft_minus_bc_sham", math.inf) >= -0.05
    )
    phenomenon = bool(
        impl and valid and coverage and context
        and metrics.get("L", -math.inf) >= 0.15
        and metrics.get("I", -math.inf) >= 0.15
        and metrics.get("L_ci_lower", -math.inf) > 0
        and metrics.get("I_ci_lower", -math.inf) > 0
        and metrics.get("L_A", -math.inf) > 0
        and metrics.get("L_B", -math.inf) > 0
    )
    fail = bool(
        impl and valid and coverage and context
        and (metrics.get("L_ci_upper", math.inf) < 0.15
             or metrics.get("I_ci_upper", math.inf) < 0.15)
    )
    if not impl:
        gate = "IMPLEMENTATION_FAIL"
    elif not valid:
        gate = "PHENOMENON_INCONCLUSIVE"
    elif phenomenon:
        gate = "PHENOMENON_PASS"
    elif fail:
        gate = "PHENOMENON_FAIL"
    else:
        gate = "PHENOMENON_INCONCLUSIVE"
    return {
        "coverage_pass": coverage,
        "context_pass": context,
        "phenomenon_pass": phenomenon,
        "phenomenon_fail": fail,
        "gate": gate,
    }


def gate_self_test() -> dict:
    base = {
        "implementation_pass": True, "measurement_valid": True, "coverage_pass": True,
        "bc_pert_success": .8, "bc_sham_success": .8, "jnom": .2,
        "ft_minus_bc_sham": 0., "L": .2, "I": .2, "L_ci_lower": .05,
        "I_ci_lower": .05, "L_ci_upper": .4, "I_ci_upper": .4, "L_A": .1, "L_B": .1,
    }
    cases = {
        "pass": (base, "PHENOMENON_PASS"),
        "fail": ({**base, "L": .1, "L_ci_lower": .01, "L_ci_upper": .14}, "PHENOMENON_FAIL"),
        "threshold_inconclusive": ({**base, "L_ci_lower": -.01, "L_ci_upper": .3}, "PHENOMENON_INCONCLUSIVE"),
        "coverage_inconclusive": ({**base, "coverage_pass": False}, "PHENOMENON_INCONCLUSIVE"),
        "one_task_not_joint": ({**base, "measurement_valid": False}, "PHENOMENON_INCONCLUSIVE"),
        "sham_context_fail": ({**base, "bc_sham_success": .4}, "PHENOMENON_INCONCLUSIVE"),
        "implementation_fail": ({**base, "implementation_pass": False}, "IMPLEMENTATION_FAIL"),
    }
    result = {}
    for name, (metrics, expected) in cases.items():
        got = gate_task(metrics)["gate"]
        result[name] = {"expected": expected, "observed": got, "pass": got == expected}
    return result
