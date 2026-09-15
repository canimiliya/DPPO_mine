"""CPU regression tests required before the S0c-R1 GPU recheck."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.distributions import Normal, kl_divergence

from s0cr1_core import (
    classify_gate,
    gaussian_kl,
    mmd2_unbiased,
    state_item_indices,
    true_ppo_blocked,
)


class S0CR1Tests(unittest.TestCase):
    def test_state_item_index_noncontiguous_full_coverage(self):
        states = np.array([2, 17, 63], dtype=np.int64)
        idx = state_item_indices(states, 10)
        expected = np.array([s * 10 + j for s in states for j in range(10)], dtype=np.int64)
        self.assert_array_equal(idx, expected)
        table = np.arange(64 * 10).reshape(64, 10)
        obs = table[..., None]
        prev = table[..., None] + 100
        nxt = table[..., None] + 200
        for arr in (obs, prev, nxt):
            self.assert_array_equal(arr.reshape(640, -1)[idx], arr[states].reshape(-1, 1))
        self.assert_array_equal(np.tile(np.arange(10), 3), np.array([j for _ in range(3) for j in range(10)]))
        toy = np.tile(np.arange(10, dtype=float), (64, 1))
        self.assertGreater(np.max(toy[states]), np.mean(toy[states]))

    def test_gaussian_kl_against_distribution(self):
        dtype = torch.float64
        old_mu = torch.tensor([[0.0, 1.0], [2.0, -1.0]], dtype=dtype)
        new_mu = torch.tensor([[0.0, 2.0], [1.0, -1.0]], dtype=dtype)
        old_lv = torch.tensor([[0.0, -0.3], [0.2, 0.7]], dtype=dtype)
        new_lv = torch.tensor([[0.0, -0.3], [-0.4, 0.1]], dtype=dtype)
        actual = gaussian_kl(old_mu, old_lv, new_mu, new_lv)
        expected = torch.stack([
            torch.stack([kl_divergence(Normal(old_mu[i, j], (old_lv[i, j] / 2).exp()),
                Normal(new_mu[i, j], (new_lv[i, j] / 2).exp())) for j in range(2)])
            for i in range(2)
        ])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(gaussian_kl(old_mu, old_lv, old_mu, old_lv), torch.zeros_like(old_mu)))
        self.assertGreater(float(actual[1, 0]), 0.0)  # unequal variance is exercised

    @staticmethod
    def _explicit_mmd(x, y, sigma):
        def k(a, b):
            return np.exp(-np.sum((a - b) ** 2) / (2 * sigma * sigma))
        xx = sum(k(x[i], x[j]) for i in range(len(x)) for j in range(len(x)) if i != j) / (len(x) * (len(x) - 1))
        yy = sum(k(y[i], y[j]) for i in range(len(y)) for j in range(len(y)) if i != j) / (len(y) * (len(y) - 1))
        xy = sum(k(x[i], y[j]) for i in range(len(x)) for j in range(len(y))) / (len(x) * len(y))
        return xx + yy - 2 * xy

    def test_independent_mmd_all_cross_terms_and_permutations(self):
        rng = np.random.default_rng(9)
        x, y, sigma = rng.normal(size=(5, 3)), rng.normal(loc=0.4, size=(7, 3)), 0.8
        expected = self._explicit_mmd(x, y, sigma)
        self.assertAlmostEqual(mmd2_unbiased(x, y, sigma), expected, places=14)
        ref = mmd2_unbiased(x, y, sigma)
        for px, py in ((rng.permutation(5), np.arange(7)), (np.arange(5), rng.permutation(7)),
                       (rng.permutation(5), rng.permutation(7))):
            self.assertAlmostEqual(mmd2_unbiased(x[px], y[py], sigma), ref, places=14)

    def test_gate_truth_table(self):
        def task(**kw):
            base = dict(n_scales_with_match=0, reliable_ratios=[], ab_agree=False,
                        energy_agree=False, paired_ci_support=False,
                        cheap_baselines_fail=False, denominator_reliable=False)
            base.update(kw)
            return base
        self.assertEqual(classify_gate(False, {}), "IMPLEMENTATION_INVALID")
        self.assertEqual(classify_gate(True, {
            "square": task(main_rho=.9, reliable_ratios=[1.5]),
            "transport": task(main_rho=.9, reliable_ratios=[1.5]),
        }), "SIMPLE_EXPLANATION_SUFFICIENT")
        residual = task(main_rho=.1, n_scales_with_match=2, reliable_ratios=[4.0],
                        ab_agree=True, energy_agree=True, paired_ci_support=True,
                        cheap_baselines_fail=True, denominator_reliable=True)
        self.assertEqual(classify_gate(True, {"square": residual, "transport": residual}), "RESIDUAL_SUPPORTED")
        self.assertEqual(classify_gate(True, {
            "square": task(main_rho=.1, n_scales_with_match=1, reliable_ratios=[1.5]),
            "transport": task(main_rho=.1, n_scales_with_match=1, reliable_ratios=[3.0]),
        }), "INCONCLUSIVE")
        self.assertEqual(classify_gate(True, {
            "square": task(main_rho=.1, n_scales_with_match=2, reliable_ratios=[4.0], ab_agree=True, energy_agree=True, paired_ci_support=True, cheap_baselines_fail=True, denominator_reliable=True),
            "transport": task(main_rho=.1, n_scales_with_match=0, reliable_ratios=[]),
        }), "INCONCLUSIVE")
        self.assertEqual(classify_gate(True, {
            "square": task(main_rho=.1, n_scales_with_match=2, reliable_ratios=[4.0], ab_agree=False, energy_agree=True, paired_ci_support=True, cheap_baselines_fail=True, denominator_reliable=True),
            "transport": task(main_rho=.1, n_scales_with_match=2, reliable_ratios=[4.0], ab_agree=False, energy_agree=True, paired_ci_support=True, cheap_baselines_fail=True, denominator_reliable=True),
        }), "INCONCLUSIVE")

    def test_true_ppo_blocked_is_not_ratio_outside(self):
        a = np.array([1.0, -1.0, 1.0, -1.0])
        r = np.array([1.2, .8, .8, 1.2])
        e = np.full(4, .1)
        self.assert_array_equal(true_ppo_blocked(a, r, e), [True, True, False, False])

    def assert_array_equal(self, a, b):
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
