# S0c-R1 Protocol

## Scope

This is a known-implementation-error repair of S0c, not a new research
hypothesis and not an algorithm-training run.  The prior S0c scientific
conclusion is withdrawn until this corrected measurement is audited.

The gradient-budget / BDUA gate remains `STOP`.  This protocol permits only
measurement implementation correction and one bounded recheck.  It forbids
new environment rollouts, optimizer steps, formal training, S1, R2, G2, the
4-pair estimator, reward/GAE/configuration changes, new samples, bandwidth
sweeps, and changes to the official submodule pointer.

## Frozen design

- Inputs: the existing Square FT `state_200.pt` and Transport FT `state_100.pt`
  together with the two S0b frozen buffers and S0A/S0B manifests.
- 16 episodes; FIT episodes 0--7; TEST episodes 8--15; four boundaries per
  episode; three directions (`D_all`, `D_first`, `D_last`).
- Targets are exactly `0.001`, `0.003`, and `0.01`; each target uses at most
  eight bracket/bisection evaluations and a +/-10% achieved-budget tolerance.
- Each reference policy/candidate uses 64-sample independent blocks A and B.
  Paired noise is retained only for the auxiliary fixed-tape action L2.
- `DIAGNOSTIC_ADVANTAGE` is derived from the frozen reward buffer and is not
  historical GAE.  It is never used to claim historical training recovery.

## Corrected measurements

1. State-to-item indexing is one shared helper:
   `state_idx[:,None] * K + arange(K)`, tested on non-contiguous states
   `[2,17,63]` and used for `obs`, `prev`, `next`, `inds`, `path_gaussian`,
   `direction_budget`, and `gaussian_metrics`.
2. Gaussian `KL(old || new)` uses
   `0.5*(new_lv-old_lv + (exp(old_lv)+(old_mu-new_mu)^2)/exp(new_lv)-1)`.
   The registered sampling standard-deviation floor remains in the diagnostic;
   this is not an exact KL of a clipped kernel.
3. The main execution marginal distance is unbiased independent-sample
   MMD-squared with all cross terms.  Energy distance is an independent
   distance-definition check; paired action L2 is auxiliary only.
4. The RBF bandwidth is one locked median heuristic from Square FIT old-policy
   samples flattened to the full `horizon * action_dim`; no sweep is allowed.
5. Affine RMSE calibration is fitted once per cheap latent baseline on Square
   FIT state-by-candidate pairs, including an intercept, then locked for Square
   TEST and Transport FIT/TEST.  TEST and Transport are never refit.
6. Bootstrap resamples TEST episodes as clusters (2000 draws); all boundaries,
   directions, targets, and A/B values within an episode move together.  The
   ordered direction pairs are fixed as `(D_all,D_first)`, `(D_all,D_last)`,
   `(D_first,D_last)`; max/min is never reselected inside bootstrap.

## Gate

The primary Spearman statistic is raw per-candidate Spearman over TEST states,
then an equal candidate average.  Per-candidate and pooled values are reported;
pooled rho is not used by the gate.

- `SIMPLE_EXPLANATION_SUFFICIENT`: both tasks have main held-out rho >= 0.8
  and every reliable matched-budget execution-marginal residual ratio is <=2.
- `RESIDUAL_SUPPORTED`: both tasks have at least two predetermined scales with
  matched Gaussian budgets and MMD residual ratio >=3, with A/B agreement,
  energy agreement, paired cluster-CI support, denominator above the noise
  floor, and cheap latent baselines insufficient.
- Otherwise `INCONCLUSIVE`.  Any regression/adapter implementation failure is
  `IMPLEMENTATION_INVALID`.

Even `RESIDUAL_SUPPORTED` does not authorize G2, S1, BDUA, or formal training.
The online measurement gate and cost gate remain `NOT_RUN`.

## Required accounting

Reports are generated from actual outputs, not fixed historical numbers.  They
distinguish full-task start/end from diagnostic-script seconds and record
`NEW_ENVIRONMENT_STEPS=0`, `FORMAL_TRAINING_ITERATIONS=0`, and
`OPTIMIZER_STEPS=0`.  Large arrays and seed metadata are written only below
`D:\AgentData\DPPO-S0c-R1`.  Existing S0C reports/CSV, checkpoints, logs,
dirty submodule contents, and history remain untouched.
