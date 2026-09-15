# S0D_R2_R1_BOUNDARY Protocol

Protocol frozen before formal Square data collection. This is a diagnostic-only
boundary measurement; it does not authorize training, optimizer steps,
Transport, mechanism intervention, or any later research route.

- Task: `S0D_R2_R1_BOUNDARY`
- Base commit: `af4ee18bc58b010ff8ddc1cf3f6ced448319ebbd`
- Diagnostic branch: `codex/s0d-r2-r1-boundary`
- Runtime: existing legacy MuJoCo 2.1 / `mujoco-py`; no dependency changes.
- Wall-clock cap: 240 minutes; no new evaluation batch after 210 minutes.
- New atomic simulator-step cap: 700,000.
- Task scope: Square only, `state_0` BC-EMA and `state_200` FT; Transport is `NOT_RUN`.
- Square: observation 23, action 7, action chunk 4, horizon 400 atomic steps.
- Sampling: official v0.6 DDPM evaluation sampler with `deterministic=True`,
  `model.eval()` and `torch.no_grad()`. This flag retains the official DDPM
  schedule, initial noise, per-step noise and final-step rule; it is not DDIM,
  ODE, or a zero-noise sampler.

## Frozen seeds and pulse domain

- FIT seeds: `930001, 930002, 930003, 930004`; FIT is calibration/precheck only.
- TEST reset seeds: integers `940001` through `940024`, all 24 seeds.
- TEST continuation draws: four per policy and magnitude; draws 1--2 are block A
  and draws 3--4 are block B.
- Bootstrap: 5,000 reset-seed-cluster resamples, seed `950001`, two-sided 95%
  percentile intervals.
- Anchor boundary: zero-based TEST seed-list index cycles through chunk 10, 20,
  40. At most one anchor per parent policy and seed.
- Pulse direction: by zero-based TEST seed-list index modulo four,
  `+x, -x, +y, -y`.
- Target pulse magnitudes: `[0, 0.125, 0.25, 0.5]` environment action units;
  default multiplier is 1.0. FIT may reduce the whole multiplier once to 0.5
  only if the maximum 0.5 pulse clip ratio exceeds 5%; this preregistered rule
  is applied before TEST and cannot be changed afterward.
- A pulse modifies robot-0 Cartesian translation only, on the first four
  atomic actions of one complete chunk. For Square the whole chunk is modified.
  The action is modified after inverse normalization in the controller input
  coordinates and then clipped by the official environment action bounds.

## Physical semantics

The resolved Square environment uses robosuite `OSC_POSE`, `control_delta=True`,
`input_min=-1`, `input_max=1`, and controller coordinates `[dx, dy, dz,
axis-angle-x, axis-angle-y, axis-angle-z]` in the controller's documented
relative command convention. The pulse indices are therefore x=0 and y=1;
this was checked against the resolved `OperationalSpaceController` source and
not inferred only from the action dimension. No pose teleportation or
observation fabrication is allowed.

Native success is queried by the actual environment `_check_success()` after
every atomic simulator step. Reward thresholds, chunk reward, and custom
success proxies are not outcomes.

## Fixed workflow

1. Verify checkpoint/normalization hashes, actor identity, runtime and official
   sampler semantics.
2. Run FIT sampling sensitivity, sampler equivalence, same-snapshot replay,
   cross-reset restore, BC-BC paired, and gate self-tests.
3. Calibrate physical pulse clipping and displacement; freeze the resulting
   multiplier before TEST.
4. Run complete BC and FT parent rollouts for all 24 TEST seeds. Parent nominal
   success is not truncated at the anchor.
5. For every valid anchor and every frozen pulse magnitude, execute sham or
   physical pulse from the same snapshot, then save the continuation snapshot.
   Retain all pulse records. Main paired analysis includes only anchors for
   which every magnitude branch is still non-success and non-done at pulse end;
   this symmetric, preregistered exclusion is not outcome re-selection.
6. Run all four paired continuation draws for BC and FT at every retained
   magnitude. Missing or partial reset-seed clusters are not imputed and do not
   enter the primary estimate.

## Estimands and gates

For each magnitude `r`, `q_policy(r)` is the reset-seed-cluster-weighted mean
native-ever-success over the retained paired anchors and their four TEST draws.

`L(r) = q_BC(r) - q_FT(r)`; `L0 = L(0)`; `Lbar = mean(L(0.125), L(0.25), L(0.5))`;
`Ibar = Lbar - L0`; `Jnom = mean_seed(FT_parent_success - BC_parent_success)`.
All bootstrap units are complete reset-seed clusters, preserving source,
magnitude, policy and draw pairing. No screening, denominator conditioning,
direction selection or max/min selection is allowed.

`MEASUREMENT_VALID` requires all prechecks, all 24 planned parent clusters,
at least 16 complete main paired reset-seed clusters, and complete pulse fields:
target/actual pulse, clip ratio, EEF/object displacement relative to sham,
success/done timing and remaining horizon.

`CONTEXT_VALID` requires BC sham success >= 0.50, BC nonzero-pulse mean success
>= 0.20, and `Jnom >= 0`.

`PHENOMENON_PASS` requires implementation, measurement and context validity,
`Lbar >= 0.10`, `Ibar >= 0.10`, positive 95% CI lower bounds for both, and
positive `Lbar_A/Ibar_A` and `Lbar_B/Ibar_B`. This only requests mechanism
review; it is not a mechanism, algorithm, or paper pass.

`PHENOMENON_FAIL` is allowed only with valid implementation/measurement/context
and an upper CI bound below 0.10 for `Lbar` or `Ibar`. Every other case is
`INCONCLUSIVE` and `NO_GO`.

## Immutable configuration

The machine-readable frozen values are in
`research/recovery_retention/s0d_r2_r1_config.json`. The protocol SHA256 and
configuration SHA256 are recorded in the manifest. Once TEST begins neither
file may be changed.
