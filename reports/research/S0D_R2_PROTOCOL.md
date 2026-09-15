# S0d-R2 protocol: local recovery-retention screening

Protocol frozen before any S0d-R2 environment evaluation.

- Protocol start: `2026-09-16T01:20:37.1925612+08:00`.
- Task: `S0d-R2`; decision route B; `CURRENT_GRADIENT_BUDGET=STOP`; `CURRENT_R1=STOP`.
- Base commit: `0124fc5b1e22bf58523de844bcfcc572eac2caa7`; diagnostic branch: `codex/s0d-r2`.
- Only Square and Transport, comparing the audited `state_0` BC-EMA policy with the specified existing `state_200` FT checkpoint.
- Runtime must remain the existing legacy MuJoCo 2.1 / mujoco-py stack. No dependency changes, training, optimizer construction or optimizer step.
- Hard wall-clock limit: 240 minutes from audit/coding start. Stop launching new evaluation batches at 210 minutes. New atomic simulator steps: at most 1,100,000.

## Fixed assets and semantics

- Square: `obs=23`, `action=7`, `chunk=4`, `max_episode_steps=400`.
- Transport: `obs=59`, `action=14`, `chunk=8`, `max_episode_steps=800`.
- Both use the existing normalization, action bounds, controller, 20 DDPM steps, 10 FT steps, official evaluation sampler semantics, noise clipping and `model.eval()+no_grad`.
- The checkpoint actors are loaded from the BC `ema`-equivalent base actor for `state_0`; no non-EMA BC substitute is allowed.
- Native success is recorded from the environment's `_check_success()` on every atomic step. Reward-derived `max(chunk_reward)/act_steps` is not used as a success substitute.

## Preflight and snapshot validity

Before scientific data collection, the implementation must pass all of the following with real exit codes recorded in the manifest:

1. Native `_check_success` adapter is callable for each task.
2. A complete evaluation snapshot includes physics state, controller targets/caches, observation history, wrapper time/action queue and required RNG state. Restoring only qpos/qvel is invalid.
3. Same snapshot plus fixed action tape, replayed twice for at least 8 atomic steps, has per-step observation/state maximum error `<=1e-6` and identical success/done.
4. Paired BC-vs-BC effect is zero within `1e-8`.
5. New loader and official sampler agree on fixed observations/noise tapes with maximum action/chain error `<=1e-6` for state_0/state_200.
6. The actual gate implementation passes synthetic PASS, FAIL, threshold-crossing INCONCLUSIVE, coverage, one-task-only, sham-degenerate and implementation-invalid cases.
7. The runner records actual test exit codes and does not write constant PASS values.

Pilot reset seeds are exactly `910001, 910002`; pilot data are not scientific anchors and cannot determine pulse size.

## Anchors and pulse

Formal reset seeds are exactly `920001..920016`, reused across tasks. Each seed runs complete BC and FT parent episodes to natural termination or the task horizon; nominal success is not truncated at an anchor. At most one anchor is retained per parent: zero-based seed-list index even after chunk 20, odd after chunk 40. An anchor already successful or terminated before its boundary is invalid and is retained only as a reasoned missing record.

At an anchor, source-policy noise generates one complete action chunk `u`, saved as a fixed tape. The sham executes `u`. The perturbed branch adds a fixed `delta=0.25` environment action unit to one robot-0 Cartesian translation component only on the first four atomic actions, then applies the official action-bound clip. Directions by zero-based index modulo four are `+x,-x,+y,-y`. The pulse is applied after inverse normalization in the controller action coordinates; no teleport or observation fabrication is permitted. For Transport, actions 5--8 remain unchanged. Both branches execute a complete chunk and preserve the original remaining episode time.

## Screening and independent test

Stage 1 uses independent BC-only screening continuation draws. Each valid anchor's sham and perturbed branches receive four BC draws. An anchor enters empirical set `C` only when both conditions succeed at least 3/4. `C` is explicitly an empirical screening set, not a proof that `q_BC>=0.75`. Required coverage is at least 12 cases, at least 8 reset-seed clusters, and at least 4 BC-source and 4 FT-source cases.

Stage 2 uses four fresh continuation draws per case for BC/sham, FT/sham, BC/perturbed and FT/perturbed. Draws 1--2 are block A and 3--4 block B. Screening successes are excluded from effect estimates. Incomplete cases are missing, not imputed.

For each complete case, `q` is the mean native-ever-success over its four TEST draws. The estimands are:

```text
L    = mean_C(q_BC_pert - q_FT_pert)
L0   = mean_C(q_BC_sham - q_FT_sham)
I    = L - L0
Jnom = mean_16resetseeds(native_FT_parent_success - native_BC_parent_success)
```

Bootstrap unit is reset-seed cluster, keeping all paired cases, conditions and draws together; use 5000 fixed-seed cluster bootstrap resamples and unconditioned two-sided 95% percentile intervals for L and I. Do not reselect C, condition on denominator, choose a direction, or take maxima/minima. Report block A/B L, pulse directions and source policies descriptively only.

## Gates and stopping

`IMPLEMENTATION_PASS` requires all real prechecks, matching input hashes/semantics and complete budgeted data. `MEASUREMENT_VALID` requires complete paired screening/TEST and valid coverage with no success/reset/time mismatch.

`CONTEXT_PASS` requires independent TEST BC perturbed success >=0.60, BC sham success >=0.60, `Jnom>=0.10`, and mean FT sham minus BC sham >=-0.05.

`PHENOMENON_PASS` additionally requires `L>=0.15`, `I>=0.15`, both 95% CI lower bounds >0, and `L_A>0`, `L_B>0`. `PHENOMENON_FAIL` requires implementation, measurement and context validity plus an upper CI bound for L or I below 0.15. All remaining states are `PHENOMENON_INCONCLUSIVE` and NO-GO.

Run Square first. If Square is not PHENOMENON_PASS, Transport main screening is `NOT_RUN` and no Transport result is labelled FAIL. Only two task PASS results permit `GO_FOR_UPPER_REVIEW`; this never authorizes training.

Forbidden in this protocol: BDUA, gradient-budget recovery, S0c/G2/R1 follow-up, optimizer steps, new predictors, recovery routers, parameter interpolation, denoising-step replacement, BC regularization, checkpoint sweep, new algorithm training, formal long run, extra anchors/seeds/pulse sizes, or automatic continuation into N1/N2.
