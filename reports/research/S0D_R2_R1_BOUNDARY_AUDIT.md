# S0D_R2_R1_BOUNDARY recovery-retention boundary audit

- TASK: **S0D_R2_R1_BOUNDARY**
- STATUS: **INCONCLUSIVE**
- CURRENT_GRADIENT_BUDGET: **STOP**
- CURRENT_R1: **STOP**
- IMPLEMENTATION: **PASS**
- MEASUREMENT: **VALID**
- CONTEXT: **FAIL**
- PHENOMENON: **INCONCLUSIVE**
- MECHANISM_GATE: **NOT_RUN**
- ALGORITHM_EFFECTIVENESS: **NOT_TESTED**
- NOVELTY_STATUS: **CANDIDATE_ONLY**
- NEXT_STAGE: **NO_GO**
- FORMAL_TRAINING_ITERATIONS: **0**
- OPTIMIZER_STEPS: **0**
- NEW_ENVIRONMENT_ATOMIC_STEPS: **252116**
- NEW_ENVIRONMENT_CHUNK_STEPS: **63019**
- WALL_CLOCK_SECONDS: **3443.791**
- TIME_LIMIT_REACHED: **False**

## Precheck

See the implementation audit for sampler equivalence, snapshot replay, cross-reset restore, BC-BC pairing, normalization, action semantics and gate self-test.

## Pulse calibration

- Result: **PASS**
- Multiplier: `1.0`; maximum clip ratio: `0.0`
- Maximum EEF displacement versus sham: `0.01911856716675236`; object displacement maximum: `0.0`

## Coverage and estimates

{
  "curves": {
    "0.0": {
      "bc": 0.358695652173913,
      "ft": 0.7119565217391305,
      "L": -0.35326086956521746,
      "bc_ci": {
        "lower": 0.20652173913043478,
        "upper": 0.5163043478260869,
        "clusters": 23,
        "n_boot": 5000,
        "seed": 950001
      },
      "ft_ci": {
        "lower": 0.5706521739130435,
        "upper": 0.8478260869565217,
        "clusters": 23,
        "n_boot": 5000,
        "seed": 950001
      }
    },
    "0.125": {
      "bc": 0.43478260869565216,
      "ft": 0.6684782608695652,
      "L": -0.23369565217391303,
      "bc_ci": {
        "lower": 0.29347826086956524,
        "upper": 0.5760869565217391,
        "clusters": 23,
        "n_boot": 5000,
        "seed": 950001
      },
      "ft_ci": {
        "lower": 0.5271739130434783,
        "upper": 0.7936141304347806,
        "clusters": 23,
        "n_boot": 5000,
        "seed": 950001
      }
    },
    "0.25": {
      "bc": 0.358695652173913,
      "ft": 0.6739130434782609,
      "L": -0.31521739130434784,
      "bc_ci": {
        "lower": 0.24456521739130435,
        "upper": 0.483695652173913,
        "clusters": 23,
        "n_boot": 5000,
        "seed": 950001
      },
      "ft_ci": {
        "lower": 0.5380434782608695,
        "upper": 0.7989130434782609,
        "clusters": 23,
        "n_boot": 5000,
        "seed": 950001
      }
    },
    "0.5": {
      "bc": 0.29891304347826086,
      "ft": 0.6086956521739131,
      "L": -0.3097826086956522,
      "bc_ci": {
        "lower": 0.19021739130434784,
        "upper": 0.41847826086956524,
        "clusters": 23,
        "n_boot": 5000,
        "seed": 950001
      },
      "ft_ci": {
        "lower": 0.47282608695652173,
        "upper": 0.7445652173913043,
        "clusters": 23,
        "n_boot": 5000,
        "seed": 950001
      }
    }
  },
  "L0": -0.3532608695652174,
  "Lbar": -0.28623188405797106,
  "Ibar": 0.06702898550724638,
  "Lbar_ci": {
    "lower": -0.3949275362318841,
    "upper": -0.18478260869565222,
    "clusters": 23,
    "n_boot": 5000,
    "seed": 950001
  },
  "Ibar_ci": {
    "lower": -0.05797101449275362,
    "upper": 0.1920742753623182,
    "clusters": 23,
    "n_boot": 5000,
    "seed": 950001
  },
  "block_A": {
    "Lbar": -0.3260869565217391,
    "Ibar": 0.02717391304347827
  },
  "block_B": {
    "Lbar": -0.24637681159420288,
    "Ibar": 0.10688405797101448
  },
  "main_paired_clusters": 23,
  "selected_anchors": 37,
  "parent_clusters": 24,
  "bc_parent_success": 0.4583333333333333,
  "ft_parent_success": 1.0,
  "Jnom": 0.5416666666666666,
  "bc_sham_success": 0.358695652173913,
  "bc_nonzero_success": 0.3641304347826087,
  "planned_reset_seeds": 24,
  "completed_reset_seeds": 24,
  "valid_parent_clusters": 24,
  "excluded_pulse_success_clusters": 3,
  "partial_clusters": 0,
  "context": false
}

## Decision answers

1. The previous S0d-R2 evaluation-sampling issue is corrected for this run: the official v0.6 DDPM sampler is invoked with `deterministic=True`, while retaining its prescribed DDPM noise schedule.
2. The physical pulse formed a measurable boundary: calibration passed with multiplier `1.0`, maximum clip ratio `0.0`, and maximum EEF displacement `0.01911856716675236` versus sham.
3. BC recovery was `0.358695652173913` at sham and `0.29891304347826086` at magnitude 0.5; FT was `0.7119565217391305` and `0.6086956521739131`, respectively. Across nonzero pulses `Lbar=-0.28623188405797106`.
4. No >=10 percentage-point recovery-retention tradeoff was established: `Lbar=-0.28623188405797106`, `Ibar=0.06702898550724638`, and the context gate failed because BC sham success was `0.358695652173913` (<0.50).
5. This result is not worth entering mechanism research: the registered phenomenon gate is INCONCLUSIVE/NO_GO, with FT outperforming BC on the measured local recovery curves rather than showing a retention loss.
6. Transport, mechanism causality, online algorithm feasibility, algorithm effectiveness, checkpoint sweep and formal training remain completely untested.

Transport: **NOT_RUN by protocol**.

The 24-seed boundary result is a bounded local screening. It cannot be generalized to all reachable states, all checkpoints, or a paper-level mechanism claim.