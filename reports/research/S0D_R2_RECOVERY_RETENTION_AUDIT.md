# S0d-R2 recovery-retention audit

This report is a bounded native-simulator screening, not training or a mechanism intervention.

## Final fields

- TASK: **S0d-R2**
- PRIMARY_DECISION: **B**
- STATUS: **INCONCLUSIVE**
- CURRENT_GRADIENT_BUDGET: **STOP**
- CURRENT_R1: **STOP**
- IMPLEMENTATION_AUDIT: **INCOMPLETE**
- MEASUREMENT_VALIDITY: **INCOMPLETE**
- PHENOMENON_GATE: **INCONCLUSIVE**
- MECHANISM_GATE: **NOT_RUN**
- ONLINE_ALGORITHM_FEASIBILITY: **NOT_TESTED**
- ALGORITHM_EFFECTIVENESS: **NOT_TESTED**
- NOVELTY_STATUS: **CANDIDATE_ONLY**
- NEXT_STAGE: **NO_GO**
- NEW_ENVIRONMENT_ATOMIC_STEPS: **60144**
- NEW_ENVIRONMENT_CHUNK_STEPS: **15020**
- FORMAL_TRAINING_ITERATIONS: **0**
- OPTIMIZER_STEPS: **0**
- WALL_CLOCK_SECONDS: **873.139**
- TIME_LIMIT_REACHED: **False**

## Square

- Gate: **PHENOMENON_INCONCLUSIVE**
- Coverage: cases=4, clusters=3, BC-source=1, FT-source=3, pass=False.
- L=None; L0=None; I=None; Jnom=0.9375.
- L 95% CI={}; I 95% CI={}.
- L_A=None; L_B=None; BC perturbed=None; BC sham=None; FT sham=None.
- Implementation precheck: {'passed': True, 'pilot': {'910001': {'passed': True, 'obs_max_abs': 0.0, 'state_max_abs': 0.0, 'done_success_equal': True, 'snapshot_fields': ['action_queue', 'controller_state', 'model', 'numpy_rng', 'obs_cache', 'obs_history', 'python_rng', 'raw_attrs', 'states', 'torch_cuda_rng', 'torch_rng', 'wrapper_chunk_steps', 'wrapper_time'], 'tape_shape': [8, 7]}, '910002': {'passed': True, 'obs_max_abs': 0.0, 'state_max_abs': 0.0, 'done_success_equal': True, 'snapshot_fields': ['action_queue', 'controller_state', 'model', 'numpy_rng', 'obs_cache', 'obs_history', 'python_rng', 'raw_attrs', 'states', 'torch_cuda_rng', 'torch_rng', 'wrapper_chunk_steps', 'wrapper_time'], 'tape_shape': [8, 7]}}}.
- Invalid/missing reasons from all Square records: `anchor_invalid_success_or_done`=7, `episode_horizon`=148, `native_success`=48, `success_at_pulse`=4, `success_or_done_during_pulse`=1, `pulse_complete`=49, and parent `horizon_or_natural_termination`=32. Screening records are retained in the CSV; they are not silently discarded.

## Transport

- Gate: **NOT_RUN**
- Reason: Square did not achieve PHENOMENON_PASS; Transport main screening prohibited..

## Implementation limitation

The real snapshot, native-success, sampler and synthetic-gate checks passed, but the runner did not execute a BC-BC paired-effect precheck and did not write dedicated pulse clip-ratio, EEF-displacement or object-displacement fields. Therefore `IMPLEMENTATION_AUDIT=INCOMPLETE`; no scientific result is promoted to PASS and no repair evaluation was added after seeing the coverage result.

## Scope interpretation

The empirical set C is a noisy BC screening set, not a proof of q_BC>=0.75. Any PASS would only request upper review for controlled mechanism design; this run does not authorize training, N1/N2, or other directions.
