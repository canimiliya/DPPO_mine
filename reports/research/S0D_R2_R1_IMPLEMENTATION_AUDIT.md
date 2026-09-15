# S0D_R2_R1_BOUNDARY implementation audit

- STATUS: **INCONCLUSIVE**
- IMPLEMENTATION: **PASS**
- No optimizer, training iteration, Transport, or mechanism intervention was run.

## Precheck evidence

- `action_coordinate`: `{"control_dim": 6, "controller": "OperationalSpaceController", "indices": {"axis_angle_x": 3, "axis_angle_y": 4, "axis_angle_z": 5, "x": 0, "y": 1, "z": 2}, "passed": true, "reference_frame": "relative controller command", "use_delta": true}`
- `bc_bc_paired`: `{"action_max_abs": 0.0, "done_success_equal": true, "obs_max_abs": 0.0, "passed": true, "state_max_abs": 0.0}`
- `gate_self_test`: `{"coverage_inconclusive": {"expected": "PHENOMENON_INCONCLUSIVE", "observed": "PHENOMENON_INCONCLUSIVE", "pass": true}, "fail": {"expected": "PHENOMENON_FAIL", "observed": "PHENOMENON_FAIL", "pass": true}, "implementation_fail": {"expected": "IMPLEMENTATION_FAIL", "observed": "IMPLEMENTATION_FAIL", "pass": true}, "one_task_not_joint": {"expected": "PHENOMENON_INCONCLUSIVE", "observed": "PHENOMENON_INCONCLUSIVE", "pass": true}, "pass": {"expected": "PHENOMENON_PASS", "observed": "PHENOMENON_PASS", "pass": true}, "sham_context_fail": {"expected": "PHENOMENON_INCONCLUSIVE", "observed": "PHENOMENON_INCONCLUSIVE", "pass": true}, "threshold_inconclusive": {"expected": "PHENOMENON_INCONCLUSIVE", "observed": "PHENOMENON_INCONCLUSIVE", "pass": true}}`
- `normalization`: `{"obs_max_abs": 0.0, "action_inverse_max_abs": 0.0, "passed": true, "atomic_steps": 0}`
- `sampler_equivalence`: `{"bc": {"action_max_abs": 0.0, "chain_max_abs": 0.0, "passed": true}, "ft": {"action_max_abs": 0.0, "chain_max_abs": 0.0, "passed": true}}`
- `sampling_calibration`: `{"records": [{"deterministic": true, "native_success": false, "policy": "bc", "seed": 930001}, {"deterministic": false, "native_success": false, "policy": "bc", "seed": 930001}, {"deterministic": true, "native_success": true, "policy": "ft", "seed": 930001}, {"deterministic": false, "native_success": false, "policy": "ft", "seed": 930001}, {"deterministic": true, "native_success": false, "policy": "bc", "seed": 930002}, {"deterministic": false, "native_success": false, "policy": "bc", "seed": 930002}, {"deterministic": true, "native_success": true, "policy": "ft", "seed": 930002}, {"deterministic": false, "native_success": true, "policy": "ft", "seed": 930002}, {"deterministic": true, "native_success": false, "policy": "bc", "seed": 930003}, {"deterministic": false, "native_success": false, "policy": "bc", "seed": 930003}, {"deterministic": true, "native_success": true, "policy": "ft", "seed": 930003}, {"deterministic": false, "native_success": true, "policy": "ft", "seed": 930003}, {"deterministic": true, "native_success": false, "policy": "bc", "seed": 930004}, {"deterministic": false, "native_success": true, "policy": "bc", "seed": 930004}, {"deterministic": true, "native_success": true, "policy": "ft", "seed": 930004}, {"deterministic": false, "native_success": true, "policy": "ft", "seed": 930004}], "sensitivity": [{"action_max_abs": 1.999997079372406, "deterministic_success": false, "non_deterministic_success": false, "policy": "bc", "seed": 930001}, {"action_max_abs": 1.999997079372406, "deterministic_success": true, "non_deterministic_success": false, "policy": "ft", "seed": 930001}, {"action_max_abs": 1.999997079372406, "deterministic_success": false, "non_deterministic_success": false, "policy": "bc", "seed": 930002}, {"action_max_abs": 1.9535856246948242, "deterministic_success": true, "non_deterministic_success": true, "policy": "ft", "seed": 930002}, {"action_max_abs": 1.999997079372406, "deterministic_success": false, "non_deterministic_success": false, "policy": "bc", "seed": 930003}, {"action_max_abs": 1.0905770063400269, "deterministic_success": true, "non_deterministic_success": true, "policy": "ft", "seed": 930003}, {"action_max_abs": 1.999997079372406, "deterministic_success": false, "non_deterministic_success": true, "policy": "bc", "seed": 930004}, {"action_max_abs": 1.999997079372406, "deterministic_success": true, "non_deterministic_success": true, "policy": "ft", "seed": 930004}]}`
- `snapshot`: `{"cross_reset": {"done_success_equal": true, "obs_max_abs": 0.0, "state_max_abs": 0.0}, "native_success_callable": true, "passed": true, "same_snapshot": {"done_success_equal": true, "obs_max_abs": 0.0, "state_max_abs": 0.0}, "snapshot_fields": ["action_queue", "controller_state", "model", "numpy_rng", "obs_cache", "obs_history", "python_rng", "raw_attrs", "states", "torch_cuda_rng", "torch_rng", "wrapper_chunk_steps", "wrapper_time"], "tape_shape": [8, 7]}`

## Pulse calibration

- Result: **PASS**
- Multiplier: `1.0`
- Maximum clip ratio: `0.0`
- Maximum EEF displacement: `0.01911856716675236`
- EEF threshold: `0.0`

## Interpretation

The prior S0d-R2 sampling error is corrected here by using the official v0.6 DDPM evaluation sampler with `deterministic=True` while retaining its prescribed DDPM noise schedule. This boundary diagnostic measures native simulator recovery only; it does not identify a diffusion-specific mechanism.