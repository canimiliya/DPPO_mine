# S0d-R2 implementation audit

## Result

**INCOMPLETE**. The real preflight completed for both tasks, but this bounded runner does not satisfy every registered implementation-precheck requirement, so the scientific output is not promoted to a valid phenomenon result.

## Real checks completed

- Checkpoint SHA256 and byte-size checks passed for Square and Transport `state_0` and `state_200`; state_0 base actors matched state_200 base actors exactly (Square 16 keys, max absolute difference 0; Transport 12 keys, max absolute difference 0).
- The actual state_0 actor is loaded from the audited BC EMA-equivalent model state; no non-EMA BC replacement was used.
- The existing legacy MuJoCo 2.1 / mujoco-py runtime was used: Python 3.12.3, torch 2.7.1+cu128, CUDA 12.8, RTX 5060 Ti, robomimic 0.5.0, robosuite 1.4.1, mujoco-py 2.1.2.14, and Gym 0.22.0. No dependency was installed or changed.
- Native robosuite `_check_success()` was called on every atomic simulator step. The Transport source and semantics remain the audited `TwoArmTransport._check_success`: payload in target bin AND trash in trash bin.
- The snapshot contains simulator model/state, controller arrays/caches, raw environment time/done fields, observation history, action queue, and Python/NumPy/Torch RNG state.
- For both pilot seeds 910001 and 910002 on both tasks, same-snapshot fixed-tape replay passed: observation max error 0, simulator-state max error 0, and done/success flags identical over 8 steps per replay.
- Actual sampler-vs-fixed-tape replay max action and chain error was 0 for BC and FT on both tasks.
- The gate self-test was called by the runner and passed PASS, FAIL, threshold-crossing INCONCLUSIVE, coverage, one-task-only, context/sham failure and implementation-failure cases.

## Missing required checks / fields

- A real BC-BC paired pipeline effect check was not executed. The synthetic gate test is not a substitute.
- Pulse records contain the actual action delta maximum, but not dedicated clip-ratio, EEF displacement and object displacement fields.
- The implementation audit is therefore **INCOMPLETE**, not PASS. This is a measurement/implementation limitation, not evidence that the recovery phenomenon is absent.

The above omissions are preserved in `S0D_R2_MANIFEST.json`; no scientific sample was added to repair them after viewing the result.
