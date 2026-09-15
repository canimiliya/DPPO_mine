# S0c-R1 upper-review archive

This note records the upper-review decision dated 2026-09-15. It does not modify the historical S0c-R1 reports or recompute their GPU experiment.

- Primary decision: **B — STOP R1; authorize one bounded S0d-R2 local recovery-retention screening**.
- R1 scientific status remains **INCONCLUSIVE** and R1 receives no additional samples, bandwidth repair, G2, BDUA or gradient-budget work.
- The R1 main baseline was `direct_mu_l2_normalized`, not Gaussian KL. The two task TEST rho values were candidate-set averages and the pooled rho values answered a different question.
- The S0c-R1 implementation had useful core corrections, but the actual bandwidth calculation reshaped `[32,64,4,7]` to `[32,1792]` rather than the registered full chunk `[2048,28]`; `same_target` could bypass TEST budget matching; pairwise A/B, energy, ratio CI and noise-floor aggregation were not a complete faithful implementation of the registered G1 gate.
- Accordingly, the R1 implementation was not unconditionally endorsed as a complete protocol audit. Its scientific result remains informative but inconclusive.
- The decision is a value-of-information screen, not a claim that execution geometry is disproven. No R1 result supports an algorithm or paper claim.

S0d-R2 is restricted to fixed BC/FT assets, finite simulator evaluation and native recovery success. It is not a mechanism intervention, algorithm-effectiveness experiment or training authorization. Even a S0d-R2 PASS only returns to upper review.
