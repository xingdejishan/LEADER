# Sparse batched selection trial (2026-10-24)

Experimental opt-in: --selection-mode sparse. Default remains full (e2e_online.py:507).
Implementation: light_selection.py:180 batched shared-map distance queries; :199 retains up to eight original hypotheses without qualified/Top2 filtering, deduplicating only identical poses. All candidate source references are excluded from the common evidence map. Bounded query 1000 points, 15000 points/reference, at most 27 references and 32-entry cloud cache. Frontend remains CUDA; sparse selector uses single-thread CPU KD-tree. No package/environment changes.

## Independent validation
48 new keys disjoint from previously inspected 96, fixed seed 20261024; seven historically visual-adopted frames plus 41 controls. Predictions and parameters fixed before evaluator-only truth read. This selected sample is not an estimate of full-test-set accuracy.
Same candidate pool full replay vs sparse live selection:
- Full mean 1016.42 ms, P95 3520.65 ms.
- Sparse all-frame mean 115.08 ms, P95 228.82 ms, max 282.58 ms, 26/48 exceed 100 ms.
- Sparse 45 active decisions: mean 122.74 ms, P95 230.03 ms; three no-proposal fallbacks.
- Full/sparse translation mean 1.457/1.406 m, P95 8.792/8.792 m; both six over-5m and zero over-10m frames. Two choice differences: one worsens 2.94->6.23 m; one improves 6.09->0.30 m. Not equivalent outputs or proven noninferiority.
- Mean live stage work 1.401 s/frame, not end-to-end latency/throughput measurement.

## Regression, not independent evidence
96 previously inspected frames: sparse selection mean 167.83 ms, P95 331.64 ms; 92 active decisions. Over-5m errors: full 3, prior Top2 light 8, sparse 3. Key 2012-05-26/1338074839327574 error: full 1.743 m, Top2 light 442.571 m, sparse 3.058 m. Translation P95 full 2.026 m vs sparse 2.921 m.

## Decision
Quality substantially recovers relative to Top2 on the regression set, but selection-only P95 remains above 100 ms. Do not replace the default or claim accuracy equivalence. No tuning after either truth evaluation. 33 unit tests pass.

Artifacts: results/sparse_validation_20261024.jsonl; results/sparse_validation_comparison_20261024.json; results/sparse_validation_accuracy_20261024.json; results/sparse_regression_20261024.json; results/sparse_regression_accuracy_20261024.json. Historical comparison script uses light_* field names for the candidate alternative; in sparse_validation_comparison those fields refer to sparse, not the previous Top2 light algorithm.
