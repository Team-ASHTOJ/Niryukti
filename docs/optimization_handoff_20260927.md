# Optimization handoff — 27 September 2026

This is a checkpoint, not completion of the multi-year solver roadmap. Work stopped at the user's requested five-minute deadline. Preserve the current NIRYUKTI frontend; the binary and C++ namespace remain `vantage`.

## Implemented in this checkpoint

- CUDA versions of the existing Halpern, restarted Halpern and reflected restarted Halpern LP operators. The CPU/CUDA acceptance mismatch described in the supplied research text was already fixed upstream.
- Opt-in CUDA Graph iteration execution on a dedicated nonblocking stream, checked API calls and owned graph resources.
- Automatic 32-bit CSR selection when dimensions/nonzeros fit; explicit `--gpu-indices 32|64` overrides.
- Experimental FP32 matrix storage with FP64 vectors/compute via `--matrix-precision mixed`. Reject overflowing/underflowing conversions. The original FP64 model remains authoritative for final verification. Fixed Halpern and sparse QP require FP64 matrices.
- Sparse off-diagonal convex QP representation, JSON/MPS input/output, CPU and CUDA `Q*x`, projected-gradient primal updates, original-space objective/stationarity and convex affine-minorant lower bounds.
- CLI/Python controls and GPU execution metadata. Benchmark runner accepts the new GPU switches.

### Sparse QP mathematics and limits

This implementation uses smooth primal-dual splitting, **not a claimed reproduction of rAPDHG/PDQP**. A conservative common step reduction ensures `tau*sigma*||A||² + tau*L_Q/2 < 1`, using absolute row/column bounds for A and a row-sum bound for Q. Existing diagonal QP proximal updates remain intact. General sparse QP disables adaptive backtracking internally.

JSON `objective.quadratic_sparse` is an array of `[row_index, column_index, coefficient]` triples for a **full symmetric matrix**, added to the optional diagonal shorthand. Objective is `0.5*x^T*Q*x + c^T*x`. QMATRIX requires full symmetric off-diagonal entries; QUADOBJ mirrors its triangular entries. Do not provide both triangles in QUADOBJ.

Symmetry is checked exactly. Symmetric diagonally dominant matrices with nonnegative diagonals pass a sufficient PSD check. Other matrices up to 256 variables receive a numerical long-double LDL-style check; larger non-dominant matrices are explicitly rejected as unsupported. This is not universal PSD validation. Dense storage is used only for that bounded validation step, never optimization iterations.

Full-Q models currently bypass presolve/scaling to preserve cross terms correctly. Their lower bounds are numerical diagnostics, not interval-certified MILP pruning bounds. MIQP remains unsupported.

Research reference: [Condat's original smooth primal-dual splitting paper](https://lcondat.github.io/publis/Condat-optim-JOTA-2013.pdf). Algorithms were independently implemented; no existing optimizer is linked into production.

## Validation

CUDA and CPU builds passed. CUDA research tests currently pass 2,613 assertions, including 120 CPU/CUDA anchored recurrence checkpoints, reset parity, graph/non-graph variants, 32/64-bit indices, mixed precision on an exactly representable matrix, and analytical sparse QP solutions on both backends. Existing core, CLI/Python, performance-profile and dependency-policy tests passed. Detailed run logs and measured binaries are in `results/optimization-20260927/`.

This checkpoint does **not** claim fresh sanitizer coverage, broad mixed-precision robustness or industrial-QP competitiveness. Full vectors still return to the host at KKT checkpoints. GPU utilization is real, but further transfer/monitoring improvements are needed.

## Pending work, in recommended order

1. Review this checkpoint, rerun sanitizer builds and expand sparse-QP tests: singular PSD matrices, indefinite rejection, fixed variables, permutation/scale metamorphic tests, large sparse models and parser variants. Reject ambiguous QUADOBJ double-triangle input explicitly.
2. Expand CPU/CUDA differential tests for complete restarted solves, certificates and PID/polishing; exercise mixed matrices with nonexact coefficients and ill-conditioned models. Final FP64 verification must remain mandatory.
3. Run a proper frozen baseline campaign with warmup plus three measured repetitions: current default, graph mode, reflected Halpern, mixed precision, combined scaling, PID and polishing. Retain timeouts and regressions. Use matching tolerances, thread limits and hardware. The short end-of-session run is preliminary.
4. Implement GPU scalar KKT/residual monitoring to remove routine full-vector D2H transfers; keep independent final CPU verification.
5. Add persistent GPU contexts/workspaces and benchmark graph overhead, including graph recapture after primal-weight changes. Investigate SpMVOp only after correctness and profiling.
6. Implement reversible cross-term QP presolve/scaling; improve scalable PSD validation; evaluate a faithful rAPDHG/PDHCG algorithm against the current smooth splitting path on public convex QP datasets.
7. Add validated research options to the dashboard and surface measured results/limitations. Current frontend design is preserved; new switches are available via CLI/Python.
8. Expand Netlib/Mittelmann/MIPLIB relaxations/QPLIB benchmark coverage with checksums, raw results and honest performance profiles.
9. Replace full per-node vectors with sparse bound deltas; add batched GPU strong branching and OBBT. Existing reliability branching is serial.
10. Extend cutting planes beyond restricted binary cover/clique cuts: valid representation-based MIR/implied-bound cuts, then propagation reasons and conflict analysis. GMI/tableau cuts need an appropriate basis engine.
11. General improving recession-ray extraction and independent unboundedness certificate verification.
12. GPU/hybrid presolve and scheduling based on actual measured costs.
13. Larger projects: sparse barrier/IPM, revised dual simplex, HPR-LP research implementation, native Python/C ABI, HIP/ROCm, concurrent portfolios and multi-GPU. These are absent and must not be presented as completed.

## Tomorrow's starting commands

```bash
cmake --build build -j5
./scripts/run_tests.sh build
cmake --build build-cpu -j5
./scripts/run_tests.sh build-cpu
.venv/bin/python benchmark/run.py datasets/afiro.mps datasets/adlittle.mps datasets/israel.mps datasets/e226.mps examples/refinery.json --solvers cpu,cuda,highs --runs 3 --time-limit 10 --cuda-graphs --output results/optimization-next
```

First read this record, `docs/optimization_audit.md`, `docs/research_features.md` and the new measured-results record before proposing further changes. Keep every experimental improvement opt-in until measurements justify changing defaults.

## Stop status

The user requested an immediate stop. The short graph/HiGHS benchmark was interrupted before completion; its raw per-run files remain in `results/optimization-20260927/graphs/raw/`. Do not quote a complete-suite speedup from that partial run. The initial baseline run retained HiGHS import failures from system Python; use `.venv/bin/python` (HiGHS 1.15.1) for the resumed campaign. No new sanitizer campaign or dashboard-option integration was completed tonight.
