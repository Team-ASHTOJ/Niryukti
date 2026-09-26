# Presentation and submission evidence

## Project identity and problem

**NIRYUKTI — Independent Sparse Optimization Engine.** A research prototype of an independent sparse mathematical optimization engine for SIH26119, sponsored by MRPL.

Industrial planning uses optimization to choose feasible, economical decisions under resource and quality constraints. The official challenge asks for an inspectable solver core, with GPU acceleration where it provides measurable benefit.

## What we can demonstrate

1. MPS, a documented LP subset, and native JSON enter a common sparse model.
2. An independently implemented primal-dual method solves LP and diagonal convex QP models on CPU and CUDA.
3. An independent verifier checks solutions in original model units.
4. Branch-and-bound calls our own LP engine and uses conservative dual bounds for pruning.
5. Synthetic refinery blending, scheduling, supply-chain and power-dispatch examples exercise the same core.
6. A local dashboard starts actual solves, displays saved numerical results, and compares them with external HiGHS benchmark runs.

The core does not call HiGHS, SCIP, Gurobi or another optimization solver. HiGHS is an external benchmark baseline only. Ordinary CUDA/cuSPARSE/runtime infrastructure is used for computation.

## Suggested slide sequence

1. Industrial problem and official scope.
2. Sparse model → presolve/scaling → CPU/CUDA iterations → independent verification.
3. LP/diagonal QP/MILP support, with explicit current limits.
4. Refinery demo: model, constraints, computed decision and objective.
5. Per-instance benchmark table, including failures and limits.
6. CPU/GPU comparison with hardware, tolerances, timings and repetitions stated.
7. Phase 2 engineering changes, before/after evidence, unresolved research.
8. Roadmap toward wider QP support, stronger numerical methods and larger industrial validation.

## Evidence rules

- Use the current reproducible benchmark report for numbers; do not copy preliminary log timings into performance claims.
- Cite the exact run directory and manifest. Show end-to-end time alongside iteration time, and distinguish own-CPU speedup from a HiGHS comparison.
- An `OPTIMAL` result is a numerical tolerance result, not a formal exact-arithmetic proof.
- For MILP report incumbent, global bound and MIP gap. Continuous KKT diagnostics of a rounded integer solution are not an MILP optimality certificate.
- Generated refinery data are synthetic; they are not MRPL operational data.
- A successful selected benchmark does not establish production readiness, million-variable scalability, or superiority over commercial solvers.
- General sparse QP, MIQP, NLP, MINLP, advanced cuts and a full industrial branch-and-cut implementation are not currently supported.

## Current evidence locations

- Historical Phase 1 results: [validation.md](validation.md).
- Work in progress and exploratory Phase 2 observations: [development_log.md](development_log.md).
- Reproducible Phase 2 results: [phase2_results.md](phase2_results.md), generated from `results/phase2-baseline` and `results/phase2-final`. NIRYUKTI solves 8/9 selected cases on both CPU and CUDA, up from 6/9; HiGHS solves 9/9. The remaining e226 failure and ADLITTLE GPU regression are retained.

The separate 200,000-iteration e226 campaign succeeds on both NIRYUKTI backends in all three repetitions. Present it as a budget-sensitivity example, alongside its standard-budget limit, rather than merging different budgets into a single solve-rate claim.
