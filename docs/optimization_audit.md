# Optimization audit — 2026-09-27

Baseline: frontend commit 47bbd5d, incorporating research commit 50dac0c. The supplied research text repeats an older assessment; actual code is authoritative. The core remains independent of external optimization engines. NIRYUKTI is the current display brand; the C++ namespace/binary remain vantage.

| Requested component | Audit result before this work |
| --- | --- |
| CPU/CUDA adaptive acceptance | Already unified at limit >= 1; direct displacement products implemented |
| Halpern / restarted / reflected LP | CPU implemented and tested; CUDA unavailable in friend's validation |
| PID / power estimate / polishing | Implemented, opt-in; GPU execution needs hardware validation |
| Combined scaling | Implemented; known per-instance regressions retained |
| Reliability branching / serial strong probes | Implemented; batched GPU probes absent |
| Binary cover/clique cuts | Implemented for eligible exact-integer binary knapsacks; MIR/GMI/conflicts absent |
| Pump / RINS | Implemented with explicit budgets; no industrial competitiveness established |
| Additional Farkas candidates | Implemented; general primal recession certificates absent |
| Performance profiles | Implemented; broad benchmark campaigns absent |
| CUDA Graphs / narrow indices / mixed matrix precision | Absent |
| GPU residual monitoring | Absent; all checkpoint vectors copied to host |
| General sparse convex QP | Absent; diagonal Q only |
| Barrier / revised simplex / HPR-LP / HIP / multi-GPU | Absent; must not be represented as completed |
| Native bindings / portfolio / GPU presolve / bound-delta nodes | Absent |

## Current implementation checkpoints

1. Preserve the baseline executable, validate CUDA research methods, add CUDA Graph/narrow-index/mixed-matrix controls with FP64 final verification.
2. Extend sparse QP representation, input, scaling, objective gradient and independent verification together; reject unvalidated/nonconvex cases explicitly.
3. Add meaningful CPU/CUDA differential, analytical and metamorphic tests, then rerun measured configurations sequentially.
4. Expose validated options and evidence through the dashboard and document actual successes, regressions, supported combinations and remaining roadmap.

Numerical performance from other papers is motivation, not a result for this code. Advanced settings remain opt-in until broad results justify defaults. GPU and general-QP algorithms require independent original-space checks before any optimal status.
