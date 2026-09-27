# Historical Phase 2 roadmap

This document records the earlier 0.2 planning baseline. Several items below are now implemented. Consult [the current completion record](solver_completion_20260927.md), [large-model evidence](stress_campaign_20260927.md) and the updated architecture/algorithms documents for current status.

The official SIH problem statement is authoritative. The long AI-generated brief supplied with the project is guidance; its complete feature list is not a claim about this prototype.

## Completed in the 0.2 implementation

Cached independent verification; violated-endpoint normalization; adaptive trial steps on CPU/CUDA; step-weighted averages and growing restart epochs; conservative integer-node propagation; replayable box/row Farkas certificates; expanded enumeration and certificate tests. See [development log](development_log.md) for validation status and [algorithms](algorithms.md) for exact behavior. General QP and advanced optimizer work below remain future work.

## Numerical work

1. Broaden regression problems before tuning. Add ill-conditioned, degenerate, nearly dependent and infeasible models, full Netlib selections and independent primal/dual audits. Preserve all unsuccessful cases.
2. Extend the now-cached independent verifier with GPU monitoring reductions and stronger numerical audits.
3. Evaluate the new adaptive trial-step method across broader suites; add diagonal step preconditioning and stronger restart merit functions.
4. Add reversible singleton processing, bound propagation, safe duplicate/redundant-row detection and comprehensive dual postsolve. Current presolve deliberately stays small.
5. Broaden certificate discovery beyond the implemented normalized-multiplier box/row Farkas test; add general recession directions and presolve/tree proof artifacts.
6. Implement general convex sparse QP, PSD checks, and an appropriate proximal/CG or forward-gradient method. Current QP support is diagonal only.

## GPU and scale

- Move monitoring reductions onto GPU while retaining final independent verification.
- Cache GPU contexts/descriptors and sparse structures across repeated solves and MILP nodes.
- Add reliable device peak-memory and event profiling, richer memory planning and allocation failure telemetry.
- Explore larger checkpoint chunks, launch fusion and CUDA graphs only after equivalent numerical behavior is verified.
- Establish CPU/GPU crossover on increasingly large sparse instances. Do not infer industrial scale or multi-GPU support from repeated synthetic blocks.

## MILP

- Replace per-node dense bound/warm-start vectors with persistent bound changes and resource budgets.
- Improve difficult LP relaxations and resolve stalled leaves without weakening bound safety.
- Add safe integer bound propagation, pseudo-cost branching, diving, cut interfaces and selected valid cover/clique cuts.
- Add independently replayable node-bound/presolve proof artifacts and stronger feasibility polishing.
- Track first incumbent, complete node-event telemetry and checkpoints. Current search retains unresolved bounds and returns a limited/unknown status when necessary.

## Interoperability and benchmark work

- Expand MPS fixed-field dialects and LP continuation syntax with corpus tests; avoid accepting ambiguous quadratic variants.
- Add native in-process Python bindings once API ownership is settled. The current Python API intentionally uses the structured CLI.
- Add QPLIB/MIPLIB download/filtering and external SCIP adapters, full performance profiles, reproducibility caching and condition/health diagnostics.
- Improve synthetic refinery models with inventory coupling, consistent linearized assay properties and literature-backed assumptions. Existing sulfur blending and scheduling examples are illustrative, not validated refinery process models.

## Current shortcuts to preserve in presentations

No full PDLP implementation, general PDHCG, cuts, simplex, IPM, nonlinear optimizer, mixed precision, multi-GPU support, formal rational certificate system, cloud service or graphical modeling environment is claimed. A local dashboard and offline reports inspect actual solves and measurements; the numerical engine remains the primary contribution.
