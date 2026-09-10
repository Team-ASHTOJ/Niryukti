# Phase 2: prioritized development

The official SIH problem statement is authoritative. The long AI-generated brief supplied with the project is guidance; its complete feature list is not a claim about this prototype.

## Numerical work

1. Broaden regression problems before tuning. Add ill-conditioned, degenerate, nearly dependent and infeasible models, full Netlib selections and independent primal/dual audits. Preserve all unsuccessful cases.
2. Develop a reusable independent-verification workspace and cached sparse transpose. Currently verification repeatedly allocates/transposes and can dominate solve time.
3. Introduce a mathematically justified adaptive step-size/backtracking rule, diagonal step preconditioning and residual balancing. Current step product uses a conservative norm bound and only the primal-dual ratio adapts.
4. Add reversible singleton processing, bound propagation, safe duplicate/redundant-row detection and comprehensive dual postsolve. Current presolve deliberately stays small.
5. Add verified general Farkas/recession certificates. General infeasible/unbounded cases currently often reach limits.
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

No full PDLP implementation, general PDHCG, cuts, simplex, IPM, nonlinear optimizer, mixed precision, multi-GPU support, formal rational certificate system, cloud service or graphical modeling environment is claimed. A polished UI was not part of the official requirement; the offline report exists to inspect actual numerical results.
