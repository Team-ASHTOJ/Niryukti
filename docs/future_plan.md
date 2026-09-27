# VANTAGE future development plan

Start from the [teammate handoff dated 27 September 2026](teammate_handoff_20260927.md). This is a gated roadmap, not a promise that every algorithm will be implemented before submission. Each milestone needs working code, independent correctness tests and reproducible measurements before its claims enter the presentation.

Current implementation and validation gaps are tracked in the [27 September roadmap audit](roadmap_audit_20260927.md). No milestone exit gate is yet declared complete.

## Milestone A — Submission-ready checkpoint

Owner: integration and validation teammate.

- Validate the merged dashboard Arena, cancellation, report downloads and model imports.
- Run CPU/CUDA tests, sanitizer checks, Docker smoke tests and the complete reproduction script.
- Freeze the revision and run three repetitions on a declared public subset with matching budgets. Retain every failure and raw output.
- Publish a supported-feature matrix, per-instance CSV/HTML, certificate examples, synthetic refinery demonstration and limitations.

Exit gate: another laptop can reproduce the demo offline; displayed results identify their revision, hardware and verification status. No unsupported algorithm or unmeasured speedup appears in submission material.

## Milestone B — Reliable continuous optimization

Owner: continuous solver teammate.

- Harden general sparse QP step control, convexity validation and postsolve. Expand analytical, randomized and transformation tests.
- Improve simplex ratio tests, refinement, degeneracy handling, time-limit behavior and numerical-failure recovery.
- Add basis warm starts and evaluate revised dual simplex as a separate substantial project.
- Strengthen general infeasibility/unboundedness ray extraction and independent verification.
- Compare current PDHG/Halpern/polishing configurations before promoting experimental defaults. Investigate rAPDHG or correctly constrained inexact proximal solves for QP from primary mathematical references.

Exit gate: better verified solve rate on a frozen public corpus without regressions in existing correctness tests; full sparse QP limitations remain explicit.

## Milestone C — Measurable GPU advantage

Owner: GPU teammate with supported hardware.

- Profile transfers, SpMV, reductions and launch overhead first.
- Keep persistent device matrices/workspaces across repeated solves; move restart/control and diagnostic work to the device where justified.
- Measure graphs, 32-bit indices and mixed precision independently with original FP64 final verification.
- Prototype batched strong branching/OBBT and lightweight GPU presolve only after stable single-relaxation execution.

Exit gate: matched-algorithm CPU/GPU crossover plots and end-to-end timings, including transfers and unsuccessful instances. Device-wide utilization must remain distinct from solver-attributed profiling.

## Milestone D — Stronger discrete optimization

Owner: MILP teammate.

- Expand reversible propagation and presolve with transformation-level tests.
- Add a controlled dynamic cut pool, supported MIR/implied-bound families and conflict explanations. Tableau cuts require a valid basis representation first.
- Benchmark reliability branching, pump, RINS and local branching independently; use valid dual bounds for pruning throughout.
- Implement exact search checkpoint/resume with fingerprints, queue state, bounds, incumbent and random state.
- Extend MIQP tests using brute-force small-domain oracles and independent incumbent checks.

Exit gate: repeated MIPLIB campaign reports verified incumbents, valid gaps, solve rates and node counts. Standalone incumbent verification must not be presented as a replayed tree proof.

## Milestone E — Platform and industrial evidence

Owner: platform/documentation teammate.

- Add live global MIP-gap history and measured memory telemetry, clear failure explanations and structural model analysis.
- Expand refinery, production, energy, scheduling and supply-chain cases with documented units, assumptions and provenance.
- Archive benchmark manifests, checksums and source/binary versions; generate presentation tables from those artifacts rather than copied numbers.
- Improve parsers, fuzzing, installation and portable numerical backends.

Exit gate: an evaluator can trace each dashboard claim to a model, command, raw result and verifier outcome.

## Later research tracks

Barrier/IPM, revised dual simplex at scale, concurrent portfolios, HIP/ROCm and multi-GPU are separate major engineering efforts. NLP/MINLP should begin only with a correctly scoped formulation and verification design. Do not implement placeholder flags or market planned algorithms as available.

## Working agreement

Use small reviewable commits on main or short task branches; fetch before integrating teammates' changes. Each task should record: problem addressed, mathematical derivation/reference, changed files, tests, measured before/after result and remaining limits. Re-run affected validation after merges. Prioritize verified solve rate and trustworthy bounds before raw speed, and raw speed before cosmetic feature count.
