# Changelog

## 0.2.0 — 2026-09-11

- CPU/CUDA adaptive PDHG trial-step control, step-weighted averaging and growing restart epochs.
- Reusable independent-verification workspace and stricter row-endpoint normalization.
- Conservative integer-bound propagation before MILP node relaxations.
- Independently replayable box/row Farkas infeasibility certificates.
- Accepted/rejected iteration and integer-bound-tightening telemetry.
- Local dashboard with actual solve execution, uploads, iteration/adaptive controls, result history and evidence-driven benchmark views.
- Before/after comparison generator, preserved raw campaigns and explicit extended-budget experiments.
- Expanded correctness suite: 55 enumerated MILPs, cached-verifier checks and certificate tampering tests; CPU/CUDA, sanitizer, CLI and browser validation.
- Documentation index, engineering log, operations guide, algorithm details and reusable presentation/submission evidence.

The standard selected suite improves from 6/9 to 8/9 optimal cases on both backends. e226 still reaches the standard iteration cap; a separately documented higher budget resolves it. ADLITTLE GPU performance regresses. General sparse QP and advanced industrial optimization remain future work. See [measured evidence](docs/phase2_results.md), not this changelog, for timings and limits.

## 0.1.0 — Initial prototype

Independent sparse LP/diagonal QP engine, basic MILP search, CPU/CUDA backends, parsers, presolve/scaling, independent verification, CLI/Python interface, synthetic industrial examples and external HiGHS benchmark adapter. [Historical validation](docs/validation.md).
