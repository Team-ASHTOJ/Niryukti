# Teammate handoff — 27 September 2026

Implementation is stopped at this checkpoint. Branch: `main`. The frontend branch was merged into main and main was pushed before this round. This document distinguishes implemented code from demonstrated readiness; no phase is declared commercially complete.

## Phase coverage

| Phase | Implemented this round / retained | Still pending |
| --- | --- | --- |
| 1 — Mathematics | General sparse convex QP path, convex MIQP relaxations and conservative bounds, revised **primal** simplex, restricted root MIR cuts, node bound deltas, node selection policies; existing Halpern/PID/polishing/reliability/pump/RINS retained | Large non-diagonally-dominant PSD validation, stronger QP algorithms, simplex numerical hardening/warm basis, dynamic cuts/conflicts, advanced reversible presolve |
| 2 — GPU | Opt-in device monitoring, scalar diagnostics, graphs and mixed precision paths, index specialization, memory-aware fallback | Persistent contexts across solves, device restart/control, GPU presolve, batched strong branching/OBBT; fresh controlled speed measurements |
| 3 — Benchmarks | Public downloader/manifests/checksums, Netlib exploratory campaign, repeated-run harness, existing performance profiles | Final repeated Netlib/MIPLIB/QPLIB campaigns, latest adapter validation, CPU/CUDA ablations and memory evidence |
| 4 — Verification | Original-space KKT checks, certificate output, fingerprint/objective/dual corruption rejection, MIQP incumbent checks | Complete MIP tree proof replay, general unboundedness certificates, broader adversarial tests |
| 5 — Inputs | Sparse quadratic JSON/MPS, continuous supported QPLIB subset, fixed-field MPS names with spaces | Wider format interoperability/fuzzing; integer/nonlinear QPLIB remains unsupported |
| 6 — Industrial examples | Existing refinery/supply-chain/scheduling plus production LP, coupled dispatch QP, integer dispatch MIQP | Larger validated domain cases and formulation documentation; current data is synthetic |
| 7 — Explainability | Selection metadata, classification and existing explain command | Stronger structural detection and measured suitability estimates |
| 8 — Selection | Guarded CPU simplex/PDHG auto selection, CUDA memory fallback | Robust failure recovery, warm-basis reoptimization, broader portfolio; no barrier/dual-simplex/concurrent engine |
| 9 — Reproducibility | CPU/CUDA Docker recipes, compose and reproduction script | CUDA container/runtime validation and full reproduction-script run |
| 10 — Checkpoint/resume | Existing warm starts only | Exact solver/tree checkpoint and resume **not implemented** |
| 11 — Observability/Arena | New comparison runner/UI, report links, device-wide GPU telemetry, existing solve logs | Latest Arena API/browser/mobile/cancellation tests and live global MIP-gap telemetry |

## Validation at the checkpoint

- Latest CPU and CUDA builds: core/research CTest, CLI/Python tests, performance-profile tests and dependency policy passed.
- Research validation reported 15,831 assertions with CUDA exercised. Assertions are correctness checks, not a performance score.
- Latest Eigen-enabled AddressSanitizer/UndefinedBehaviorSanitizer build: both core and research tests passed.
- Latest CPU Docker image built successfully (`vantage:submission-cpu`); confirm final runtime smoke output before relying on it for submission. CUDA Docker has not been validated.
- New dashboard JavaScript syntax and Python compilation passed. Earlier seven dashboard API tests passed **before** the newest Arena/telemetry changes; latest functional browser validation is pending.
- An exploratory 91-model Netlib campaign used one repetition, two-second limits and a loaded machine, before subsequent hardening/Eigen integration: VANTAGE CPU auto returned 28 OPTIMAL, HiGHS reported 84 OPTIMAL. These are historical exploratory results, not current final solve rates or speed evidence. Latest baseline verification/status handling still needs a campaign run.

## Immediate teammate checklist

1. Validate Arena end to end: CPU/CUDA/HiGHS availability, errors retained, median summaries, cancellation, safe report paths and mobile layout. Run dashboard tests against latest code.
2. Run CPU and CUDA Docker smoke tests and `scripts/reproduce_submission.sh` end to end. Test `VANTAGE_SPARSE_LU=OFF` build as well.
3. Freeze the commit/binary, then run repeated public campaigns with equivalent tolerances/time/thread limits. Record load, raw failures, fingerprints, end-to-end timings and independent verification; avoid reusing historical timing as current evidence.
4. Run matched-algorithm GPU ablations for monitoring, graphs, indices and mixed precision. CPU auto may choose simplex while CUDA uses first-order methods: label that difference explicitly.
5. Harden sparse QP convexity handling and simplex stability before promoting automatic paths. Add fallback policy for numerical failure.
6. Expand reversible presolve and validated branch-and-cut incrementally; keep experimental features opt-in until benchmarked.
7. Implement genuine checkpoint/resume if required; warm-start files do not restore a search tree.

## Important limits for submission claims

General sparse QP uses a smooth primal-dual splitting path, not a claimed implementation of rAPDHG/HPR/PCG. Large Q validation currently accepts a conservative sufficient class; it does not recognize every PSD matrix. Simplex is revised **primal**, not revised dual; default PDHG remains available. Sparse LU uses vendored Eigen numerical linear algebra, not an external optimization solver. Root MIR cuts are restricted, not a full generic separator suite. Standalone MIP certificates validate incumbents, not the complete tree optimality proof. GPU monitoring reduces transfers but does not make all control/presolve device-resident. No barrier/IPM, HIP, multi-GPU, NLP or MINLP implementation is claimed.

## Files and restart commands

See `docs/submission_round_20260927.md`, `docs/algorithms.md`, `docs/dependency_policy.md`, `docs/supported_formats.md` and `docs/development_log.md` for implementation details. New engines: `src/simplex.cpp`, `src/cuts.cpp`; dashboard: `dashboard/server.py`, `dashboard/static/app.js`; campaigns: `benchmark/download_public.py`, `benchmark/run.py`.

Start with the documented build/test commands in README and `scripts/reproduce_submission.sh`. Existing validation logs are local `/tmp/submission-tests.log`, `/tmp/submission-cuda-tests.log`, `/tmp/submission-sanitize.log`, `/tmp/submission-docker.log`; they are not durable repository artifacts. Public datasets/results are intentionally ignored by Git; regenerate from manifests and downloader. Review and archive final validation output in a dated results directory on the teammate's laptop.
