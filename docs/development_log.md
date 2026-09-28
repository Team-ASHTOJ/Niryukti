# NIRYUKTI engineering record

This is the working record for implementation, experiments and decisions. It distinguishes preliminary observations from repeatable benchmark evidence. The official SIH26119 statement governs scope; the supplied AI plans are suggestions.

## 2026-09-11 — Phase 2, continuous solver and verification

**Problem.** Phase 1 solves six of the nine selected demonstration/benchmark cases. Israel and e226 reach iteration limits; refinery scheduling cannot close its MILP bound. Candidate verification repeatedly constructs a sparse transpose and calculates an expensive conservative LP bound.

**Implemented so far (CPU validated; CUDA adaptation in progress).**

- A reusable independent verifier caches the transpose and activity buffers. Monitoring uses ordinary floating-point diagnostics; final convergence and MILP pruning still require the conservative bound calculation.
- Row violations are normalized against the violated endpoint itself. An unrelated large endpoint must not conceal infeasibility.
- Adaptive PDHG trial steps use a local displacement/coupling stability check, reject unstable trials, and form step-weighted averages. Initial primal/dual weighting uses objective and right-hand-side magnitudes. The forced restart interval grows with elapsed iterations.
- The implementation is written independently from mathematical descriptions. It does not import optimization solver code.

**Mathematical reference.** Applegate et al., *Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient*, sections 3.1–3.3: <https://arxiv.org/abs/2106.04756>. Our implementation uses an absolute coupling test and a KKT-based restart heuristic; it is not the full PDLP algorithm and does not inherit a convergence guarantee for every heuristic combination.

**Checks completed at this checkpoint.** CPU build and all 143 existing assertions pass. Single-run exploratory solves at the normal 1e-6 tolerance produced:

| Case | CPU status | Accepted iterations | Observed elapsed seconds |
| --- | --- | ---: | ---: |
| afiro | OPTIMAL | 1,100 | 0.0044 |
| adlittle | OPTIMAL | 29,300 | 0.1528 |
| israel | OPTIMAL | 50,300 | 0.6976 |
| e226 | ITERATION_LIMIT | 100,000 | 1.6360 |
| synthetic refinery | OPTIMAL | 500 | 0.0028 |
| diagonal dispatch QP | OPTIMAL | 100 | 0.0005 |
| synthetic scheduling MILP | OPTIMAL | 215,800 total node iterations | 0.6001 |
| synthetic supply chain MILP | OPTIMAL | 205,300 total node iterations | 0.6498 |

These are exploratory runs, **not publication-ready medians or fair baseline comparisons**. MILP optimality is assessed by incumbent feasibility/integrality and global bound gap, not the continuous KKT metric of a rounded incumbent. The remaining e226 failure is retained.

**Artifacts.** `results/phase2/*.cpu.json` contains raw results. `results/phase2/before/vantage` preserves the previous CUDA executable for controlled comparisons. `docs/validation.md` records the Phase 1 benchmark snapshot.

**Next checkpoints.** Match adaptive behavior on CUDA without per-iteration host vector transfers; add focused regression tests; rerun the complete selected suite and preserve unsuccessful runs; update dashboard and submission-ready evidence only from measured results.

### CUDA, certificate and MILP implementation checkpoint

- CUDA now performs local trial reductions, accept/reject decisions and weighted averaging on device. Only control scalars return per monitoring chunk. Full vectors are downloaded at candidate checkpoints. Accepted and rejected steps are reported separately; GPU checkpoint positions can differ from CPU because chunks contain attempted steps.
- Both CPU and CUDA exploratory Israel runs converge at 1e-6. e226 remains at the 100,000-iteration limit on both backends. No default tolerance or iteration ceiling was weakened to change these outcomes.
- Added a box/row Farkas certificate: evaluate the zero-objective Lagrangian conservatively and require a positive margin. Saved rays can be independently replayed with `verify`; zero or incorrectly signed rays fail. This does not provide general recession certificates or proof artifacts for all presolve/MILP outcomes.
- Added up to five integer-bound propagation passes per MILP node, using outward-rounded activity intervals. Thirty additional small signed integer models are checked against exhaustive enumeration. These augment the existing 25 binary enumeration cases.
- A compiler error from mixed signed/unsigned `auto` declarations in propagation was corrected before validation. Current source is being rebuilt and tested; preliminary timings remain separate from final measurements.

For submission preparation, use [presentation_evidence.md](presentation_evidence.md). The implementation details and formulas are maintained in [algorithms.md](algorithms.md), and the documentation map is [README.md](README.md).

### Validation completed before performance measurement

- CUDA Release: 262 assertions passed, including 55 exhaustively enumerated MILPs.
- CPU-only Release: 240 assertions passed.
- CPU AddressSanitizer + UndefinedBehaviorSanitizer: the same 240 assertions and CLI integration tests passed with leak detection enabled.
- CLI tests replay a valid saved infeasibility ray and reject a tampered zero ray; saved solutions, malformed input, Python API and changed-model warm starts also pass.
- Production dependency policy passes. No comparison solver dependency entered the core.
- Before/after campaign settings: all nine selected cases, 3 measured repetitions, one separate CUDA warmup, 4 threads, 1e-6 tolerance and 20 s time limit. The first baseline attempt was interrupted because compilation was still active; its partial directory is retained by the harness and is excluded from final comparison. Final campaigns run sequentially after builds/tests finish. The host remains a shared desktop, so timing variability is explicitly disclosed.
- `benchmark/compare.py` generates the complete comparison document directly from raw runs, checks equal settings and dataset checksums, and refuses incomplete campaigns. It shows speed ratios only for cases optimal in every before/after repetition.

### Phase 2 measured outcome and dashboard integration

The complete matched campaigns are archived under `results/phase2-baseline` and `results/phase2-final`; [the generated table](phase2_results.md) records hashes, timestamps, settings and every result. Both NIRYUKTI backends improve from 6/9 to 8/9 cases optimal in all three repetitions. HiGHS solves 9/9. Worst relative objective difference among optimal NIRYUKTI runs is 1.26488e-7.

Large synthetic LP median end-to-end times: old CPU 18.128 s → new CPU 1.297 s; old GPU 11.785 s → new GPU 1.065 s. Current HiGHS is 0.661 s. GPU is 1.22× faster than our current CPU on this case, but slower than HiGHS. These shared-laptop measurements include variability; use the raw repetition spread when discussing speed.

ADLITTLE GPU regresses from 0.781 s to 1.821 s. CPU and CUDA checkpoint positions differ under rejected steps, which can change restart decisions; this is a plausible investigation target, not an established causal attribution. e226 remains at the iteration cap with KKT error above tolerance. Neither result is hidden.

Dashboard now displays the completed Phase 2 campaign (8/9 coverage, measured timings and objective agreement), with manifest-derived methodology. Six dashboard/API tests pass. Browser validation passes desktop/mobile layout, filtering, evidence drawers, actual solves, model upload and solution download without browser errors. The one-command demo completes; the changed-demand warm solve is optimal in 300 iterations versus 500 cold. Demo timings are demonstration output, not substitutes for the isolated sequential before/after campaigns.

Remaining priorities: convergence/restart merit functions for e226 and ADLITTLE, general sparse convex QP, wider benchmark coverage, persistent GPU contexts and stronger MILP algorithms. This milestone is a validated prototype upgrade, not completion of the full long-term solver roadmap.

### e226 investigation and execution controls

A separate extended-budget CPU experiment reaches `OPTIMAL` at 138,100 iterations (KKT 8.8941e-7). With restarts disabled, 500,000 iterations still hit the limit (KKT 3.86e-5); fixed-step mode also hits that limit (KKT 1.057e-3). These are exploratory ablations, not full-suite performance claims. Commands and results are preserved in `results/phase2/e226_extended.json`, `e226_without_restart.json`, and `e226_fixed_step.json`.

CUDA independently reaches `OPTIMAL` at 144,479 accepted iterations (KKT 9.0678e-7) with an extended budget; saved result: `results/phase2/e226_extended_cuda.json`. This establishes that the 100,000-iteration benchmark failure is not a claim that the solver can never solve e226. The published standard-budget table remains 8/9 and is not overwritten.

The dashboard now exposes iteration budget (default 100,000, maximum 2,000,000) and adaptive-step toggle. Server validation and the actual CLI command carry these options. Benchmark runner also accepts `--iterations`, and comparison/report methodology reads the recorded budget. Seven dashboard/API tests pass; the browser test additionally checks that a custom iteration budget is accepted by a real run.

### Extended-budget repetition check completed

`results/phase2-e226-extended` uses the identical 0.2 executable, 200,000 NIRYUKTI iterations, 20 s, 4 threads and three measured runs per backend. CPU, CUDA and HiGHS all return OPTIMAL in all three repetitions for e226. This separate experiment is now generated into the comparison document via `--extended`; the standard nine-case table remains unchanged.

- cpu: 3/3 optimal; median end-to-end 2.236620 s.
- cuda: 3/3 optimal; median end-to-end 6.159601 s.
- highs: 3/3 optimal; median end-to-end 0.019490 s.

No numerical core changes were made after the tested/measured 0.2 executable was built. Subsequent work added execution controls, benchmark budget metadata and documentation. The dashboard is available at http://127.0.0.1:8080.


## 27 September 2026 — optimization checkpoint

Added and tested CUDA Halpern variants, graph execution, 32/64-bit CSR selection, experimental mixed matrices and sparse off-diagonal convex QP with CPU/CUDA gradient paths and original-space verification. CUDA research assertions: 2,613. Stopped at user request; partial benchmark evidence retained without speedup claims. See [handoff and pending work](optimization_handoff_20260927.md) and [audit](optimization_audit.md).

## 2026-09-27 — main integration and submission engineering

Frontend fast-forward merged and pushed to main; default branch updated. Sparse QP transformations, conservative MIQP bounds, compact revised primal simplex, MIR cuts, node storage/policies, CUDA diagnostic gating and certificate checks are being implemented and validated. See [the detailed record](submission_round_20260927.md); benchmark claims will be added only after measured campaigns.

## 2026-09-27 — state continuation, numerical recovery and large-model evaluation

Implemented actual CPU/CUDA first-order iterate checkpoints and full MIP frontier snapshots; sparse singular PSD recognition; revised dual-simplex reoptimization; independent predictor-corrector barrier with optional GPU sparse Newton solves; verified concurrent continuous portfolio; reusable CUDA matrix contexts, batched branching and directed-rounding integer propagation; restricted global/local MIR cut pools, conservative binary conflicts and local branching. All support boundaries and validation evidence are recorded in [the completion record](solver_completion_20260927.md).

The scheduling regression exposed tiny forbidden-sign simplex row multipliers that prevented a finite safe bound. Projecting multipliers onto the exact interval dual domain, followed by independent re-verification, restored the proof path. Continuous node bounds now also receive conservative propagation. CPU/CUDA research tests include this regression, checkpoint continuation, GPU numerical Newton solves, large singular PSD acceptance/rejection and enumerated MILPs.

Added an isolated SCIP baseline and an explicit large-model screening campaign: public railway/MIPLIB instances, Netlib convergence tests and a streamed million-variable planted LP. [The stress protocol](stress_campaign_20260927.md) distinguishes public models, derived LP relaxations and synthetic data. AMD remains deferred. Full GPU control/compaction, unrestricted conflict analysis, all-method checkpointing and physical second-laptop/container-GPU validation remain incomplete; no claim of commercial-grade completeness is made.

## 2026-09-28 — upstream integration and distribution

Pulled the friend’s replanning, native C ABI, evidence and stability extensions from origin/main (84bcc7e, 4c0f066). Added self-contained CPU Python wheel/sdist builds, installed native-artifact discovery and CLI, source-based npm API with TypeScript and cancellation, distribution CI, Docker native/Python support and a draft local Arch recipe. Full CPU/CLI/Python/dashboard checks passed. Registry publication and a top-level redistribution license remain pending. See [the integration and distribution record](distribution_20260928.md).

## 2026-09-28 — copyleft licensing and release automation

Adopted AGPL-3.0-only for original code at the user’s request to keep covered redistributed/network modifications open. Added LICENSE, NOTICE, package SPDX metadata, contribution terms and full numerical license notices. Added a manually triggered manylinux Python/npm build-and-test workflow with separate OIDC publishing environments. Account-side PyPI/npm setup is still required; passwords/tokens are not requested in chat. See [publishing guide](publishing.md).

## 2026-09-28 — public NIRYUKTI distribution names

Renamed PyPI/npm distributions and installed CLI to `niryukti`; added `niryukti` Python imports with legacy implementation compatibility. The native CLI now builds as `niryukti`, retaining a developer-only `vantage` copy for existing scripts. Publisher configuration must use the PyPI/npm project name `niryukti`, not the earlier provisional name.

## 2026-09-28 — public package release

Published `niryukti` 0.2.0 on PyPI via GitHub OIDC and on npm after the owner completed browser 2FA approval. Public registry installations passed Python CLI/subprocess/native and Node source-build/solve/error/cancellation checks; registry checksums matched retained artifacts. Recorded limits: initial prebuilt Python wheels are Linux x86_64 CPU, npm requires a compiler, CUDA remains a separately built backend, and future npm OIDC configuration is still an owner-side step. See [release evidence](release_20260928.md).
