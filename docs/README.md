# NIRYUKTI documentation

## Start here: current implementation and full demonstration

- [Complete implemented-feature and mathematics reference](implemented_features_and_mathematics.md), including mapped primary research papers, source links and exact scope.
- [Complete demo walkthrough and speaking script](complete_demo_walkthrough.md), including dashboard, codebase, CLI, packages/API, industrial planning, verification and evidence; full and 15-minute routes.
- [0.2.2 release evidence](release_0_2_2.md) and [latest GPU/conflict validation](gpu_presolve_bound_conflicts_20260928.md).

Older dated records are historical. Use the current reference for support status;
do not interpret an earlier “pending” note as the present implementation state.


Start here when preparing explanations, slides, submissions, or further research.

| Document | Purpose |
| --- | --- |
| [Latest solver completion](solver_completion_20260927.md) | New engines, actual checkpoints, GPU/MIP changes, validation and limitations |
| [Submission round](submission_round_20260927.md) | Current integration, algorithms, scope and pending work |
| [Operations guide](operations.md) | Build, run, validate, demonstrate and maintain the documentation |
| [Large-model stress campaign](stress_campaign_20260927.md) | Public railway/Netlib and million-variable synthetic screening, protocol and retained limits |
| [Development log](development_log.md) | Dated changes, decisions, experiments, evidence and unresolved work |
| [Architecture](architecture.md) | Module boundaries, ownership and execution flow |
| [Algorithms](algorithms.md) | Mathematical formulation, iteration rules and verification |
| [Supported formats](supported_formats.md) | Input/output syntax and supported problem classes |
| [Phase 2 comparison](phase2_results.md) | Generated before/after tables and current comparison with HiGHS |
| [Validation](validation.md) | Measured benchmark evidence and its practical limits |
| [Benchmark methodology](benchmark_methodology.md) | How results are produced and compared fairly |
| [Research features](research_features.md) | Opt-in experiments, source review, unfinished work and validation |
| [Numerical robustness](numerical_robustness.md) | Numerical safeguards, tests and known weaknesses |
| [Dependency policy](dependency_policy.md) | Independence of the solving core and external baseline separation |
| [Phase 2 roadmap](phase2.md) | Remaining implementation and research priorities |
| [Presentation evidence](presentation_evidence.md) | Reusable project narrative and rules for citing results |
| [Dashboard guide](../dashboard/README.md) | Starting and using the local dashboard |

Raw run files remain in `results/`; generated results are not automatically versioned. Archive the exact result directory, executable, manifest and dataset checksums with any submission. Documentation must name the measured version/configuration and distinguish historical results from current experiments.

## Independent certificates

See [certificate lifecycle, CLI/API, replay checks and limitations](certificates.md).
Run `scripts/run_certificate_demo.sh` for an offline valid → invalid → valid demonstration.

| [Auto Solver](auto_solver.md) | Existing method/device advisor, structural analysis, dry-run and policy limits |
