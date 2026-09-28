# NIRYUKTI documentation

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
