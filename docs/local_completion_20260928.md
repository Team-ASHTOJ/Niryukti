# Local completion and recording handoff — 28 September 2026

This is a bounded engineering pass on the current NVIDIA laptop. It does not convert a research prototype into a production solver. These source improvements postdate the immutable published 0.2.1 artifacts; build from this revision to use them.

## Implemented in this pass

- Barrier checkpoints save actual primal/equality-dual/inequality-dual/slack Newton state, best independently checked candidate and iteration counter. Resume validates dimensions, finite values, strict slack/multiplier positivity, model fingerprint and numerical configuration. Time and iteration budgets can extend.
- Simplex checkpoints save the standardized basis, phase (including warm dual reoptimization), degeneracy counter and cumulative pivots. Resume checks original and transformed model fingerprints and basis uniqueness/range, then refactorizes the saved basis. This preserves the algorithmic state; floating-point trajectories need not be bitwise identical after refactorization.
- Concurrent portfolios save a manifest plus separate actual child-engine states. Keep the manifest and child files together; relative references allow moving their directory. Threads/engine set and numerical configuration must match. Resume restarts the race from the saved engine states, without promising the same timing-dependent winner. Checkpoints are trusted local continuation data, not proof certificates.
- `--method auto --resume ...` recognizes barrier, simplex and portfolio schemas and resumes their engine rather than feeding their state to PDHG.
- Large singular sparse QP validation uses AMD ordering before guarded semidefinite elimination. A 1,200-variable singular weighted-star Gram matrix exercises this path; an indefinite variant must still be rejected. No diagonal perturbation changes the user's objective. Arbitrary fill-heavy or numerically uncertain PSD matrices remain guarded.
- Presolve/scaling checks cancellation and the local time budget at major boundaries. Sparse factorization and individual large numerical operations are still not strictly preemptible.
- A short offline recording script solves industrial models, independently verifies results, creates reports and demonstrates a genuine barrier checkpoint/resume. All generated evidence is retained with checksums.

## Run and record

```sh
NIRYUKTI_BINARY=/path/to/build/niryukti ./scripts/recording_demo.sh
NIRYUKTI_BINARY=/path/to/build/niryukti python3 dashboard/server.py --host 127.0.0.1 --port 8080
```

Suggested video: model analysis → refinery solve and verified residuals → coupled QP → supply-chain integer incumbent with proof scope visible → checkpoint/resume → frozen benchmark report. Show measured losses as well as wins. The 108-run frozen campaign is documented in `docs/release_0_2_1_campaign.md`; these changes do not retroactively alter its measurements.

## Validation from this laptop

CPU and CUDA full suites passed, including CLI/Python API, certificates, MIP conflicts, service, cancellation and checkpoint tests. Research tests passed 15,191 assertions on CPU and 16,384 with the RTX 4060 CUDA backend. Desktop/mobile route, live solve, import and download browser checks passed without browser errors. A real native E226 PDHG process was interrupted with SIGINT after entering its iteration loop and returned `INTERRUPTED` cleanly. The recording script completed and generated independently verified results and HTML reports, including an explicit CUDA graph/device-monitor solve.

Public npm 0.2.1 metadata is visible, and its tarball SHA-256 matches the CI-tested artifact. Native JSON/certificate version fields were found to still report hardcoded 0.2.0; source now derives these from the CMake project version instead.

A normal `--gpus all` container attempt failed with `failed to discover GPU vendor from CDI: no known GPU vendor found`. No host daemon configuration was changed. The machine needs NVIDIA Container Toolkit/CDI setup before that validation can pass.

## Still pending

Complete GPU presolve compaction/reversible dual provenance, fully device-resident control, general dual/implication conflicts and tableau separators, unrestricted difficult singular QP support, direct production-grade GPU barrier, broader performance/calibration campaigns, and full integer tree-proof replay remain substantive projects. Second-laptop evidence cannot be generated on this laptop. Standard `docker --gpus all` also requires the absent NVIDIA Container Toolkit/CDI host configuration; manual device passthrough is not equivalent validation.

The existing first-order/MIP checkpoints, GPU graphs/monitoring/batched branching, dynamic restricted cuts/conflicts, CPU/GPU verification, API and dashboard remain in place. AMD/HIP remains intentionally deferred.
