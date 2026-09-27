# Roadmap implementation audit — 27 September 2026

Baseline: [dated handoff](teammate_handoff_20260927.md), checked against the current working tree. **The roadmap is not fully implemented; none of its milestone exit gates is established by this audit.** Existing `src/io.cpp` edits were retained and included in builds, not authored by this audit. No revision was committed or frozen. A passing test suite does not establish a benchmark improvement.

## Task coverage

| Task | Current evidence and remaining work |
| --- | --- |
| A: Arena, cancellation, downloads, imports | API/worker coverage expanded to 12 passing tests. Real Arena worker retains CPU/CUDA/HiGHS outcomes. Queued cancellation tested; running-process/browser/mobile validation remains pending. |
| A: CPU/CUDA, sanitizers, Docker, reproduction | CPU reproduction, ASan/UBSan core/research, and non-Eigen core/research passed locally. CUDA compiler/device unavailable; Docker daemon stopped. |
| A: frozen repeated public subset | Four cached public Netlib models measured three times at equal CPU budgets; all outputs retained. Dirty-tree diagnostic run, not a submission freeze or CPU/GPU/baseline comparison. |
| A: feature matrix, CSV/HTML, certificates, refinery, limits | Matrix below; reports and serialized solutions now contain separate verification evidence. Synthetic refinery runs pass. Offline reproduction on a second laptop remains untested. |
| B: sparse QP hardening and tests | `model.cpp`, `solver.cpp`, `preprocess.cpp`, research tests implement sparse convex QP and original-space verification. Large non-diagonally-dominant PSD recognition and broader transformation tests remain incomplete. |
| B: simplex numerics and recovery | Revised primal simplex has basis solves, degeneracy handling, limits and correctness tests. E226 still fails numerically under automatic selection; numerical-failure recovery remains incomplete. |
| B: basis warm starts and revised dual simplex | Not implemented; initial primal/dual vectors are not a saved basis. |
| B: general rays | Verified Farkas candidates and restricted recession handling exist. General independently verified unbounded rays remain missing. |
| B: configuration comparison and QP research | PDHG/Halpern/polishing options and tests exist. No new frozen comparison or primary-reference rAPDHG/inexact-proximal implementation completed. |
| C: transfer/SpMV/reduction/launch profiling | Transfer/iteration timers and device-wide monitoring exist; solver-attributed profiling/crossover evidence remains pending on supported hardware. |
| C: persistent contexts and device control | CUDA workspaces exist within a solve, not across repeated solves. Some scalar control is on device; full requested work is incomplete. |
| C: graphs, indices, mixed precision | Opt-in implementations and CUDA research tests exist; no fresh device validation or independent measurements here. |
| C: batched branching/OBBT/GPU presolve | Not implemented. |
| D: reversible presolve/propagation | Existing presolve mappings and outward-rounded integer propagation are tested; broader transformations remain pending. |
| D: dynamic cuts/conflicts | Restricted root binary/MIR cuts exist; controlled dynamic pools, general implied-bound separation, conflict explanations and tableau cuts are missing. |
| D: branching and heuristics | Reliability branching, pump and RINS exist with safe-bound logic; local branching and repeated independent ablation campaigns remain missing. |
| D: exact checkpoint/resume | Not implemented. Queue, bounds, pseudocosts, incumbent and search state are not serialized; warm starts do not satisfy this task. |
| D: MIQP oracles | Existing research tests enumerate a small integer domain and check conservative bounds/incumbents. Broader randomized small-domain coverage remains pending. |
| E: gap, memory, explanations, analysis | Final gap, solve diagnostics and classification exist. Live global MIP-gap history, measured solver memory and expanded structural analysis remain missing. |
| E: industrial cases | Synthetic refinery, production, dispatch, scheduling and supply-chain models exist. Larger validated cases and complete units/provenance documentation remain pending. |
| E: evidence archives and generated tables | Runner already archives binary, checksums and manifest. Now also copies project sources/model inputs and emits verifier evidence in generated HTML. Vendored dependencies/toolchains are not bundled into a self-contained source distribution. |
| E: parsers/fuzzing/install/backends | Parser rejection tests and Eigen/non-Eigen builds exist. Continuous fuzzing, wider interoperability and installation matrix remain incomplete. |
| Later research | Barrier/IPM, revised dual simplex at scale, portfolios, HIP/ROCm, multi-GPU, NLP and MINLP are not implemented. No placeholder options added. |

## Supported-feature matrix

| Feature | Implementation | Claim limit |
| --- | --- | --- |
| Continuous LP | CPU/CUDA primal-dual; CPU revised primal simplex | Numerical tolerance results, not exact-arithmetic proofs; CUDA not retested here |
| Convex sparse QP | Smooth primal-dual splitting and original-space KKT | Not rAPDHG/HPR/PCG; large Q requires conservative PSD certificate |
| MILP / convex MIQP | Branch-and-bound, conservative bounds, restricted root cuts | Standalone verifier checks incumbents, not tree replay |
| MPS/LP/JSON/QPLIB | Documented subsets in `supported_formats.md` | Integer/nonlinear QPLIB unsupported |
| Warm start | Primal/dual vectors | Neither basis warm start nor exact search resume |
| GPU telemetry | Device-wide NVIDIA counters | Not solver-attributed utilization or peak memory |
| Industrial examples | Synthetic demonstration models | Not operational MRPL evidence |

## Changes and validation

Problem addressed: reproduction omitted dashboard checks; dashboard tests relied on ignored local results; CPU/CUDA benchmark rows lacked separate standalone verifier outcomes. No numerical algorithm or mathematical stopping criterion changed. The existing original-space verifier is the correctness reference; standalone tolerance is explicitly 1e-6, even when solve tolerance differs.

Changed files: `dashboard/server.py`, `dashboard/tests/test_dashboard.py`, `scripts/run_tests.sh`, `tests/test_benchmark.py`, `benchmark/run.py`, `benchmark/report.py`, and documentation. Non-object JSON API requests now return HTTP 400. Tests use temporary benchmark fixtures and the requested binary. CPU/CUDA serialized solutions are independently re-read; unverified `OPTIMAL` claims become `VERIFICATION_FAILED` in campaign records while raw solver output stays intact.

Commands run:

```sh
VANTAGE_REPRO_BUILD=build-roadmap bash scripts/reproduce_submission.sh results/roadmap-verified
VANTAGE_TEST_BINARY=build-roadmap/vantage python3 dashboard/tests/test_dashboard.py
cmake -S . -B build-roadmap-sanitize -DVANTAGE_SANITIZE=ON -DVANTAGE_OPENMP=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build build-roadmap-sanitize -j 4
ctest --test-dir build-roadmap-sanitize --output-on-failure
cmake -S . -B build-roadmap-portable -DVANTAGE_SPARSE_LU=OFF -DVANTAGE_OPENMP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-roadmap-portable -j 4
ctest --test-dir build-roadmap-portable --output-on-failure
python3 benchmark/run.py datasets/afiro.mps datasets/adlittle.mps datasets/israel.mps datasets/e226.mps --binary build-roadmap/vantage --method auto --solvers cpu --runs 3 --time-limit 2 --output results/roadmap-public
```

The full reproduction run passed core/research, CLI/Python, profile, benchmark evidence, 11 API tests and dependency checks; the subsequent Arena worker test brought the dashboard suite to 12 passing tests. Sanitizer and portable builds each passed both CTest suites. Localhost tests require network-bind permission in a restricted sandbox.

Before/after evidence change: the initial `results/roadmap-audit` campaign has no CPU standalone-verification status; `results/roadmap-verified` has it, retained certificates and visible report provenance. Both retain 21 OPTIMAL and 3 UNKNOWN demo runs. The three UNKNOWN runs are scheduling; their incumbents verify feasible, without a tree proof. No speedup or solve-rate improvement is claimed.

`results/roadmap-public` retains all 12 measured runs: AFIRO, ADLITTLE and ISRAEL each verify optimal three times; E226 returns NUMERICAL_ERROR three times. These were diagnostic measurements on a shared machine, with build activity, not controlled performance evidence. All result folders are ignored local artifacts; archive them separately before moving machines. Inspect each manifest for binary hash, source hashes, revision, dirty state and hardware.

Next gates remain substantive implementation and validation work, particularly numerical failure recovery, exact checkpoint/resume, general certificates, public MIPLIB/QPLIB campaigns and supported-device GPU validation. This audit does not certify submission readiness.
