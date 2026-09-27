# NIRYUKTI

**Independent Sparse Optimization Engine**

An independent sparse optimization engine for SIH26119, built in C++20 with an optional CUDA backend. The first prototype solves LPs, supported sparse convex QPs, small MILPs and convex MIQPs using its own numerical algorithms. No existing optimization solver is used to solve a NIRYUKTI model.

**Research prototype:** numerical correctness is tested on representative cases; industrial robustness, unrestricted large sparse PSD validation, and competitive large-scale MILP performance remain development work. The name does not imply support for general nonconvex global optimization.

## Dashboard

```bash
./scripts/run_dashboard.sh
```

Open **http://127.0.0.1:8080** for the local dashboard: solver overview, capability/pending-work cards, large-model and public-baseline comparisons, accuracy details, model library, live CPU/GPU solves, model import and downloadable results. The interface uses real saved measurements and the actual solver executable. See [dashboard documentation](dashboard/README.md).

## Run it

```bash
./scripts/build.sh                 # Detect CUDA; otherwise build CPU only
./scripts/run_tests.sh
./build/vantage devices
./build/vantage solve examples/refinery.json --device cuda --json-out result.json
./build/vantage verify examples/refinery.json result.json
./scripts/run_demo.sh
```

Open `results/demo/index.html` after the demo. It contains actual CPU/CUDA/HiGHS measurements, every measured run's status, raw outputs, and downloadable CSV. The demo works offline with the checked-in small instances. HiGHS is optional and runs in an external comparison process.

For a CPU-only build:

```bash
cmake -S . -B build-cpu -DVANTAGE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel 4
./scripts/run_tests.sh build-cpu
```

Requirements: CMake ≥3.24, a C++20 compiler, Python ≥3.10 for scripts. CUDA builds require the CUDA toolkit and a compatible host compiler. The build script selects GCC 15 when available; override CMake settings for other installations. GPU runtime tests run only when a usable CUDA device is present. Default arithmetic is FP64; unsafe fast-math is not enabled.

## Current prototype capabilities

| Component | Implemented behavior |
|---|---|
| Independent solver core | Own PDHG updates, preprocessing, scaling, verification, and branch-and-bound |
| Sparse matrices | COO construction with duplicate aggregation; CSR and explicit sparse transpose; 64-bit offsets and indices |
| CPU and CUDA | Shared first-order formulation; OpenMP CPU loops; Eigen numerical sparse factorization; cuSPARSE SpMV/SpMM, reusable matrix contexts, graph execution and device diagnostics |
| LP | Adaptive/restarted PDHG, opt-in Halpern/reflected methods; compact revised primal/dual simplex with recovery and basis reoptimization; experimental predictor-corrector barrier and verified concurrent portfolio |
| Convex QP / MIQP | Diagonal proximal updates; symmetric sparse convex Q via smooth splitting; QP relaxations and conservative minorant bounds for MIQP |
| MILP / MIQP | Bound-delta nodes, search policies, reliability branching, restricted root/node/global cut pools, binary no-good conflicts, pump/RINS/local branching; verified incumbents and conservative bounds |
| Preprocessing | Fixed-variable and bounded isolated-column elimination, empty rows, row-activity infeasibility checks, reversible reconstruction |
| Scaling | Iterative diagonal equilibration with original-space verification |
| Input | MPS linear/integer sections and ranges; symmetric/triangular `QMATRIX`/`QUADOBJ`; documented LP text subset; sparse JSON; supported continuous QPLIB |
| Verification | Original objective, row/bound feasibility, integrality, projected stationarity, complementarity, and Lagrangian lower bound |
| Infeasibility certificates | Independently verified box/row Farkas rays, saved as JSON and replayable through `verify` |
| API and CLI | C++ library, Python subprocess API, solve/inspect/explain/verify/convert/devices commands |
| Demonstration | Synthetic refinery blending LP, refinery scheduling MILP, supply-chain MILP, production planning LP, coupled power-dispatch QP and integer-dispatch MIQP |
| Benchmarks | External HiGHS/default-IPM and SCIP adapters, CPU/CUDA comparisons, public railway/Netlib and million-variable synthetic screening, retained failures, checksums, offline HTML/CSV and performance profiles |
| State continuation | Atomic MIP tree snapshots and CPU/CUDA first-order iterate/controller snapshots; compatible-model/backend/configuration checks |

## CLI examples

```bash
./build/vantage inspect datasets/afiro.mps
./build/vantage explain examples/refinery.json
./build/vantage solve datasets/afiro.mps --device cpu --tol 1e-6 --threads 4
./build/vantage solve examples/dispatch.json --device cuda
./build/vantage solve examples/supply_chain.json --time-limit 30 --mip-gap 1e-4
./build/vantage convert examples/toy.lp /tmp/toy.mps
./build/vantage solve examples/refinery.json --json-out /tmp/first.json
./build/vantage solve examples/refinery.json --warm-start /tmp/first.json
python3 examples/warm_resolve.py
```

A changed model requires `--warm-start previous.json --allow-model-change`; variable and row names/order must still agree. The parametric demo increases demand by 5% and records both cold and warm solves. It does not assume that warm starting always helps.

`solve` writes JSON to stdout; `--verbose` writes iteration progress to stderr. `--json-out` / `--solution-out` save the same JSON format. Exit codes: `0` optimal/successful command; `2` non-optimal solve or failed solution verification; `1` invalid input, unsupported parser syntax, or execution error. `verify` checks feasibility at `1e-6` and reports continuous optimality only if KKT also passes. MILP verification checks the incumbent, not a replay of the search tree.

## Python

[Stable replanning and signed evidence](docs/trust_and_stable_planning.md) let you penalize changes to an approved plan, lock executed decisions, and authenticate solve bundles with separately trusted public keys. Run `python3 examples/stable_refinery_plan.py` for the synthetic CPU demonstration.

The [native interface, evidence bundles and measured sensitivity reports](docs/competitor_features_20260928.md) provide in-process C/Python sessions and offline checks of exported solves. These extensions preserve the independent numerical core; GPU validation remains separate.

Persistent model sessions, coupled refinery disruption scenarios and explicitly authorized infeasibility repair proposals are available. See [refinery replanning](docs/refinery_replanning.md); run `python3 examples/refinery_replanning.py` after building. These are synthetic CPU-validated demonstrations, not industrial performance claims.

```bash
export PYTHONPATH="$PWD/python"
# Optional packaging: pip install -e . ; set VANTAGE_BINARY if the binary is elsewhere.
```

```python
import vantage

model = vantage.Model("production")
x = model.add_var("x", ub=10)
y = model.add_var("y", ub=10)
model.add_constraint({x: 1, y: 2}, ">=", 4)
model.set_objective({x: 1, y: 1})
result = model.solve(device="cpu")
print(result["status"], result["objective"], result["primal"])
```

The Python API launches the C++ binary; it is not a Python numerical implementation or a wrapper around a third-party solver. The C++ sparse API is in [`include/vantage/vantage.hpp`](include/vantage/vantage.hpp).

## Benchmarks and larger cases

```bash
python3 -m venv .venv
.venv/bin/pip install -r benchmark/requirements.txt
.venv/bin/python benchmark/run.py datasets/afiro.mps datasets/adlittle.mps \
  datasets/israel.mps datasets/e226.mps --runs 3 --threads 4 --output results/netlib

python3 examples/generate.py --crudes 16 --products 8 --periods 365 \
  --output datasets/refinery_large.json
./build/vantage convert datasets/refinery_large.json datasets/refinery_large.mps
.venv/bin/python benchmark/run.py datasets/refinery_large.mps \
  --runs 3 --threads 4 --time-limit 60 --output results/scalability
```

The larger example has 46,720 variables and 186,515 nonzeros. It is synthetic and largely separable by period; it is a sparse-computation demonstration, not evidence of realistic industrial scheduling difficulty. Request bigger dimensions for scalability experiments, with appropriate memory limits.

Read [`docs/benchmark_methodology.md`](docs/benchmark_methodology.md) before interpreting timing ratios. Reports retain unsuccessful runs. A lower primal objective alone is not a win: feasibility, tolerances, gap, and model class matter.

Opt-in research features include combined scaling, CPU Halpern/restarted/reflected Halpern LP methods, PID weighting, power estimates, LP feasibility polishing, reliability branching, binary cover/clique cuts, and feasibility-pump/RINS heuristics. The default remains PDHG with Ruiz scaling and most-fractional branching. See [research integration and validation](docs/research_features.md) for examples, supported combinations, source references and remaining work.

The historical [Phase 2 measured comparison](docs/phase2_results.md) records 8/9 selected cases solved by both NIRYUKTI backends versus 9/9 by HiGHS, including the remaining e226 limit and ADLITTLE GPU regression. A separate 200,000-iteration experiment solves e226 on CPU/CUDA in all three repetitions; it does not replace the standard-budget result. The [Phase 1 snapshot](docs/validation.md) remains available as historical evidence.

## Limits and next phase

Supported sparse QP and convex MIQP, revised dual simplex, an experimental barrier, restricted cut/conflict pools, reusable CUDA matrix contexts and actual first-order/tree checkpoints are implemented. This does not imply mature industrial robustness. Uncertain large PSD recognition, unrestricted global cut/conflict analysis, full GPU presolve compaction and fully device-resident control remain pending. Barrier/simplex/concurrent full-state checkpointing is unsupported; nonlinear and general nonconvex optimization remain out of scope. AMD/HIP is deferred.

MILP may return no incumbent or a feasible incumbent with an unresolved gap. Time checks occur at control boundaries; parsing, presolve, numerical kernels and final verification can overrun the budget. The public railway stress campaign explicitly retains these limits and process watchdog timeouts.

Device monitoring reduces routine vector downloads when enabled. Final original-space verification remains independent; host control and periodic candidate reconstruction still exist. GPU iteration speed does not imply end-to-end speedup when parsing/setup dominates. See [current implementation boundaries](docs/solver_completion_20260927.md) and [large-model protocol and evidence](docs/stress_campaign_20260927.md).

See [architecture](docs/architecture.md), [mathematics](docs/algorithms.md), [formats](docs/supported_formats.md), [Phase 2 work](docs/phase2.md), and [dependency policy](docs/dependency_policy.md).

## Documentation for development and submissions

The [documentation index](docs/README.md) links the [development log](docs/development_log.md), mathematical explanations, validation records and a [presentation/submission evidence guide](docs/presentation_evidence.md). Preliminary experiments are labeled separately from reproducible benchmark results.

## Submission round and reproducibility

Current scope, integration and remaining algorithms: [27 September engineering record](docs/submission_round_20260927.md).

```bash
./build/vantage solve datasets/adlittle.mps --device cpu --method auto
./build/vantage solve examples/coupled_dispatch.json --device cuda --gpu-monitor
./build/vantage solve examples/integer_dispatch.json --certificate-out /tmp/miqp.json
./build/vantage verify examples/integer_dispatch.json /tmp/miqp.json
python3 benchmark/download_public.py --suite netlib
python3 benchmark/download_public.py --suite qplib
./scripts/reproduce_submission.sh
# Optional baseline: VANTAGE_BENCH_PYTHON=.venv/bin/python VANTAGE_REPRO_SOLVERS=cpu,highs ./scripts/reproduce_submission.sh

docker build -f Dockerfile.cpu -t vantage:cpu .
docker run --rm vantage:cpu solve examples/production.json --method auto
docker compose up --build
# CUDA image requires a compatible NVIDIA driver/container toolkit.
docker build -f Dockerfile.cuda -t vantage:cuda .
docker run --gpus all --rm vantage:cuda devices
```

Dockerfiles provide reproducible build recipes; image validation and actual measured campaigns are recorded separately. `auto` selects compact CPU simplex where supported; otherwise it retains PDHG. Root cuts and new CUDA monitoring remain opt-in. First-order/tree checkpoints, experimental barrier/dual simplex, batched branching probes and restricted binary conflicts are now implemented and tested. Their limitations and hardware-validation boundaries are documented in the completion record; AMD remains deferred.

Current completion-round implementation, usage, trust boundaries and hardware validation: [solver completion record](docs/solver_completion_20260927.md).
