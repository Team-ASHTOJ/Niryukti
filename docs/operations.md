# Running, validating and presenting NIRYUKTI

## Start from a clean build

Run `./scripts/build.sh`, then `./scripts/run_tests.sh`. `./build/vantage devices` reports usable hardware. A CPU-only installation is supported; a requested but unavailable CUDA device returns `UNSUPPORTED`.

Start `./scripts/run_dashboard.sh` and open <http://127.0.0.1:8080>. Use Solve workspace to select or import a model, choose CPU/CUDA/auto, set tolerance/time/threads and start a real execution. Download the result JSON for later verification. The dashboard is intended for local use; no cloud account or network service is required.

## Numerical validation commands

```bash
./build/vantage solve examples/refinery.json --device cpu --json-out /tmp/refinery-result.json
./build/vantage verify examples/refinery.json /tmp/refinery-result.json
./build/vantage solve examples/dispatch.json --device cuda
./build/vantage solve examples/scheduling.json --mip-gap 1e-4
python3 examples/warm_resolve.py
```

A result contains model fingerprint, objective, primal/dual vectors, residuals, original-space diagnostics and timings. Phase 2 also reports rejected trial steps, integer-bound tightening counts and a saved Farkas certificate when one was found.

`verify` recomputes continuous feasibility/KKT or a stored Farkas ray. For MILP it checks the incumbent's feasibility and integrality; it does not independently replay the search tree. Always preserve status and MIP gap with the incumbent objective.

## Before presenting

1. Keep models, fonts, binaries and result directories locally available.
2. Run the core and CLI tests, then finish builds before benchmarking.
3. Benchmark the full chosen set and keep unsuccessful results.
4. Open the dashboard and inspect benchmark methodology, status and accuracy fields.
5. Demonstrate the refinery LP, scheduling MILP and dispatch QP, then explain the distinction between numerical success on these cases and industrial robustness.
6. Use [presentation evidence](presentation_evidence.md) for a slide outline and [Phase 2 comparison](phase2_results.md) for measured numbers.

## Reusable documentation workflow

For each subsequent numerical change, append its motivation, implementation scope, mathematical reference, tests, exploratory observations and unresolved issues to [development log](development_log.md). Update [algorithms](algorithms.md) when formulas or stopping criteria change; update [supported formats](supported_formats.md) when input support changes. Regenerate comparison evidence after final tests. Keep past observations labeled by phase/version; do not silently rewrite historical results.

Archive the exact source revision (including uncommitted changes if applicable), measured executable, raw output directories, manifests and model checksums with each submission or research report. Use a fresh dated output directory or rely on automatic campaign archiving.

## When a case reaches its iteration cap

The time limit and iteration budget are separate. An `ITERATION_LIMIT` can occur while time remains. Inspect residuals, then increase **Iteration budget** in the dashboard or use the CLI:

```bash
./build/vantage solve datasets/e226.mps --device cpu --iterations 200000 --time-limit 20
./build/vantage solve datasets/e226.mps --device cuda --iterations 200000 --time-limit 20
```

In the Phase 2 exploratory runs, e226 converges after 138,100 CPU / 144,479 GPU accepted iterations. The standard comparison still records its failure at the common 100,000 cap. Larger budgets are not guaranteed to resolve every limited model.

## Independent certificates

See [certificate lifecycle, CLI/API, replay checks and limitations](certificates.md).
Run `scripts/run_certificate_demo.sh` for an offline valid → invalid → valid demonstration.
