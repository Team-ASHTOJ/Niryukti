# Auto Solver and model analysis

NIRYUKTI already had structural method and device selection. This document describes that policy and the analysis output added around it. Auto mode is a deterministic rule set, not a performance predictor or learned configuration system.

## Usage

```sh
niryukti solve model.mps --auto
niryukti solve model.mps --method auto --device auto
niryukti solve model.mps --auto --device cpu
niryukti solve model.mps --auto --dry-run
niryukti analyze model.mps --device auto
niryukti explain model.mps
```

The CLI already defaults `method` to `auto`; `--auto` makes that choice explicit. Explicit method and device flags remain authoritative. For example, `--auto --device cpu` runs the automatic method rule with CPU forced. `--dry-run` parses and analyzes the model without running an optimization method. `analyze` accepts selection overrides such as `--device`, `--method`, `--scaling`, `--branching`, `--node-selection`, and `--primal-heuristic`. Output is machine-readable JSON. `inspect` remains a compact model summary.

Python's existing `solve(..., method='auto', device='auto')` interface uses the same native advisor and solver. The dashboard defaults the device to automatic selection and keeps PDHG as the selected method so the existing live convergence graph continues to receive its iteration trace; automatic method selection remains available in Algorithm controls. The Auto Solver panel shows the actual runtime method, relaxation method, backend, configured options, selection reason, and certificate check status.

## Classification and structure

The reader validates supported models and the native model represents linear objectives/constraints, optional convex quadratic objectives, and continuous/integer/binary variables. Class labels are LP, QP, MILP, and MIQP. Unsupported nonlinear models are rejected by the reader; the advisor does not invent a nonlinear method.

`analyze` reports variable/row/nonzero counts, integer/binary/continuous and finite-bound counts, fixed variables, equality/ranged constraints, diagonal/off-diagonal quadratic terms, matrix density, matrix coefficient range, and connected components of the variable-incidence graph. The component count is descriptive: NIRYUKTI does not automatically reorder or decompose the model. Sparse-storage labeling compares the estimated CSR and dense matrix storage directly. The report does not estimate runtime or label a model easy/hard without evidence.

## Method and device policy

The native `advise_model` is used by actual solves and reports. On CPU, the revised simplex candidate is limited to LPs with at most 50,000 variables, 300,000 stored coefficients, and a transformed-row estimate within the basis limit (4,096 rows with sparse LU support, 512 otherwise). A compatible initial basis selects dual simplex; otherwise the candidate is primal simplex. A small coupled QP may use predictor-corrector barrier when built with sparse LU, with at most 512 variables and constraints, tolerance at most 1e-7, and estimated Newton fill no greater than 1,000,000. Other supported continuous cases use PDHG. When simplex or barrier is selected, the initial attempt receives 35% of the time limit; recoverable failure can use the remaining budget for PDHG. Candidates are checked in original coordinates.

Automatic device choice requires a detected CUDA backend, known free memory, and an estimated model-storage footprint within an 80% memory guard. The estimated footprint is `48 * stored_AQ_nonzeros + 240 * (rows + columns + 2)` bytes. A workload normally also needs at least 100,000 stored A/Q nonzeros; explicitly requesting CUDA graphs, mixed matrix precision, GPU presolve, or batched strong branching can trigger GPU selection below that size when it fits. Smaller ordinary workloads stay on CPU to avoid transfer and launch overhead. Explicit CUDA requests retain their existing behavior; an automatic allocation failure can fall back to CPU for compatible FP64 operation. `hardware` and `selection.reason` record what the advisor observed and chose. Memory estimates are estimates, not measurements.

MILP/MIQP uses NIRYUKTI branch-and-bound (or branch-and-cut when cuts were explicitly enabled). Each relaxation can use the continuous auto method. Branching, node selection, cuts, primal heuristics, scaling, precision, and stopping limits otherwise retain their explicit settings/defaults. In particular, auto mode does not silently enable experimental cuts/heuristics or claim that a feasible incumbent is optimal. Results expose both the tree method and root relaxation method, as well as actual backend, configured options, and independent verification certificate.

No conservative/balanced/aggressive profiles or YAML config format is supplied: the current benchmark evidence does not validate competing bundles of scaling, branching, cuts, precision, and limits. Specify those options directly when needed; explicit flags override method and device automation.

## JSON and verification

`analyze`/`--dry-run` reports `problem_class`, structural model statistics, hardware availability, selected method/device/algorithm, the policy reason, current solver configuration, and the expected verification path. Solve output includes `selection` with requested and selected methods, actual device, root relaxation method when applicable, reason, and configuration. It also contains the same `verification_certificate` generated for all other solve modes. Automatic selection never skips original-model verification.

For MILP, reports distinguish solver tree status from independently replayable evidence. Certificate optimality requires a verifier-recomputed lower bound closing the gap; otherwise report feasible with unresolved gap.

## Benchmarks

Run a reproducible comparison on the checked-in four-model Netlib smoke suite with:

```sh
.venv/bin/python benchmark/run.py \
  datasets/afiro.mps datasets/adlittle.mps datasets/israel.mps datasets/e226.mps \
  --solvers auto,cpu,cuda,highs --method pdhg --runs 3 --threads 1 \
  --time-limit 5 --output results/auto-solver-<date>
```

`auto` is the actual NIRYUKTI automatic method/device path; `cpu` and `cuda` run explicit PDHG defaults; `highs` is the external reference adapter. Every attempt, including unavailable CUDA, timeout, limit, and failed verification, stays in the raw records. Each NIRYUKTI output is independently verified. This smoke campaign is directional only; consult `docs/benchmark_methodology.md` before interpreting it. It is not a claim of a general speed improvement.

### Retained campaign (28 September 2026)

The campaign used the nine listed models, three runs per engine, one CPU thread, 100,000 iterations, a 5-second solver limit and 1e-6 tolerance. The exact manifest, raw outputs, commands, input/source checksums, verification files and reports are retained locally in `results/auto-solver-20260928/`. This environment had no CUDA build, so all 27 explicit CUDA runs returned `UNSUPPORTED`; they are recorded rather than presented as timing comparisons. HiGHS 1.15.1 was available through `.venv`.

| Model | Auto selected method | Auto median s | CPU PDHG median s | HiGHS median s | Auto verification |
|---|---|---:|---:|---:|---|
| refinery.json | revised primal simplex | 0.0041 | 0.0059 | 0.0010 | verified optimal |
| dispatch.json | PDHG | 0.0005 | 0.0006 | 0.0010 | verified optimal |
| supply_chain.json | branch-and-bound; revised primal simplex relaxations | 0.0020 | 0.0030 | 0.0171 | incumbent feasible; tree proof not replayed |
| scheduling.json | branch-and-bound; revised primal simplex relaxations | 0.0056 | 0.0206 | 0.0233 | incumbent feasible; tree proof not replayed |
| afiro.mps | revised primal simplex | 0.0018 | 0.0072 | 0.0011 | verified optimal |
| adlittle.mps | revised primal simplex | 0.0440 | 0.6011 | 0.0024 | verified optimal |
| israel.mps | revised primal simplex | 0.2282 | 2.7824 | 0.0052 | verified optimal |
| e226.mps | simplex, then PDHG recovery | 5.02 | 5.02 | 0.0098 | time limit; final candidate failed 1e-6 verification |
| refinery_large.mps | PDHG | 9.17 | 9.13 | 0.4078 | time limit; final candidate failed 1e-6 verification |

These are local end-to-end medians from three short repetitions. Auto and CPU each reported optimal solver status for 7/9 models, with both unresolved on e226 and refinery_large; Auto selected simplex for compact LPs and improved several CPU PDHG timings, while remaining much slower than HiGHS on the larger LPs. The two MILP statuses called OPTIMAL by solver telemetry only received independent incumbent-feasibility verification, not tree-proof verification. Auto did not outperform every alternative, and the unavailable CUDA arm prevents any CPU/GPU comparison.

## Limits

The advisor uses fixed, inspectable structural guards. It does not select by measured per-GPU performance, compiler/device-specific calibration, or runtime feedback. CUDA compatibility is represented by the available backend and allocation guards; compute capability is not separately used. Matrix density, coefficients, and incidence components are reported but currently do not alter the method/device thresholds. Automatic scaling selection, time-budget prediction, parameter tuning, block decomposition, and advanced MILP profiles remain future work. There is no fabricated runtime estimate.
