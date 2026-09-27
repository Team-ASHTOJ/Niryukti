# Solver completion round — 27 September 2026

This record describes source changes made after `1511bee`, independently of historical campaigns. Benchmarks were initially deferred; the user subsequently authorized a final campaign after correctness checks. Functional correctness checks do not establish speedups or commercial readiness.

## Implemented paths

| Component | Implementation | Deliberate boundary |
|---|---|---|
| Numerical recovery | CPU automatic simplex falls back to independently verified PDHG after numerical failure, using the remaining time budget | Does not establish that E226 now converges within a particular budget |
| Sparse QP convexity | Sparse LDLT recognizes additional large strictly positive-definite matrices; small PSD and diagonal-dominance checks retained | Sparse semidefinite elimination admits exact zero pivots only with zero remaining rows; fill/operation guards and uncertain curvature can still reject models |
| GPU context | Per-thread shared immutable A/transpose/context, matched by exact model data, backend, precision and index width; stream-ordered allocation pool | Iterate workspaces and all host decisions are not persistent/device-only |
| Batched branching | Shared-matrix GPU SpMM projected primal-dual probes, independently verified per child; reliability branching consumes verified outcomes | LP only, at most four candidate variables/eight probes; no batched QP/OBBT |
| GPU propagation | Directed-rounding row activities and integer bound tightening in synchronous device passes | Does not compact the model or replace all CPU presolve; CUDA only |
| Branch-and-cut | Root cover/clique/MIR, bounded violated node-local MIR separation, and a bounded global root-valid MIR pool with efficacy ranking, aging and deduplication | Restricted MIR, not basis-tableau GMI; local cuts discarded on node exit; global pool capped at 64 with at most 16 selected rows/node |
| Conflicts | Bounded binary no-good pool learned from established infeasibility | No general implication graph; refuses learning when nonbinary branch bounds would invalidate the explanation |
| Local branching | Hamming-ball restricted subproblem around verified incumbent | Heuristic only; neighborhood proof bounds never used globally |
| Presolve | Exact parallel-row bound intersections and guarded activity redundancy | Endpoint provenance reconstructs dual multipliers and contradiction rays; no general affine row aggregation |
| Checkpoint/resume | Atomic `vantage-tree-1` snapshots of frontier, bound deltas, warm starts/bases, incumbent, conflicts, pseudocosts, closed/unresolved bounds and counters | Trusted local state; first-order continuous CPU/CUDA snapshots additionally save backend iterates, averages, anchors, step adaptation and host restart/PID state |
| Barrier | Independent infeasible-start Mehrotra predictor-corrector LP/convex QP engine using Eigen sparse numerical Newton factorization | CPU sparse LU or CUDA sparse BiCGSTAB Newton solves; CPU KKT assembly/control, dimension/fill guards; no homogeneous infeasibility embedding |
| Revised dual simplex | Explicit structurally compatible basis import, dual-feasibility recheck, revised dual ratio tests and product-form updates | Cold start uses primal two-phase initialization; incompatible basis safely cold starts |
| Concurrent portfolio | PDHG, CPU barrier and CPU simplex race; original-space verifier selects an optimal winner; sibling-local cancellation | Engines use at least one CPU thread each, so very small requested budgets may be exceeded |
| HIP | Experimental compile-time HIP/hipSPARSE adapter and backend selection | No ROCm compiler/device here; not compilation/runtime validated; directed-rounding propagation unavailable |
| Platform | Actual child-process cancellation regression; additional dashboard method selectors and configurable binary | Browser/mobile and container outcomes are recorded below after evaluation |

## Usage

```sh
./build/vantage solve examples/refinery.json --method barrier --device cpu
./build/vantage solve examples/refinery.json --method concurrent --device auto
./build/vantage solve model.mps --method dual-simplex --warm-start previous.json
./build/vantage solve examples/supply_chain.json --branching reliability --cuts --primal-heuristic all
./build/vantage solve model.mps --device cuda --branching reliability --batch-strong-branching --gpu-presolve
./build/vantage solve model.mps --checkpoint-out tree.json --checkpoint-nodes 100
./build/vantage solve model.mps --resume tree.json --checkpoint-out tree.json
```

A resume requires the same original model fingerprint and algorithm settings. Time/node budgets can be extended. Processed-node counts remain cumulative; elapsed wall-clock budget starts afresh. Nested heuristic searches cannot overwrite the parent snapshot. Continuous first-order snapshots use `vantage-continuous-1` and the same checkpoint/resume CLI options. They preserve current/averaged/extrapolated iterates, anchors, adaptive step/controller and restart state. Backend and mathematical options must match; iteration/time budgets can be increased. Barrier/simplex/portfolio full-state checkpointing remains unsupported.

## Mathematics and trust boundaries

Barrier iterates maintain positive slack `s` and inequality multipliers `z`. Newton directions solve the regularized sparse KKT system with `Q + Gᵀ diag(z/s) G`, equality blocks, affine predictor and complementarity correction. Fraction-to-boundary steps maintain positivity. Newton residuals are checked, and original-space KKT verification alone permits `OPTIMAL`. Numerical regularization is not an optimality certificate.

Dual simplex accepts a basis only after structure fingerprint/index validation and reduced-cost feasibility checks. Negative basic values are repaired using a dual ratio test; basis solves use factorization plus product-form updates, periodically refactorized. Bounds/RHS may change while compatible matrix structure is retained. Infeasibility needs independent ray validation.

GPU propagation uses old bounds for all row calculations within a pass. Directed arithmetic encloses activities; atomic updates combine tightening without using a partly updated interval as the old interval. Integer rounding is guarded in the exactly representable integer range. CUDA functionality is tested against enumerated feasible assignments.

MILP pruning consumes existing conservative relaxation lower bounds, not approximate primal objectives. Restricted/local cuts and heuristic neighborhoods cannot leak multipliers or proof bounds into unrelated nodes. Binary conflicts contain a sufficient complete assignment mask, rather than claiming a minimal conflict explanation. Checkpoints are trusted solver snapshots; modifying frontier bounds can invalidate proof state, so they are not exchangeable external certificates.

## References used for mathematical understanding

No optimization solver implementation was transplanted or linked into the solving path.

- [Mehrotra, predictor-corrector interior-point method](https://epubs.siam.org/doi/10.1137/0806004).
- [Huangfu and Hall, parallel revised dual simplex](https://link.springer.com/article/10.1007/s12532-017-0130-5).
- [Mixed-integer rounding lecture notes](https://coral.ise.lehigh.edu/~ted/files/ie418/lectures/Lecture14.pdf).
- [AMD HIP mathematical functions and rounding limitations](https://rocmdocs.amd.com/projects/HIP/en/develop/reference/math_api.html).

## Remaining work and hardware gates

GPU direct sparse barrier factorization, a full device-resident restart/control loop, full GPU model-compaction presolve, general tableau-derived cut families, and implication/dual conflicts remain separate engineering tasks. The user explicitly deferred AMD/HIP; its experimental adapter is not a supported or validated feature. A second physical laptop cannot be validated from this machine; reproduction instructions provide a handoff, not a fabricated result. NVIDIA container execution requires the NVIDIA container runtime, which is absent here.

## Evaluation

Executed on the local RTX 4060 Laptop GPU / CUDA 13.4 toolchain:

- CUDA Release core: **263 assertions**, including 25 independently enumerated MILPs.
- CUDA Release research: **16,357 assertions**, including CPU/CUDA state continuation, tree resume, GPU barrier LP/QP, batched relaxation bounds, directed-rounding propagation and large singular PSD rejection/acceptance.
- CPU ASan/UBSan: both core and research suites passed.
- CPU portable build without Eigen sparse factorization: both suites passed; unsupported feature guards exercised.
- CUDA Compute Sanitizer research run: **0 errors**.
- CLI, malformed input, independent verification, Python API and warm re-solve tests passed.
- Dashboard API: **14 tests passed**, including cancellation of a genuinely running child process.
- Browser: desktop widths 1440/1920/2560 and mobile width 390; navigation, themes, search, details, real solve, import and download passed with no page errors.
- CPU Docker image built, ran its internal test suites, and independently verified the refinery barrier solve.
- CUDA Docker image built successfully; its CPU fallback solved coupled dispatch. NVIDIA container execution rejected by the local Docker daemon: `failed to discover GPU vendor from CDI: no known GPU vendor found`. Host CUDA execution works; container GPU access is a separate environment gate.

E226 direct CPU `auto` diagnostic, 30-second budget, four threads: **OPTIMAL**, objective −11.638930043471596, KKT 9.701436868163563e−7, end-to-end 3.524 seconds. Primal simplex encountered a basis feasibility error and automatic PDHG recovery converged. CPU barrier on that same model returned **NUMERICAL_ERROR**, KKT about 0.0031; the new barrier is opt-in and is not represented as generally robust. These are individual diagnostics, not broad performance conclusions.

The initial three-run CPU/CUDA/HiGHS campaign and retained raw failures are recorded below. Scheduling was subsequently fixed: exact dual-domain projection removes tiny forbidden simplex multiplier signs, and continuous node-box propagation was strengthened. Both changes require fresh final campaign measurements rather than rewriting historical runs. An initial partial timing run overlapped CUDA memory checking and was discarded as timing evidence. No speedup is claimed from functional checks.

## Reproduction on another laptop

```sh
./scripts/validate_completion.sh build-completion-cpu
VANTAGE_VALIDATE_CUDA=ON VANTAGE_CUDA_ARCHITECTURES=89 ./scripts/validate_completion.sh build-completion-cuda
VANTAGE_BINARY="$PWD/build-completion-cuda/vantage" python3 dashboard/server.py --port 8080
# In another terminal, after installing dashboard/browser-tests dependencies:
DASHBOARD_URL=http://127.0.0.1:8080 node dashboard/browser-tests/check.cjs
```

Set the architecture to the target NVIDIA card instead of copying `89` blindly. A second-laptop result is pending until someone executes these commands there. Builds with a full home filesystem can set `CCACHE_DIR` to a writable directory. Docker GPU execution additionally requires correctly configured NVIDIA Container Toolkit/CDI.

## Final measured campaign

[Per-instance CSV](evidence/completion_20260927/runs.csv), [HTML report](evidence/completion_20260927/index.html), [performance profiles](evidence/completion_20260927/profile.html), [manifest](evidence/completion_20260927/manifest.json).

Four local Netlib instances and six synthetic industrial models, three measured repetitions/engine, one discarded CUDA warmup/instance, 20 seconds, 500,000 iterations, four CPU threads and 1e−6 tolerance. Configuration: automatic continuous method, reliability branching, cuts and all primal heuristics; CUDA scalar monitoring enabled. This is a selected 10-model campaign, not full Netlib/MIPLIB/QPLIB coverage. Background desktop load is recorded in the manifest.

| Instance | CPU median end-to-end | CUDA median end-to-end | HiGHS median end-to-end | Caveat |
|---|---:|---:|---:|---|
| afiro.mps | 0.001235 s | 0.304349 s | 0.003536 s | All three repetitions OPTIMAL per engine |
| adlittle.mps | 0.006802 s | 2.683537 s | 0.007354 s | All three repetitions OPTIMAL per engine |
| e226.mps | 3.525660 s | 9.371298 s | 0.034535 s | All three repetitions OPTIMAL per engine |
| israel.mps | 0.031138 s | 3.714329 s | 0.018168 s | All three repetitions OPTIMAL per engine |
| refinery.json | 0.001746 s | 0.253455 s | 0.003345 s | All three repetitions OPTIMAL per engine |
| supply_chain.json | 0.003897 s | 1.051643 s | 0.041770 s | All three repetitions OPTIMAL per engine |
| coupled_dispatch.json | 0.000978 s | 0.224695 s | 0.003273 s | All three repetitions OPTIMAL per engine |
| integer_dispatch.json | 0.007456 s | 0.340913 s | 0.001306 s | highs: NOT_SET |
| scheduling.json | 0.080717 s | 4.166417 s | 0.045820 s | cpu: UNKNOWN |
| production.json | 0.000530 s | 0.242119 s | 0.002047 s | All three repetitions OPTIMAL per engine |

CUDA: 30/30 measured runs optimal; CPU: 27/30, scheduling returns a feasible incumbent and UNKNOWN with a 0.267% global gap; HiGHS: 27/30, unsupported MIQP has NOT_SET and no solution. HiGHS therefore solves 9/9 supported models, and this does not show VANTAGE dominating it. Small CPU timing advantages depend on end-to-end measurement conventions; process wall timings and startup overhead are retained separately. CUDA is slower than our CPU backend on every selected small instance.

E226 medians: CPU 3.526 s, CUDA 9.371 s, HiGHS 0.0345 s. The recovery closes the recorded E226 failure, but a large speed gap remains. LP/QP OPTIMAL results underwent standalone original-space verification; MILP/MIQP verification independently checks incumbents, while tree optimality is solver telemetry rather than independently replayed proof.

## Stress-campaign design

The user explicitly requested larger cases after seeing small-model results. Added an isolated SCIP 10.0.2/PySCIPOpt 6.2.1 comparison adapter. It returns original-column-order primal solutions and node/gap/bound telemetry; it supplies no LP duals, so the SCIP comparison is used for MIP incumbent verification rather than claiming independently verified LP KKT. [SCIP installation documentation](https://pyscipopt.readthedocs.io/en/latest/install.html).

Public data: [MIPLIB rail507](https://miplib.zib.de/instance_details_rail507.html), 63,019 variables, 509 rows and 468,878 nonzeros according to the publisher; its continuous relaxation is explicitly labeled as a derived LP. The external HiGHS API is used only to remove integrality and write benchmark data, without solving during conversion. Failed download requests for rail2586 and neos-911880 remain in the dataset manifest. Netlib dfl001 and stocfor2 provide additional numerical/convergence challenges.

Synthetic stress: `examples/generate_sparse_stress.py` streams an MPS file with 1,000,000 variables, 500,000 equality rows and 1.5 million nonzeros. This is a planted cyclic LP, not real industrial data. For each block, `e_i + .25 e_(i+1) + .5 o_i = 1.25`, bounds `[0,2]`, `c_e=d_i+.25 d_(i-1)` and `c_o=.5 d_i+3`. The feasible point `(e,o)=(1,0)`, dual `y=-d` and reduced costs `(0,3)` prove expected objective 3,437,485 for seed 42. This supports scale/correctness checks but does not establish success on arbitrarily difficult million-variable industrial models.

## Dashboard evidence refresh

The dashboard now reads compact measured CSV summaries, merges additional SCIP/IPM baselines without discarding prior engines, labels actual repetitions and separates iteration/end-to-end speedup. The model library includes large railway and planted sparse cases. Capability/pending-work cards and checkpoint instructions reflect the current engine. Large-result API previews preserve complete downloads. API tests: 17 passed; desktop/mobile browser interactions passed. See [the dashboard guide](../dashboard/README.md).
