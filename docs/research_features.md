# Research integration record

This records the implemented portions from the supplied research brief. It does **not** represent completion of the full heterogeneous-solver roadmap or a reproduction of every cited solver. New solver features are opt-in. Production still uses independently written numerical code and no external optimization engine.

## Implemented and exposed

| Change | Entry point | Verification |
| --- | --- | --- |
| Correct CPU adaptive acceptance to match CUDA and documented energy inequality | Default PDHG | Deliberately unsafe scalar trial must be rejected; CUDA parity test runs when available |
| Stable direct displacement product for line search | CPU and CUDA PDHG | Existing LP/QP and enumerated MILP tests; GPU execution pending |
| Geometric mean → Ruiz → Pock–Chambolle scaling | `--scaling combined` | Primal/dual/objective transformation identities, diagonal QP postsolve, empty rows/columns, zero-pass identity |
| Fixed-operator Halpern LP experiment | `--method halpern --no-adaptive` | Independent scalar recurrence, anchor reset, bounded/ranged LP, warm start, limits, original-space KKT |
| Reliability pseudocost branching with bounded serial probes | `--branching reliability` | Exhaustively enumerated binary problems, conservative bounds, probe accounting, limited tree retention |
| Restarted Halpern and reflected Halpern CPU LP | `--method rhpdhg` / `r2hpdhg`, with `--no-adaptive` | Scalar reflected recurrence, fixed-point restart/reset, original-space KKT and iteration budgets |
| Safeguarded displacement PID weighting | `--primal-weight pid` | Feedback direction, saturation and zero-displacement handling |
| Power-iteration norm estimate | `--power-iterations 20` | Known spectra, empty matrix and extreme scaling; only adaptive backtracking may use estimate for steps |
| Primal/dual feasibility polishing | `--polishing` | Separate feasibility formulations, multiplier/reduced-cost cone signs, shared iteration budget, original-model verification |
| Binary minimal-cover and clique cuts | `--cuts` | Exhaustive assignment checks across 40 generated knapsacks; MILP original-space checks |
| L1 feasibility pump and RINS neighborhoods | `--primal-heuristic pump`, `rins`, or `all` | Independent projection optimum, exercised pump/RINS solves, shared RINS node budgets |
| Additional Farkas candidates | Default certificate checks | Original-space outward-rounded verification of iterate, alternate iterate and epoch-displacement candidates |
| Dolan–Moré performance profiles | `benchmark/profiles.py` and benchmark reports | Hand-computed ties, failures, missing cells, repeated runs and nonfinite times |
| Reject NaN integrality tolerance | All methods | CLI regression |
| Reproducible ablation flags | `benchmark/run.py` | Flags recorded in manifests; raw failures/limits retained |

C++ and Python expose `method`, `scaling`, `branching`, `primal_weight`, `power_iterations`, `polishing`, `cuts`, and `primal_heuristic`. Python uses `adaptive=False` for fixed-operator methods. CLI result options record selections. No experimental method is selected by default. Existing dashboard controls remain valid and continue using the default method.

```bash
cmake -S . -B build-research -DCMAKE_BUILD_TYPE=Release -DVANTAGE_CUDA=OFF
cmake --build build-research --parallel 4
bash scripts/run_tests.sh build-research

build-research/vantage solve datasets/e226.mps --scaling combined
build-research/vantage solve examples/toy.lp --method halpern --no-adaptive
build-research/vantage solve examples/supply_chain.json --branching reliability
```

```python
result = model.solve(binary="build-research/vantage", scaling="combined")
result = model.solve(binary="build-research/vantage", method="halpern", adaptive=False)
result = mip_model.solve(binary="build-research/vantage", branching="reliability")
```

## Additional options and precise scope

```bash
build-research/vantage solve examples/toy.lp --method rhpdhg --no-adaptive --primal-weight pid
build-research/vantage solve examples/toy.lp --method r2hpdhg --no-adaptive
build-research/vantage solve datasets/e226.mps --scaling combined --power-iterations 20 --polishing
build-research/vantage solve examples/scheduling.json --branching reliability --cuts --primal-heuristic all
python3 benchmark/profiles.py baseline=results/research-serial-baseline/runs.csv default=results/research-serial-default2/runs.csv --output results/research-profile
```

`rhpdhg` and `r2hpdhg` use the fixed-point metric of PDHG; sufficient decrease is 0.2, necessary decrease is 0.8 with a subsequent increase, and an artificial restart limits epoch length. These practical thresholds differ from some theoretical paper configurations. `r2hpdhg` uses reflection coefficient 1. Both require a conservative fixed step within each epoch and currently support CPU continuous LP only. They are distinct from the HPR-LP splitting, which is not implemented.

PID uses weighted epoch displacement error, gains P=0.3/I=0.01/D=0.1, bounded corrections, clipped integral and anti-windup. These are explicit implementation choices, not tuned upstream defaults. Power iteration reports an estimate, not a certified upper bound. Fixed methods retain the conservative norm bound; adaptive PDHG can use the estimate because its acceptance test remains mandatory.

Polishing starts near small relative gap, tries at doubling iteration checkpoints, and allocates at most one eighth of consumed iterations to each auxiliary solve within the shared continuous iteration/time budget. It retains a candidate only when original-model KKT improves. It never returns an auxiliary infeasibility status as the original result. It currently rejects QP requests.

Cuts are global root inequalities for positive integral binary knapsack rows, with coefficients/RHS at most 2^52. Validity arithmetic uses exact unsigned integers. Other rows are skipped. At most 64 inequalities are added; there is no general separator loop or cut aging. Added dual coordinates are removed before exporting an original-model incumbent.

The pump performs at most eight 2,000-iteration LP projections at the root when round-and-repair has not found an incumbent. General integer distances use nonnegative auxiliary variables. RINS fixes agreeing integer coordinates and searches at most 20 nodes with recursive neighborhood heuristics disabled. Sub-MIP nodes count toward the global node limit; their bounds never enter the main search's bound. Both respect remaining solve time and independently verify incumbents. As in the existing MILP solver, the iteration option applies per relaxation; aggregate MILP iteration counts include heuristic work.

Reports add `weight_updates`, `operator_norm_estimate`, `polishing_attempts`, `polishing_iterations`, `cuts_added`, `pump_rounds`, `rins_calls`, and `heuristic_nodes`. The dashboard still uses the original defaults; advanced settings are available through CLI/C++/Python.

## Source review and decisions

Primary sources were checked on 2026-09-27. The table distinguishes implemented portions from future work. Papers with accessible HTML informed algorithm review; other entries were assessed from project documentation or abstracts. This is a relevance and dependency assessment, not a claim that every proof, implementation, or performance number was independently reproduced.

| Reference from the brief | Useful idea and disposition |
| --- | --- |
| [Practical PDHG/PDLP (2021)](https://arxiv.org/html/2106.04756v2) | Adaptive line search and diagonal preconditioning. Corrected acceptance; added optional combined scaling. |
| [Updated PDLP](https://arxiv.org/html/2501.07018v2) | Implemented separate primal/dual feasibility subproblems with shared budgets and original-model KKT checks. This is a bounded LP polishing variant, not a reproduction of all PDLP features. |
| [Restarted Halpern PDHG](https://arxiv.org/html/2407.16144v2) | Implemented anchored and reflected CPU updates with optional fixed-point restart tests. General box/interval implementation uses practical restart thresholds; this is not a claim of reproducing every theorem or upstream configuration. |
| [cuPDLPx paper](https://arxiv.org/html/2507.14051v2) and [project](https://github.com/MIT-Lu-Lab/cuPDLPx) | Implemented geometric scaling, reflected Halpern experiments, safeguarded PID weighting and power estimates. Active-set boosting and GPU execution of these new methods remain pending. The paper's PID error uses weighted epoch displacements, not simply the residual ratio suggested in the brief. |
| [HPR-LP](https://arxiv.org/html/2408.12179v2) | Separate HPR splitting remains pending; do not relabel the Halpern PDHG experiment as HPR. |
| [2026 FOM comparison](https://www.global-sci.com/JCM/article/view/24701) | Abstract motivates controlled comparisons of splitting/restart combinations. Full comparative reproduction remains pending. |
| [PDHG infeasibility detection](https://arxiv.org/abs/2102.04592) | Added epoch-displacement and alternate-iterate dual-ray candidates, accepted only by the existing independent Farkas verifier. General primal recession certificates remain pending. |
| [PSLP](https://arxiv.org/html/2604.23951v1) | FOM-oriented presolve is relevant. New reductions require reversible primal/dual postsolve records before use. |
| [GPU presolve/cuPSLP](https://arxiv.org/html/2609.16182v1) | GPU reductions remain pending; require validated CPU reductions and device-side structural compaction. |
| [rAPDHG/PDQP](https://arxiv.org/html/2311.07710v3) and [PDQP.jl](https://github.com/jinwen-yang/PDQP.jl) | Sparse PSD Q support remains pending. Requires model, parser, scaling, gradient, verifier and lower-bound changes together. |
| [GPU second-order solvers](https://arxiv.org/html/2508.16094v1) | Barrier/IPM remains pending; needs sparse factorization, regularization, refinement and failure handling. |
| [Condensed IPM](https://arxiv.org/html/2405.14236v2) | Condensed KKT formulation remains pending; conditioning and fill-in must be measured on the intended model classes. |
| [Batched LP for MIP](https://arxiv.org/html/2601.21990v1) | Implemented bounded serial strong probes as groundwork. Batched GPU relaxation storage/kernels and OBBT remain pending. |
| [cuOpt MIP settings](https://docs.nvidia.com/cuopt/user-guide/latest/mip-settings.html) | Implemented opt-in directional reliability branching; no cuOpt dependency. |
| [SCIP separators](https://scipopt.org/doc/html/group__SEPARATORS.php), [conflict analysis](https://www.scipopt.org/doc-8.0.3/html/CONF.php), [workshop](https://scipopt.org/workshop2018/SCIP-Intro.pdf) | Some requested pages were unavailable. The [official knapsack implementation documentation](https://www.scipopt.org/doc-6.0.0/html/cons__knapsack_8c.php) informed minimal-cover/clique cuts. MIR/GMI, lifting and conflict analysis remain pending. |
| [Feasibility pump](https://publications.polymtl.ca/24161/) | Added a bounded round/project L1 pump with deterministic cycle perturbations, informed by the [author paper](https://www.dei.unipd.it/~fisch/papers/feasibility_pump.pdf). It supplements existing round-and-repair; advanced pump variants remain pending. |
| [RINS](https://link.springer.com/article/10.1007/s10107-004-0518-7) | Implemented bounded sub-MIP neighborhoods fixing integer coordinates where the relaxation and incumbent agree. Recursive RINS is disabled, the global node budget is shared, and neighborhood bounds never prune the main tree. |
| [Parallel revised dual simplex](https://webhomes.maths.ed.ac.uk/hall/HuHa13/) | Revised simplex remains pending; needs basis state, stable updates, pricing and numerical recovery before parallelization. |
| [HiGHS](https://highs.dev/) and [cuOpt portfolio settings](https://docs.nvidia.com/cuopt/user-guide/latest/convex-settings.html) | Multiple methods motivate explicit method selection. A concurrent portfolio requires multiple validated production engines; not implemented. |
| [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) | Capture remains pending; must preserve adaptive accept/reject state and interruption boundaries. |
| [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) | Existing sparse products retained. Index-width changes, mixed precision and new SpMV APIs require toolkit checks and hardware validation. |
| [Stream-ordered allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html) | Persistent workspace/pools remain pending; ownership must span repeated relaxations safely. |
| [cuDSS](https://docs.nvidia.com/cuda/cudss/) | Possible factorization infrastructure for future IPM, not an implemented barrier engine. |
| [hipSPARSE](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/index.html) | HIP backend remains pending; requires AMD hardware coverage and a backend abstraction beyond conditional CUDA calls. |
| [cuOpt convex features](https://docs.nvidia.com/cuopt/user-guide/latest/convex-features.html) | Crossover/multiple devices remain pending; require basis recovery or partitioning infrastructure. |
| [Dolan–Moré profiles](https://link.springer.com/article/10.1007/s101070100263) | Implemented standalone SVG/HTML/JSON performance profiles. Failed/missing cells remain at infinite ratio; every recorded repetition must succeed before its median time is eligible. |
| [Netlib](https://www.netlib.org/lp/data/), [MIPLIB](https://miplib.zib.de/), [QPLIB](https://qplib.zib.de/) | Used existing checked-in Netlib subset. Broad benchmark acquisition remains pending; QPLIB includes classes beyond current diagonal convex QP support. |
| [Maros–Mészáros benchmark](https://github.com/qpsolvers/maros_meszaros_qpbenchmark), [QPBenchmark](https://github.com/qpsolvers/qpbenchmark) | Comparison infrastructure references only. General-Q benchmark integration follows sparse-Q support. |

## Validation and limits

Release and AddressSanitizer/UndefinedBehaviorSanitizer CPU builds pass core, research, CLI/Python and dependency-policy checks. Seven dashboard HTTP tests pass with localhost binding enabled. Research tests are registered with CTest, so the existing CI release/debug sanitizer workflow runs them automatically. AppleClang on this machine did not find OpenMP; parallel OpenMP execution was not tested. CUDA was neither compiled nor executed here. The scalar CPU/CUDA acceptance test is conditional on an available CUDA backend and device.

The first acceptance-only fix exposed a real backtracking stall in the pre-existing enumerated MILP suite. Direct `A*dx` evaluation resolved it without weakening acceptance. CPU activity is refreshed at monitoring checkpoints. Original-model KKT verification, conservative MILP pruning, status semantics, parsers, and persisted model fingerprints continue to govern results.

The original binary was retained under `results/research-baseline/vantage`. Serial comparison artifacts are under `results/research-serial-{baseline,default,combined,halpern,reliability}/`; each has a manifest, per-run raw outputs and HTML report. These local results are ignored by git. A compact measured summary is in `research_validation.md`.

Do not promote experiments to defaults based on this small suite. GPU extensions, large public benchmark coverage, the full reversible presolver, sparse-Q support, barrier, simplex, MIR/GMI cuts, conflict analysis, HIP and the concurrent portfolio remain unfinished.
