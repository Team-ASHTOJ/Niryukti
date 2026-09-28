# NIRYUKTI — complete project demo and speaking script

**Companion technical reference:** [all implemented features and mathematics](implemented_features_and_mathematics.md).
**Release snapshot:** 0.2.2, 28 September 2026.

This script is designed to show the complete project, not only the frontend.
The full route takes approximately **25–30 minutes**, including code/math and
planning demonstrations. A **15-minute edit** is provided below. Use the longer
version for an explanatory team/research video, then export a shorter submission
cut if required. Spoken text is a draft you can deliver naturally.

## 1. Recording preparation

### Tools and layout

- Record at 1080p or higher with readable browser/terminal/editor zoom.
- Keep three main windows: dashboard, code editor and terminal. Open public package
  pages in separate tabs. Open the generated report before recording.
- Use one narrator, or divide into platform, numerical engine and developer
  integration sections. Introduce the next speaker through the feature, not a long
  personal introduction.
- Turn off notifications. Hide accounts/tokens/private keys and unrelated windows.
- Build/install before recording; npm compiles C++ and is not a useful live wait.
- For the mathematical section, show the formulas in the companion document beside
  the source rather than scrolling through hundreds of unintroduced lines.
- Record chapter-sized takes. If a run is repeated or replaced with a saved run,
  identify it accurately. Edit idle time without changing outcomes/order.

### Prepare local evidence

From the repository root:

```bash
./scripts/build.sh
bash scripts/run_tests.sh build
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export NIRYUKTI_BINARY="$PWD/build/niryukti"
export NIRYUKTI_LIBRARY="$PWD/build/libniryukti_c.so"
mkdir -p results/full-video
NIRYUKTI_DEMO_DIR="$PWD/results/full-video/core" ./scripts/recording_demo.sh
python3 examples/refinery_replanning.py --output results/full-video/replanning
python3 examples/stable_refinery_plan.py --output results/full-video/stable-plan
./scripts/run_certificate_demo.sh
python3 dashboard/server.py --host 127.0.0.1 --port 8080
```

On non-Linux hosts the native-library suffix may differ. The NVIDIA demonstration
is intended for the validated Linux laptop. The dashboard command blocks while
serving; use another terminal for later commands.

Open `http://127.0.0.1:8080`, the generated HTML reports and
`results/full-video/core/refinery-cuda.json`. Keep the frozen 0.2.1 benchmark label
visible if using its numbers. Public installs can be prepared with:

```bash
python -m pip install --upgrade niryukti==0.2.2
npm install niryukti@0.2.2
# If your npm policy blocks install scripts:
node node_modules/niryukti/install.js
```

Do not confuse a CPU package install with enabling CUDA. Source GPU builds and
`NIRYUKTI_BINARY`/`NIRYUKTI_LIBRARY` select the GPU-enabled local engine.

## 2. Chapter map and feature coverage

| Chapter | Full-video time | What it covers | Feature reference |
|---|---|---|---|
| A | 0:00–1:00 | Problem, product and independence | Architecture, M01–M04 |
| B | 1:00–2:30 | Sparse model, inputs and validation | M01–M12 |
| C | 2:30–5:30 | Dashboard walkthrough and real solving | U01–U05, U10, B03 |
| D | 5:30–7:30 | CLI, advisor and explicit CUDA execution | A19–A21, G01–G12, U06 |
| E | 7:30–10:30 | LP mathematics, scaling and research variants | P01–P11, A01–A10 |
| F | 10:30–12:00 | QP, singular PSD and MIQP | M04/M11/M12, A11, I01/I02 |
| G | 12:00–13:30 | Simplex, barrier and concurrent portfolio | A12–A18 |
| H | 13:30–16:30 | Branch-and-cut, heuristics and conflicts | I01–I19 |
| I | 16:30–18:30 | Verification, certificates and tamper rejection | V01–V07 |
| J | 18:30–20:00 | Real checkpoints and cancellation | C01–C07 |
| K | 20:00–23:00 | Industrial replanning, repair and sensitivity | D01–D07 |
| L | 23:00–25:30 | Python, native C ABI, npm, HTTP and reports | U06–U12 |
| M | 25:30–27:00 | Evidence bundles, signatures, benchmarks and tests | D08/D09, B01–B04 |
| N | 27:00–28:00 | Scope, practical value and future work | Reference section 9 |

These times are editorial targets, not timers you must obey. Every inventory group
has a home in this walkthrough; deeper formulas are available in the companion.

## 3. Full narration and actions

### A. Opening: make the project understandable

**Show:** NIRYUKTI overview/logo, refinery planning example and a small architecture
view. Do not begin with a long benchmark table.

**Say:**

> “This is NIRYUKTI, an independently implemented mathematical optimization engine.
> It turns costs, capacities, quality limits and discrete decisions into a numerical
> optimization problem, then computes a plan and checks the result against the
> original model.
>
> We built the solver internals, not only a modeling interface. The prototype
> supports linear programming, supported convex quadratic programming, mixed-integer
> linear programming and convex mixed-integer quadratic programming. It has CPU and
> NVIDIA CUDA execution, a usable local dashboard, developer packages and a separate
> numerical verification layer.”

**Point out:** own core versus allowed numerical infrastructure. Open
`scripts/check_dependencies.py` briefly if the audience asks about wrappers.

> “Eigen and cuSPARSE provide numerical building blocks. HiGHS and SCIP are external
> comparison programs, not the algorithms behind a user solve.”

### B. Codebase and canonical model

**Show these files, in order:**

1. `include/vantage/vantage.hpp`: `Sparse`, `Model`, `Options`, `Verifier`, `Result`.
2. `src/model.cpp`: sparse construction, transpose, validation and PSD recognition.
3. `src/io.cpp`: input-format dispatch and validation.
4. `src/solver.cpp`: solve orchestration and validation boundaries.

**Say:**

> “All input formats become one sparse mathematical representation: a linear or
> convex quadratic objective, row intervals, variable bounds and variable types.
> Linear, quadratic and integer methods therefore share model handling rather than
> having separate incompatible parsers.
>
> Matrix construction aggregates duplicates and retains sparse storage. The core
> uses 64-bit offsets; the GPU can use a smaller index representation when safe.
> The model accepts equalities, inequalities, ranged rows and free variables.
> Invalid names, dimensions, bounds and nonfinite coefficients fail before kernels
> or numerical factorization are called.”

**Show:** `examples/toy.lp`, `examples/refinery.json`, one Netlib `.mps` and
`examples/toy.qplib`. Mention that QPLIB is a supported continuous subset and LP
text is linear-only. Do not imply every historical MPS/QPLIB extension is accepted.

**Optional command:**

```bash
./build/niryukti inspect examples/refinery.json
./build/niryukti convert examples/toy.lp results/full-video/toy.mps
./build/niryukti inspect results/full-video/toy.mps
```

### C. Frontend walkthrough: one useful workflow

**Click sequence and narration:**

| Show/click | What to say |
|---|---|
| Overview and workspace navigation | “The dashboard is a local operating workspace for the actual engine. These summary cards come from recorded results; they are not promotional sample numbers.” |
| Theme switch and a narrow/mobile viewport | “The interface has responsive navigation, accessible readable measurements and saved theme/motion preferences. Fonts and assets are local, so the presentation works offline.” |
| Model library | “We keep public instances and clearly labeled synthetic industrial examples together. The same solver handles different application formulations.” |
| Import | “A local MPS, LP or JSON file becomes a canonical model. Unsupported syntax is reported rather than silently ignored.” |
| Solve workspace; select refinery | “Here we choose the model, method, device, tolerance, time, iterations and threads. Automatic method selection is available; for this take I’ll demonstrate CUDA explicitly.” |
| Start solve; real log and convergence fields | “This is a real process. The log and objective/residual fields come from the solver. These curves are iteration evidence, not simulated progress.” |
| Auto Solver panel | “This records the actual backend, method, settings and advisor reason. A small model may correctly stay on CPU; using the GPU everywhere is not the goal.” |
| Result and Verification | “The candidate is reconstructed and checked in original units. We inspect feasibility, stationarity, gap and finite-value checks before discussing numerical optimality.” |
| View/Explain Certificate and downloads | “The certificate states what was verified. We can retain the full solution and independently replay the numerical checks.” |
| Run history | “Past runs retain settings, outcomes and downloadable evidence. A large preview is not the full transferable solution; the download contains complete data.” |
| Hardware and Diagnostics | “Hardware identity and raw logs are real. We do not invent utilization or memory measurements when they are unavailable.” |
| Docs | “The implementation scope and limits are visible inside the workspace and in the repository.” |

**Important visual distinction:** decorative refinery/branch/computation graphics
are illustrations. Do not point at them as actual search topology or GPU load.

**Benchmark/Arena shot:** choose a small model and available CPU/CUDA/HiGHS engines.
Let the Arena perform its actual sequential measurements; inspect status/objective
and time, then download the comparison. If a baseline is unavailable, say so.

> “This is a comparison, not a scripted win. We retain failed and limited outcomes.
> Kernel/iteration time, end-to-end solver time and full process wall time answer
> different questions.”

### D. CLI, automatic advisor and actual GPU path

**Run:**

```bash
./build/niryukti devices
./build/niryukti analyze examples/refinery.json --auto
./build/niryukti solve examples/refinery.json --device cuda --method pdhg \
  --gpu-presolve --gpu-monitor --cuda-graphs --iterations 1000000 \
  --time-limit 30 --json-out results/full-video/refinery-gpu.json
./build/niryukti verify examples/refinery.json results/full-video/refinery-gpu.json
```

**Say:**

> “The CLI exposes the same engine without the frontend. The advisor uses inspectable
> structure and GPU-memory guards to choose a compatible method and backend. It is
> a rule-based policy, not an invented runtime prediction.
>
> This explicit CUDA run performs sparse products and vector updates on the GPU.
> Device diagnostics and trial acceptance avoid full-vector transfers every
> iteration. Graph execution, index selection, mixed matrix precision and batched
> branching probes are optional paths, with independent final FP64 checks.”

**Show code:** `src/advisor.cpp`, then `src/cuda.cu` around sparse products,
`advance`, monitoring, resident restart, propagation and `cuda_compact_matrix`.

> “GPU bound proposals now become actual continuous reductions only after a
> derivation record is replayed. That record maps the tightened-box duals back into
> original constraint duals. Retained CSR entries are counted, scanned and compacted
> on CUDA. Presolve decisions/proof metadata and parts of control remain on CPU.”

If discussing mixed precision, use the actual mode name: FP32 **matrix storage**
with FP64 vectors/compute, not “the entire solver is FP32.” Persistent context/cache
reuse is bounded and invalidated on model/device changes.

### E. Explain the LP mathematics through code

**Show:** companion reference sections 3.1–3.4, alongside `src/cpu.cpp`,
`src/preprocess.cpp` and `src/solver.cpp`. Highlight a short update block at a time.

**Say:**

> “PDHG separates a primal decision update and a dual constraint update. The primal
> step follows the objective and transposed constraint gradient, then projects into
> variable bounds. The dual proximal step respects whether each row is a lower,
> upper or equality constraint. These operations reduce mostly to sparse A*x and
> A-transpose*y, which map naturally to parallel hardware.
>
> Step selection is not just a fixed magic number. We estimate a conservative
> matrix norm and check trial displacements against a coupling condition. CPU and
> CUDA now use the same acceptance inequality. Rejected trials do not modify
> accepted state. We average iterates, compare current and average candidates in
> original coordinates and restart when progress/epoch rules justify it.”

**Point at E and C in the reference**, explain `E >= 2|C|` without reciting every
exponent in the tuning rule.

> “Scaling balances rows and columns before solving and is reversed before the
> final checks. Ruiz is the base path; combined scaling adds geometric and
> Pock–Chambolle stages. Bound elimination and row reductions retain reconstruction
> metadata. We do not delete small coefficients just to make a case easier.”

**Research menu/code shot:** show `halpern`, `rhpdhg`, `r2hpdhg`, PID, power estimate
and polishing controls in source/CLI help. Optionally run one short experiment:

```bash
./build/niryukti solve examples/toy.lp --device cpu --method rhpdhg --no-adaptive
```

> “Anchored methods blend a fixed-point step with an anchor; the reflected experiment
> changes that operator. These are selectable research variants with guarded
> combinations, not a claim that we exactly reproduced every paper. PID controls
> primal/dual weighting from epoch displacements. Power iteration is an estimate,
> not a proof. Feasibility polishing solves targeted primal/dual auxiliary problems
> and only accepts their recombination after checking the original model.”

### F. Quadratic and singular-PSD support

**Show:** `examples/dispatch.json`, `examples/coupled_dispatch.json`,
`examples/integer_dispatch.json`; `src/model.cpp` PSD/AMD path and the smooth
Hessian product in `src/solver.cpp`/backends.

```bash
./build/niryukti solve examples/coupled_dispatch.json --method auto --device cpu
python3 examples/generate_singular_qp.py --help
```

Use the generator's actual displayed options if recording the 1,200-variable
weighted-star example; do not create a huge instance during the take.

**Say:**

> “Quadratic support is broader than a diagonal cost vector. The canonical Hessian
> can contain off-diagonal sparse coupling. Diagonal costs have an exact separable
> proximal update; general supported convex Hessians use smooth primal-dual splitting.
>
> Convexity is checked numerically before a convex algorithm is used. Sparse
> semidefinite elimination with AMD ordering recognizes additional tested singular
> structures without the dense fill of natural ordering. Uncertain curvature is
> rejected, not silently converted to a convex problem. Adding integer decisions
> gives convex MIQP, whose node bounds use a conservative affine minorant.”

Do not call this implementation rAPDHG/PDHCG or unrestricted large-PSD support.

### G. Multiple engines: simplex, barrier and portfolio

**Show code:** `src/simplex.cpp`, `src/barrier.cpp`, `src/portfolio.cpp`. Highlight
standardization/phase logic, basis solve/refactorization, predictor-corrector
Newton direction and winner verification/cancellation.

**Say:**

> “One algorithm is not ideal for every geometry. We also implement revised simplex
> with phase I and phase II, sparse basis solves and recovery, plus compatible-basis
> dual-simplex reoptimization. A bad or incompatible basis can fall back safely.
>
> Our barrier engine uses primal-dual predictor-corrector steps, positive slack and
> multipliers, and checked Newton systems. Its CUDA path currently uses host KKT
> assembly and GPU iterative linear solving; this is an experimental barrier,
> not a production direct sparse GPU factorization engine.
>
> The concurrent option races supported engines and accepts the first independently
> verified optimum. Automatic selection and recovery can use these engines without
> changing the input representation or verification contract.”

**Optional short commands:** `--method simplex`, `--method barrier`,
`--method concurrent --threads 3` on supported small examples. A successful example
is evidence of that example, not universal numerical robustness.

### H. Show that integer optimization is real

**Run:**

```bash
./build/niryukti solve examples/supply_chain.json --method auto --device cpu \
  --branching reliability --cuts --primal-heuristic all --node-limit 10000 \
  --json-out results/full-video/supply-chain.json
./build/niryukti verify examples/supply_chain.json results/full-video/supply-chain.json
```

**Show:** returned `mip` block, method/root relaxation selection and certificate
scope. Then `src/mip.cpp`, `src/cuts.cpp`, `src/research.cpp`.

**Say:**

> “The integer engine solves relaxations using our own continuous solver, branches
> on fractional decisions and keeps an incumbent plus a global bound. Queue policies
> can prioritize depth, best bound or estimates, but that does not change proof rules.
> Queued nodes retain bound changes rather than full root-bound copies.
>
> We do not prune from an approximate primal objective. Bounds come from conservative
> dual evaluation or inherited valid bounds. Reliability branching uses historical
> directional pseudocosts and bounded strong probes. On CUDA, related MILP probes can
> be batched through sparse matrix–matrix products. Probe scores choose the branch;
> they are not proof that a child can be discarded.”

**Cut tour:** show the cover/clique code, MIR separator, lattice separator and
pool fields. Use one simple cover inequality from the companion reference.

> “The cut layer contains guarded cover, clique, mixed-integer rounding and exact
> integer-lattice strengthening. Root-valid global cuts have aging and efficacy;
> node-local cuts stay local. This is a real controlled cut pipeline, not every
> commercial separator family.”

**Conflict tour:** highlight `learn_bound_conflict`, original-row replay and unit
propagation; optionally show a saved general-integer clause from the test.

> “An infeasible conjunction of bounds can teach a conflict. We replay its
> contradiction through original rows, shorten the explanation and propagate it in
> later nodes. General integer bounds now work as well as binary decisions. We do
> not learn from a timeout, and we do not claim a general implication-graph engine.”

**Heuristic tour:** show pump/RINS/local counters.

> “Rounding and repair, feasibility pump, RINS and local branching search for good
> incumbents. Each candidate is checked against the original model. Restricted
> neighborhood bounds cannot incorrectly prune the global problem.”

### I. Verification is an inspectable feature

**Show:** `src/verify.cpp`, `src/certificate.cpp`, Verification panel and
`verification_certificate` in result JSON.

**Say:**

> “The iteration code produces a candidate. A separate verifier recomputes original
> matrix activity, objective, bounds, integrality, stationarity, complementarity
> and available dual bounds. Scaling cannot hide a failure because checks return
> to original units. NaNs and overflow do not become optimal statuses.
>
> A certificate binds the candidate to a canonical model identity and recorded
> tolerances. We can replay its numerical checks without running optimization
> again. For an integer model, feasible incumbent evidence is distinct from a
> full search-tree optimality proof. The certificate says exactly which scope
> was established.”

**Run:**

```bash
./scripts/run_certificate_demo.sh
```

**Say while the three outcomes appear:**

> “Original evidence verifies. Changing the objective fails. Restoring the original
> evidence verifies again. This is a useful demonstration because the checker is
> allowed to disagree with a saved result.”

Discuss SHA-256 carefully: payload hashes detect edits but are not an author
signature. Farkas infeasibility needs a verified positive separation margin;
an arbitrary infeasible status is not a certified explanatory ray.

### J. Actual state continuation and interruption

**Run this known short barrier demonstration:**

```bash
./build/niryukti solve examples/coupled_dispatch.json --method barrier --device cpu \
  --no-presolve --iterations 2 --checkpoint-out results/full-video/newton-state.json \
  --json-out results/full-video/newton-limited.json
# Exit 2 is expected if the iteration limit is reached.
./build/niryukti solve examples/coupled_dispatch.json --method auto --device cpu \
  --no-presolve --iterations 10000 --resume results/full-video/newton-state.json \
  --json-out results/full-video/newton-resumed.json
./build/niryukti verify examples/coupled_dispatch.json results/full-video/newton-resumed.json
```

**Show:** state schema, iteration count and x/y/z/slack fields, not every vector.
Then point to first-order state, MIP frontier and simplex basis schemas in code.

**Say:**

> “This is actual state, not a fresh solve renamed resume. First-order checkpoints
> retain iterates, averages, anchors and controllers. Tree checkpoints retain the
> frontier, incumbent, bounds, pseudocosts, cuts and conflicts. Simplex retains
> phase/basis information, barrier retains Newton state, and the portfolio retains
> child states. Model and mathematical configuration must match.
>
> Bases are refactorized on resume and concurrent timing is not reproduced bit for
> bit. These are trusted local continuation files, not portable mathematical proofs.”

For interruption, stop a longer dashboard job or press Ctrl+C on a deliberately
long CPU solve. Show `INTERRUPTED` and retained candidate/status. Do not promise a
synchronous factorization can stop at an arbitrary instruction boundary.

### K. Industrial value beyond one static solve

**Show:** `results/full-video/replanning/` summary, model snapshots and verified
plans; `examples/refinery_replanning.py` and `examples/stable_refinery_plan.py`.

**Say:**

> “Industrial planning changes after deliveries or capacity change. Our coupled
> seven-day example conserves crude and product inventory and enforces a linear
> sulfur limit. These are synthetic coefficients, not actual MRPL operating data.
>
> Persistent sessions keep a parsed model and accept transactional named RHS,
> bound and objective changes. We retain compatible primal/dual and basis metadata
> for reoptimization. A failed update rolls back rather than corrupting the model.
> Warm-start timing is measured; we do not assume it always improves speed.”

**Delayed feed/outage:** point at exactly the changed row and resulting plan.

**Stable plan:** show operating cost, weighted change cost and changed feed values.

> “The cheapest revised plan may disrupt many earlier decisions. Stable replanning
> adds weighted absolute deviations from the reference plan. Hard locks preserve
> executed decisions. Operating cost and disruption penalty remain separate, and
> safety/quality constraints are not silently softened.”

**Repair:** show explicitly permitted demand rows, violation caps and relaxed-model
verification.

> “When the model is infeasible, diagnosis uses verified ray evidence where
> available. Repair changes only explicitly permitted constraint sides and reports
> the required violations. This is a proposal for a relaxed model, not a claim
> that the original infeasible problem became feasible without changing anything.”

**Sensitivity mini-demo:**

```python
from niryukti import Model, rhs_sensitivity
m = Model('sensitivity_demo')
m.add_var('x', ub=10)
m.add_constraint({'x': 1}, '>=', 2, name='demand')
m.set_objective({'x': 3})
response = rhs_sensitivity(m, 'demand', 0.1, device='cpu', method='auto')
print(response['status'], response['slopes'])
```

> “We solve a small RHS change on both sides and report measured slopes. At a kink
> the two sides can differ. This is a finite perturbation experiment, not exact
> differentiation or a certified allowable range.”

Mention power dispatch, production, supply-chain and scheduling models here if
they were not all individually solved in the earlier chapters.

### L. Public packages, native integration, API and reports

**Show public pages:**

- <https://pypi.org/project/niryukti/0.2.2/>
- <https://www.npmjs.com/package/niryukti/v/0.2.2>

**Say:**

> “The engine is reusable outside the dashboard. Both public packages use the
> NIRYUKTI name. Python wheels bundle a CPU executable and native library; npm
> builds the included C++ source. GPU execution requires the separately built
> CUDA engine. Documentation and licensing are included.”

**Python native-session example:**

```python
from niryukti import Model, NativeSession
m = Model('installed_demo')
m.add_var('x', ub=10)
m.add_constraint({'x': 1}, '>=', 2)
m.set_objective({'x': 3})
with NativeSession(m) as session:
    result = session.solve(device='cpu', method='auto')
    print(result['status'], result['objective'])
```

Show `include/niryukti/c_api.h`, `src/c_api.cpp`, `src/session.hpp` and
`python/vantage/native.py`.

> “The native Python session crosses a small C ABI using ctypes. It runs the same
> C++ engine in-process; the subprocess API is also available for isolation.
> This is not a zero-copy pybind11 binding. Handles and updates have explicit
> ownership and validation.”

**Node example, in a consumer directory with the package installed:**

```javascript
const {solve, renderReport} = require('niryukti');
const fs = require('node:fs');
(async () => {
  const result = await solve('/absolute/path/to/examples/refinery.json');
  console.log(result.status, result.objective);
  fs.writeFileSync('refinery-report.html', renderReport(result));
})();
```

> “Node provides an async API, cancellation through an AbortSignal, CLI forwarding
> and report rendering. It launches the native engine; it does not replace the
> numerical solver with JavaScript.”

**HTTP API shot:** start in a second terminal, generating a token without printing it:

```bash
export NIRYUKTI_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
PYTHONPATH=python python3 -m niryukti serve --host 127.0.0.1 --port 8090
```

Use the same environment in another client terminal or prepare the client before
recording; a new shell does not automatically inherit the token.

```bash
curl -s http://127.0.0.1:8090/v1/health \
  -H "Authorization: Bearer $NIRYUKTI_API_TOKEN"
```

Show `/v1/solve` and `/v1/report` using a prepared valid JSON request from
[API documentation](api.md), not a made-up endpoint. Do not show secrets in a
verbose curl trace.

> “This local API is authenticated, bounds request size/concurrency/time, and
> returns structured solver results. It is synchronous; client disconnect is
> not job cancellation. Dashboard jobs have a separate cancellation lifecycle.”

**Report:**

```bash
PYTHONPATH=python python3 -m niryukti report results/full-video/refinery-gpu.json \
  --output results/full-video/refinery-gpu.html
```

> “The report works offline and prints cleanly. It preserves status, objective,
> diagnostics, timing and MIP scope. Rendering supplied JSON is presentation;
> independent verification is a separate operation.”

### M. Evidence, signatures, benchmarks and tests

**Portable evidence:**

```bash
PYTHONPATH=python python3 -m niryukti.evidence create examples/refinery.json \
  results/full-video/refinery-evidence.zip --method auto
PYTHONPATH=python python3 -m niryukti.evidence verify \
  results/full-video/refinery-evidence.zip
```

For stronger integrity identification, add `--expected-sha256` using the digest
printed by creation. Creation refuses to overwrite an existing archive, so use
a fresh output name for a repeat take.

> “The bundle includes the model, candidate, checks and manifest. Import rechecks
> hashes and runs fresh numerical verification. A stored success report is not
> trusted just because it is inside an archive.”

**Optional signatures:** show signing/verification commands from
[trust documentation](trust_and_stable_planning.md) with placeholder key paths,
or a disposable demonstration key. Never open a private key on camera.

> “Detached signatures use standard OpenSSL and caller-managed trusted keys.
> Authenticity and mathematical validity are different: a signed wrong answer
> still fails the numerical checker.”

**Benchmark page/code:** `benchmark/run.py`, adapters, `freeze_campaign.py`,
report/profile generator and manifest. Point out optimizer imports only under
comparison infrastructure. Show the fixed campaign with the version and budget.

> “The frozen earlier-version campaign retains 108 attempts and their configuration.
> Large public and synthetic screening also retains timeouts and failures. Our
> own CPU/GPU algorithms and external solvers are compared transparently. We do
> not turn one favorable timing or a planted easy optimum into a universal claim.”

Use E226 as an honest example: all frozen runs verified at that campaign budget,
but HiGHS was much faster. Treat million-variable synthetic and rail relaxations
as distinct from full industrial integer solves. Performance profiles retain
unsolved cases in the denominator.

**Tests/code:** `tests/test_research.cpp`, `test_conflict_learning.py`,
`test_engine_checkpoints.py`, certificate tests and browser tests.

> “Tiny analytic optima, enumerated integer assignments, CPU/CUDA checks, corrupted
> certificates, actual interruption/resume and sanitizer tests attack different
> failure modes. The latest CUDA suite passed 17,461 research assertions, alongside
> regression checks. Package installation is also tested on actual built artifacts.”

Show Docker recipes and `scripts/validate_cuda_laptop.sh` briefly. State that CPU
container validation exists; standard NVIDIA-container/second-laptop validation
is still pending. Do not portray a Dockerfile as a passed GPU-container test.

### N. Closing with a precise claim

**Say:**

> “NIRYUKTI is a working independent optimization prototype with multiple numerical
> engines, a real NVIDIA path, controlled integer-search improvements and original-model
> verification. It is usable through a dashboard, CLI, packages and API, and its
> results can be retained, checked and reported.
>
> The remaining work is numerical and hardware hardening: broader device-resident
> control/presolve, conflict analysis, direct GPU barrier factorization, difficult
> PSD coverage and larger fresh performance campaigns. We are showing what runs,
> the evidence behind it and exactly where its current guarantees stop.”

## 4. The 15-minute edit

Use this if a shorter comprehensive overview is preferable:

| Time | Keep |
|---|---|
| 0:00–0:45 | Opening and architecture |
| 0:45–2:45 | Dashboard model/import/solve/result/verification/history; theme shot |
| 2:45–4:15 | CLI advisor, explicit CUDA, original-model verify |
| 4:15–6:15 | Canonical model, PDHG/scaling, QP and short method portfolio tour |
| 6:15–8:15 | Integer tree, bounds, cuts/conflicts and heuristic counters |
| 8:15–9:30 | Certificate tamper demo and integer evidence distinction |
| 9:30–10:30 | Barrier checkpoint/resume and cancellation mention |
| 10:30–12:00 | Replanning, stable policy, repair and sensitivity overview |
| 12:00–13:30 | PyPI/npm, native Python, API and report |
| 13:30–14:30 | Bundle/test/benchmark evidence |
| 14:30–15:00 | Precise scope and future work |

Put detailed mathematical derivations, full cut formulas, signature commands and
all package setup in the longer video or companion reference. Do not squeeze text
so tightly that the viewer cannot read the evidence.

## 5. Presenter questions and short answers

| Likely question | Answer and screen evidence |
|---|---|
| Is it a wrapper? | Own solver sources; dependency scanner; external adapters separated. |
| Is CUDA real? | `devices`, explicit CUDA result, CUDA kernels/SpMV and backend tests. |
| Why does auto select CPU? | Small models can be dominated by transfer/launch overhead; show actual advisor reason. |
| How is correctness checked? | Original-model native verifier, certificate replay and tamper rejection. |
| Does a green MIP status prove the entire tree? | No; inspect certificate scope and available gap/bound evidence. |
| Is the refinery data real? | No; synthetic, stated formulation assumptions and units. |
| Does warm start always win? | No; compatible metadata is reused and outcomes/timings must be measured. |
| Is the QP support general? | Sparse off-diagonal convex cases are supported, but numerical PSD recognition is guarded. |
| Are all advanced cuts implemented? | No; show exactly the cover/clique/MIR/lattice and bound-clause scope. |
| Can users install it? | Public 0.2.2 PyPI/npm pages and clean-consumer smoke checks. |
| Is the API production cloud infrastructure? | No; bounded authenticated local service and separate dashboard jobs. |
| Is the reported speedup current and fair? | Identify version, selected subset, repetitions, budgets and timing metric; retain failures. |

## Research-credit shot

Open [section 10 of the comprehensive reference](implemented_features_and_mathematics.md#10-research-papers-and-implementation-relationships). Show the PDHG/PDLP, Halpern, cuPDLPx, infeasibility, predictor-corrector, feasibility-pump/RINS, batched LP and performance-profile rows.

Say: “These papers informed the mathematics and engineering choices. We wrote our
selected implementations independently. The table distinguishes bounded variants
from reviewed future directions; it does not import another paper's performance
claims into our results.” Keep this shot near the relevant algorithm chapters,
or use it as a 45-second research-context insert.

## 6. Final take checklist

- Each major claim has a corresponding visible file, result or action.
- Show one actual CUDA solve and one independently verified original-model result.
- Show one integer case with honest certificate/gap scope.
- Show off-diagonal QP and distinguish it from diagonal-only optimization.
- Show real state continuation and a portable report/certificate.
- Mention additional planning/integration features even if their deeper takes are separate.
- Use current NIRYUKTI naming, readable zoom and accurate package versions.
- Keep synthetic labels, historical benchmark versions and unsupported limits visible.
- No secrets, fabricated utilization, hidden failed runs or unsupported performance promises.
