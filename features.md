# New features: competitive gap closure for SIH26119

This document records the features added to NIRYUKTI after comparing it with the other
SIH26119 prototypes (Kavach-Opt, OPTIMACORE, Sovereign Solver, Trust the Verifier, ApexOPT,
DN and SANKHYA-OPT). It covers what each feature does, how to run it, the mathematics, how
results are verified, measured evidence and remaining limits.

All new code is original C++20 in the existing engine. It uses no external solver, and
every new path reuses NIRYUKTI's own LP engines and its independent verifier.

---

## 1. Gap analysis

| Capability seen in competing prototypes | Who has it | NIRYUKTI before | Action |
|---|---|---|---|
| Primal/dual simplex, IPM, PDHG | Metallica, ApexOPT, Khayali Pulao | Present (simplex, dual simplex, barrier, PDHG variants) | None |
| Concurrent engine race | Khayali Pulao | Present (`--method concurrent`) | None |
| Ruiz / Pock–Chambolle scaling | Khayali Pulao | Present (`--scaling ruiz\|combined`) | None |
| Farkas infeasibility certificates | Khayali Pulao | Present (`BOX_ROW_FARKAS`) | None |
| Warm restart after price/capacity changes | Metallica | Present (`--warm-start`, sessions) | None |
| Build check that bans external solvers | Metallica | Present (`scripts/check_dependencies.py`) | None |
| **Shadow prices + sensitivity *ranges*** | Metallica, ApexOPT | Only finite-difference RHS slopes in Python (`rhs_sensitivity`, which says it is "not an allowable RHS range") | **Added §2** |
| **Plain-language diagnosis of conflicting constraints** | Metallica | Farkas support only (`diagnose_infeasibility`, which says it is "not a minimal IIS") | **Added §3** |
| **McCormick relaxation + spatial B&B for bilinear pooling** | Convictionist | Not present (no bilinear support) | **Added §4** |
| **Three-state trust badge (Certified / Bounded / Local)** | Convictionist | Not present | **Added §4 and §5** |
| **Adaptive strategy layer learned from historical solve logs** | Metallica | Rule-based advisor only ("not a learned runtime prediction") | **Added §6** |
| **Mixed-precision iterative refinement** | SANKHYA-OPT | GPU `--matrix-precision mixed` for PDHG only | **Added for basis solves (§2.4)** |
| **Tensor-core style FP32/FP64 mixed-precision Newton solves** | SANKHYA-OPT | Not present for barrier | **Added §7** |
| **Hypergraph structure detection + Dantzig–Wolfe** | Khayali Pulao | Incidence components reported only, "no automatic decomposition" | **Added §8** |
| **Differentiable optimisation layer (OptNet-style)** | SANKHYA-OPT | Not present | **Added §9** |
| **GNN-guided branching** | SANKHYA-OPT | Rule-based fractional/reliability only | **Added §10** |
| **Plain-English model entry (SLM Studio)** | SANKHYA-OPT | Not present | **Added §11** (deterministic grammar, not an LLM) |

---

## 2. Basis-exact LP sensitivity analysis (shadow prices, reduced costs, ranging)

### Purpose
A refinery planner needs to know what one more unit of CDU capacity is worth, how far a
crude price can move before the plan changes, and over what range those answers hold.
Metallica and ApexOPT advertise shadow prices and sensitivity ranges. NIRYUKTI previously
could only re-solve with a perturbed RHS. It now computes the classical basis-exact ranges
in a single analysis.

### Usage
```bash
niryukti sensitivity examples/refinery.json             # solves, then analyses
niryukti sensitivity model.mps saved_result.json        # analyses an existing solution
niryukti sensitivity examples/supply_chain.json         # MILP: integers fixed at incumbent
```
```python
import niryukti
report = niryukti.sensitivity("examples/production.json")
for line in report["findings"]:
    print(line)
```
Exit code `0` means an optimal, dual-feasible basis was found (`OPTIMAL_BASIS`); `2` means a partial or unsupported result.

### Mathematics
The LP is written in bounded form with one logical (slack) variable per row:

  min cᵀx  s.t.  Ax − s = 0,  l ≤ x ≤ u,  r_l ≤ s ≤ r_u.

A basis B is m columns of [A −I]. With π = B⁻ᵀc_B:

* **Shadow price** of row i = π_i, which is ∂z/∂(active bound of row i). It is reported in the
  original objective sense (sign flipped for maximisation).
* **Reduced cost** d_k = c_k − πᵀa_k (slack column: d = π_i).
* **RHS ranging**: if row i's slack is nonbasic at bound b, moving b by δ moves the basic
  variables by δ·w with B w = e_i. The allowable interval is the largest δ-interval that keeps
  every basic variable within its bounds. The objective changes linearly at rate π_i
  throughout that interval.
* **Cost ranging, nonbasic j**: at lower bound the coefficient may fall by d_j; at upper
  bound it may rise by −d_j.
* **Cost ranging, basic j in position r**: with ρ = B⁻ᵀe_r and α_k = ρᵀa_k, changing c_j by
  δ changes every nonbasic reduced cost to d_k − δα_k. The range is the δ-interval that
  keeps each d_k sign-correct for its bound status.

References: Dantzig (1963); Chvátal, *Linear Programming* (1983) ch. 10; Bertsimas &
Tsitsiklis (1997) §5.1–5.2.

### Algorithm
1. Solve with the revised simplex, or accept a supplied solution.
2. **Basis reconstruction**: variables strictly between their bounds must be basic.
   At-bound columns are ranked by the solver's reduced-cost magnitude (complementary
   slackness) and added through greedy rank-revealing Gaussian elimination until B is
   nonsingular. A non-vertex point is rejected with `NON_VERTEX`.
3. **Degenerate basis repair**: on a degenerate optimum, the first nonsingular basis can be
   dual-infeasible. A bounded primal simplex with Bland's rule pivots until every reduced
   cost has the correct sign. At an optimal degenerate vertex the steps are zero, so the plan
   is unchanged and only the basis moves. Pivots are counted in `basis.degenerate_pivots`.
4. Compute π, reduced costs, both ranging families, and ranked plain-language findings.
5. **Independent verification**: (x, y = −π) goes to the existing original-space verifier,
   which reports KKT error, residuals and the outward-rounded safe dual bound.

### Mixed-precision iterative refinement (2.4)
Basis systems use a new dense LU (`src/dense_lu.hpp`). B is factorised in **FP32**. Each solve
computes FP64 residuals r = b − Bx and applies corrections through the FP32 factors
(Wilkinson/Moler iterative refinement; Carson & Higham, SISC 2018) until the normwise
residual reaches about 1e-15. If the FP32 factor is singular or refinement stalls, the code
falls back to an FP64 factorisation. This is the CPU analogue of SANKHYA-OPT's "FP32
projections with FP64 residual correction". The report records `solves`,
`refinement_steps`, `fp64_fallbacks` and `worst_relative_residual`.

### Output (abridged, `examples/production.json`)
```json
"constraints": [{"name": "balance_1", "status": "binding_equality", "shadow_price": 5.8,
                 "upper_bound_range": [5.0, 75.0]}, ...],
"variables":   [{"name": "production_1", "status": "basic", "reduced_cost": 0.0,
                 "objective_coefficient_range": [null, 5.8]}, ...],
"findings": ["Constraint 'balance_2' is fixed at 70. Each unit increase of that limit changes
              the objective by 6 (deterioration), valid while the limit stays within [35, 110]."]
```
Every number in this example matches a hand derivation (cost 517; period-1 demand priced at
6 − 0.2 = 5.8; capacity-1 worth 1.8 per unit).

### Evidence
Results were checked by finite differences: perturb a row bound or cost coefficient inside
the reported range, re-solve, and compare with the predicted change.

| Model | Rows | Binding rows | Perturbation checks | Mismatches | KKT error | Time |
|---|---|---|---|---|---|---|
| production (example) | 4 | 3 | 11 | 0 | 7e-15 | <1 ms |
| refinery (example) | 36 | 16 | 36 | 0 | 8e-14 | 1 ms |
| afiro (Netlib) | 27 | 22 | 24 | 0 | 1e-12 | 1 ms |
| adlittle (Netlib) | 56 | 46 | 29 | 0 | 4e-11 | 5 ms |
| israel (Netlib) | 174 | 71 | 34 | 0 | 2e-8 | 57 ms |
| e226 (Netlib) | 223 | 140 | 27 | 0 | 8e-12 | 231 ms |

On e226, 8 additional reference re-solves did not reach optimality (`ITERATION_LIMIT`), so
they were skipped rather than counted as checks. Across all runs, mixed-precision
refinement needed **0 FP64 fallbacks**; worst relative residual was ≤ 8e-16. Degenerate repair
is covered by a dedicated test where the index-ordered basis is dual-infeasible and one Bland
pivot fixes it.

### Limits
* Linear objectives only. QP returns `UNSUPPORTED`. MILPs are analysed as the
  fixed-integer LP; the report says so.
* Dense basis algebra is limited to 2,000 rows. Basic-variable cost ranging is skipped above
  1,500 rows. Degenerate repair runs up to 600 rows.
* Ranges are for one parameter change at a time and are specific to the basis found. A
  degenerate optimum can have other valid bases with different ranges, and the report says this.

---

## 3. Irreducible Infeasible Subsystem (IIS) with plain-language repair

### Purpose
When a scenario is infeasible, for example a demand surge beyond CDU capacity, planners need
to know which requirements conflict and the smallest change that fixes the plan. Metallica
offers "plain-language diagnosis of conflicting constraints". NIRYUKTI previously returned
the Farkas support, which is not minimal.

### Usage
```bash
niryukti iis examples/infeasible_blend.json
```
```python
niryukti.find_iis("examples/infeasible_blend.json")["explanation"]
```

### Algorithm
1. **Seed** from the support of a verified Farkas ray of the full model: rows with nonzero
   multipliers plus the bounds of the variables they touch.
2. **Deletion filter** (Chinneck & Dravnieks, 1991): try removing each row and then each
   variable bound. A member stays removed only if the remaining subsystem is still
   **certified** infeasible. The result is irreducible: dropping any member yields a
   verified-feasible system.
3. **Repair proposals**: for each IIS member, an elastic LP relaxes only that member against
   the **complete** model and minimises the violation. It reports the exact new limit, or that
   other conflicts remain.

### Verification design
Each oracle verdict must be proven:
* *Infeasible* requires a Farkas ray that passes an outward-rounded margin check against the
  true subsystem.
* *Feasible* requires a primal point that passes the verifier.
* Otherwise the verdict is *unknown*. The result is then reported as
  `INFEASIBLE_SUBSYSTEM` with `irreducible: false`, never as an IIS.

Two new pieces make this work when bound removal leaves free columns:
* **Big-box surrogate**: LPs are solved with ±10⁶·scale boxes in place of removed bounds, but
  every verdict is checked against the true unbounded subsystem. This stays rigorous because
  surrogate-feasible points are truly feasible, and rays must verify with the infinite bounds.
* **Exact-sign Farkas margin** (`src/farkas.hpp`): interval rounding cannot prove that a
  reduced cost such as −1 + 1 is exactly zero, so the existing verifier rejects any ray touching a
  free column. The new margin computes the exact sign of Σ a_ij y_i using error-free
  transformations: TwoProduct via `fma` and Shewchuk's Grow-Expansion. An exactly cancelling
  free column contributes zero, while all other arithmetic stays outward-rounded. Noisy rays
  are also snapped to dyadic grids and kept only if the snapped ray verifies.
* Row-activity infeasibility found by presolve (no ray returned) is certified by testing
  single-row rays ±e_i.

### Example output (`examples/infeasible_blend.json`)
```
IIS_FOUND
  fuel_demand: light_crude + heavy_crude >= 120
  sulfur_spec: -0.5*light_crude + 2*heavy_crude <= 0
  Bound: light_crude <= 50
Lower the minimum of 'fuel_demand' from 120 to 62.5 (change 57.5) to make the whole model feasible.
Raise the maximum of 'sulfur_spec' from 0 to 115 (change 115) to make the whole model feasible.
Raise the upper bound of 'light_crude' from 50 to 96 (change 46) to make the whole model feasible.
```
The synthetic refinery model with two demands raised to 180 gives an IIS of {demand_1_0,
demand_1_1, cdu_1}. It is short by exactly 60 units, with a repair for each member. That run
took 22 oracle solves in 3 ms.

### Limits
* Integrality is relaxed (LP-relaxation IIS), and the report says so. If the relaxation is
  feasible, the tool reports that integer infeasibility is not diagnosed.
* The deletion filter performs O(rows + bounds) LP solves, which suits model-sized
  diagnosis, not very large instances. The `--time-limit` setting stops it early and the
  report then says it is not irreducible.

---

## 4. Global optimisation of bilinear pooling problems (McCormick + spatial B&B)

### Purpose
Refinery blending tracks qualities (sulfur %) through pools, so constraints contain
**quality × volume** products. These make the problem nonconvex, and local solvers stall on
plateaus. In the classic Haverly problem, a local method can return a profit of 100 when 400
is attainable. Convictionist presented McCormick relaxations, spatial splitting and an
auditable badge. NIRYUKTI previously had no bilinear support.

### Input format
Standard NIRYUKTI JSON with an optional `bilinear` list of `[var, var, coefficient]` on the
objective and on any constraint. The objective's `linear` may be an array or a name→coefficient
map. See [examples/haverly_pooling.json](examples/haverly_pooling.json).

### Usage
```bash
niryukti global examples/haverly_pooling.json [--node-limit N] [--time-limit S] [--mip-gap 1e-4]
```
```python
niryukti.solve_global("examples/haverly_pooling.json")["badge"]
```

### Mathematics
Each product w = x_a·x_b on the node box [l, u] is replaced by its **McCormick envelope**
(McCormick 1976), which is the convex hull of the product over the box:
```
w ≥ l_b x_a + l_a x_b − l_a l_b        w ≤ u_b x_a + l_a x_b − l_a u_b
w ≥ u_b x_a + u_a x_b − u_a u_b        w ≤ l_b x_a + u_a x_b − u_a l_b
```
Square terms (a = b) use the same formulas, giving tangent and secant cuts. The LP over these
envelopes is a valid relaxation at each node.

* **Lower bound** (minimisation form): the verifier's outward-rounded Lagrangian bound of the
  node LP. This is a rigorous floating-point bound, not the LP's reported objective. Envelope
  right-hand sides are also loosened outward by 1e-12 relative to absorb coefficient rounding.
* **Upper bound**: an **alternating LP heuristic**. A greedy vertex cover of the product graph
  is fixed at relaxation values, which makes every product linear, and the remaining LP is
  solved. Then the complement is fixed and the LP solved again, for up to 6 rounds. Each point
  is accepted only after exact long-double re-evaluation of every linear and bilinear row.
* **Branching**: pick the product with the largest weighted envelope violation |w − x_a x_b|.
  Split whichever of its variables has the larger relative width, at the relaxation value
  clamped to the central 80% of the interval. This guarantees width reduction.
* **Search**: best-first priority queue. Nodes are pruned by bound or by certified LP
  infeasibility; a relaxation point that satisfies every product closes its node.
* **Global bound** = min over open, unresolved and bound-pruned nodes; infeasible regions
  contribute nothing.

References: McCormick (1976); Al-Khayyal & Falk (1983); Tawarmalani & Sahinidis (2002);
Haverly (1978).

### Assurance badge
| State | Colour | Meaning |
|---|---|---|
| `CERTIFIED_GLOBAL` | green | The rigorous bound meets the best feasible plan within tolerance |
| `BOUNDED` | amber | Feasible plan plus a proven ceiling; `unverified_gap` gives the remaining gap in objective units (e.g. dollars) |
| `LOCAL_ONLY` | red | Feasible plan with no valid global bound |
| `CERTIFIED_INFEASIBLE` | green | Every region pruned by a verified Farkas certificate |

### Evidence
| Instance | Literature optimum | NIRYUKTI | Badge | Nodes | Time |
|---|---|---|---|---|---|
| Haverly 1 (example file) | 400 | 400.0000 | CERTIFIED_GLOBAL | 3 | 2 ms |
| Haverly 1, wide bounds | 400 | 400.0000 | CERTIFIED_GLOBAL | 7 | 2 ms |
| Haverly 2 (X demand 600) | 600 | 600.0000 | CERTIFIED_GLOBAL | 7 | 2 ms |
| Haverly 3 (crude B at $13) | 750 | 750.0000 | CERTIFIED_GLOBAL | 11 | 3 ms |
| Haverly 3, `--node-limit 1` | 750 | 708.82 | BOUNDED (ceiling 875, gap 166.18) | 1 | — |
| Haverly 1 + infeasible sales floor | — | — | CERTIFIED_INFEASIBLE | — | — |

The root McCormick bound of 500 on Haverly 1 matches the published value. The `improving_solutions`
trace shows the search escaping intermediate local plans (for example 387.46 → 400 and
710.01 → 750).

### Limits
* Continuous variables only. Every variable in a product needs finite bounds; the error
  message names the variable that lacks them.
* No bound tightening (FBBT/OBBT) yet, so larger pooling instances will need more nodes than
  commercial global solvers.
* Higher-order polynomials and general nonlinear functions are not supported.

---

## 5. Assurance badge on every `solve`

`niryukti solve` now adds an `assurance` object derived only from independently verified
quantities:

* `CERTIFIED_OPTIMAL` (green): LP/QP, verified KKT ≤ tolerance, and the verified dual bound
  meets the objective within tolerance.
* `BOUNDED` (amber): verified feasible plan plus a finite bound. For MILP the bound is the
  tree's best bound; the badge states that the tree is not replayed, consistent with the
  project's existing evidence rules.
* `FEASIBLE_ONLY` / `UNVERIFIED` (red): no valid bound, or no verified feasible plan.
* `CERTIFIED_INFEASIBLE` (green): verified Farkas certificate.

Each badge also carries `objective`, `lower_bound`/`upper_bound` (original sense) and
`unverified_gap`.

---

## 6. Learned strategy layer from solve history

### Purpose
Metallica's adaptive layer "maintains historical solve logs and automatically selects the
optimal algorithm". NIRYUKTI's advisor was rule-based only. It now also learns from local,
verified outcomes.

### Usage
```bash
niryukti solve model.mps --history runs.jsonl                    # append a record
niryukti recommend model.mps --history runs.jsonl                # ranked methods
niryukti solve model.mps --method learned --history runs.jsonl   # use the recommendation
```
```python
niryukti.recommend("model.mps", "runs.jsonl")
```

### Method
* Each record is one JSON line with 7 structural features: log rows, log columns, log
  nonzeros, integer fraction, quadratic flag, log coefficient range, and log mean row degree.
  It also stores the method family, device, status, a **verified success** flag
  (OPTIMAL plus independent feasibility checks), seconds, iterations and nodes.
* A recommendation uses k-nearest-neighbour algorithm selection (Rice 1976; Kotthoff 2016)
  over records within distance 1.5 in feature space. An exact model fingerprint counts as
  distance 0. Each method@device group is scored as
  `weighted mean log10(time | success) + 3·(1 − weighted success rate)`, with weight
  1/(0.1 + distance), so one failure costs three decades of runtime.
* `--method learned` uses the top method only with at least 3 similar records and at least
  2 runs and 1 verified success for that method. Otherwise it falls back to the structural
  advisor and records why in `selection.learned.reason`.

### Limits
Advisory only, with no runtime guarantee. History is local and append-only; malformed lines
are skipped.

---

## 7. Mixed-precision iterative refinement for barrier Newton systems

### Purpose
SANKHYA-OPT advertises FP32 work on Tensor Cores with FP64 residual correction. NIRYUKTI now
applies the same scheme to its predictor-corrector barrier. The expensive KKT factor (CPU) or
KKT operator (GPU) runs in FP32, and accuracy is restored in FP64.

### Usage
```bash
niryukti solve examples/refinery.json --method barrier --newton-precision mixed
niryukti solve model.mps --method barrier --device cuda --newton-precision mixed
```
`solve` output gains a `mixed_precision` block with `refined_solves`, `refinement_steps`,
`fp64_fallbacks` and `worst_relative_residual`.

### Method (Wilkinson; Moler 1967; Carson & Higham, SISC 2018)
1. **CPU:** factor K in FP32 (`Eigen::SparseLU<SparseMatrix<float>>`). Before converting, the
   code checks that no coefficient overflows or underflows to zero; if one would, it uses FP64 instead.
2. **GPU:** the existing BiCGSTAB runs with FP32 matrix values and FP64 vectors and accumulation.
   This is the same cuSPARSE configuration that the validated `--matrix-precision mixed` PDHG uses.
3. **Refinement:** x₀ = solve₃₂(b). Then repeat: r = b − K x (FP64), x += solve₃₂(r), until
   ‖r‖∞ ≤ 1e-12 (1 + ‖b‖∞).
4. **Safety:** if the residual stops contracting (FP32 conditioning limit), the FP64 factor or
   FP64 GPU solve takes over. The barrier's existing residual check still applies afterwards.

### Evidence (CPU, Apple M-series, release build)
| Model | FP64 objective | Mixed objective | Iterations | Refinement steps | FP64 fallbacks | Worst residual |
|---|---|---|---|---|---|---|
| coupled_dispatch (QP) | 2833.30986 | 2833.30986 | 5 / 5 | 6 | 0 | 9.9e-13 |
| dispatch (QP) | 2782.88462 | 2782.88462 | 5 / 5 | 8 | 0 | 6.1e-14 |
| refinery (LP) | 40729.8132 | 40729.8132 | 10 / 10 | 30 | 0 | 3.4e-13 |
| afiro (Netlib) | −464.753143 | −464.753143 | 12 / 12 | 37 | 0 | 6.2e-13 |

On adlittle and israel the barrier also fails in plain FP64; this is a limit of the
existing barrier, not of the refinement.

### Limits
The GPU change is small: an FP32-operator flag and a tolerance argument for
`cuda_linear_solve`. It reuses a cuSPARSE configuration already validated by mixed PDHG, but
**it has not been run on a GPU in this session** (no NVIDIA device here). Run
`./scripts/validate_cuda_laptop.sh` and the barrier mixed tests on a CUDA machine before
claiming GPU results. Memory and speed benefits on CPU are not claimed; this demonstrates
correctness.

---

## 8. Structure detection and Dantzig–Wolfe decomposition

### Purpose
Multi-site refinery planning is block-angular: each refinery's balances are independent,
while shared jetties and national demand link them. Khayali Pulao showed hypergraph detection
plus Dantzig–Wolfe. NIRYUKTI previously reported incidence components only.

### Usage
```bash
niryukti decompose examples/multi_refinery.json                 # detect + solve
niryukti decompose model.json --detect-only                     # structure only
niryukti decompose model.json --linking 'jetty_*,demand_*'      # user-specified border
python3 examples/generate_multi_refinery.py 40 /tmp/mr40.json   # larger synthetic case
```

### Structure detection
Two detectors run, and the configuration with the better score wins. The score is
log₂(blocks) × balance − 4 × (border rows / rows).
* **Dense-row border search** removes the densest rows one at a time and finds the connected
  components of the remaining variable–constraint graph.
* **Row-family border search** treats indexed constraint families (`jetty_3` → `jetty`) as
  candidate borders. It seeds with the best *pair* of families, because shared supply and
  shared demand often only split the model together, then extends greedily.

This is a heuristic stand-in for hypergraph partitioning (Ferris & Horn 1998; Bergner et al.
2015). On the 4-refinery example it finds exactly the 3 jetty and 2 demand rows.

### Algorithm (Dantzig & Wolfe 1960)
* **Master:** linking rows, one convexity row per block, columns λ for block extreme points,
  and master-only variables.
* **Phase 1:** elastic artificials on the linking rows are driven to zero (no big-M).
* **Phase 2:** optimise the true objective.
* **Pricing:** min (c_b + A_linkᵀ y_link) x over each block's polyhedron. All blocks are solved
  **concurrently** (`std::async`), and a column enters when its reduced cost is below −1e-9.
* **Rigorous bound:** linking duals from the master and block-row duals from the pricing
  problems form a full dual vector y. The **independent verifier** then evaluates the
  outward-rounded Lagrangian bound of the *original* model at that y.
* **Primal recovery:** x = Σ λ·(extreme points), verified on the original model.

### Evidence
| Instance | Blocks / border rows | DW iterations | Columns | DW objective | Direct objective | Verified gap |
|---|---|---|---|---|---|---|
| multi_refinery (4) | 4 / 5 | 8 | 23 | 36960.8770 | 36960.8770 | 4e-10 |
| 12 refineries | 12 / 5 | 9 | 73 | 111429.5770 | 111429.5770 | 2e-9 |
| 40 refineries | 40 / 5 | 9 | 237 | 372387.1034 | 372387.1034 | 9e-9 |

### Limits
On these models the direct simplex is faster (7 ms vs 16 ms at 40 blocks), so no speedup is
claimed; decomposition pays off when blocks are large or individually hard. Continuous LPs
only. Unbounded block variables are boxed at ±10⁶·scale, and the bound is then withheld.

---

## 9. Differentiable LP layer (OptNet-style)

### Purpose
SANKHYA-OPT offers a PyTorch layer that backpropagates through optimisation. NIRYUKTI now
returns exact derivatives of an LP optimum and ships an optional PyTorch adapter. The engine
itself needs no ML packages.

### Usage
```bash
niryukti differentiate examples/production.json                       # objective gradients
niryukti differentiate examples/production.json --upstream g.json     # VJP for dl/dx = g
niryukti differentiate model.json --upstream g.json --samples 64 --sigma 0.05   # + cost gradient
```
```python
import torch, niryukti
layer = niryukti.torch_lp_layer(model_dict, samples=32)
c, lo, up = (torch.tensor(v, dtype=torch.float64, requires_grad=True)
             for v in niryukti.lp_parameters(model_dict))
x = layer(c, lo, up); loss = f(x); loss.backward()
```

### Mathematics (implicit differentiation at the optimal basis; Amos & Kolter 2017)
With x_B = −B⁻¹N x_N and w = B⁻ᵀ g_B for an upstream gradient g = ∂ℓ/∂x:
* ∂ℓ/∂(active bound of row i) = w_i; inactive rows give 0.
* ∂ℓ/∂(bound of nonbasic variable j) = g_j − wᵀa_j.
* ∂ℓ/∂A_ij = −w_i x_j.
* Objective gradients: ∂z/∂c = x, ∂z/∂b = shadow prices, ∂z/∂A_ij = −π_i x_j.
* **Costs:** x*(c) is piecewise constant, so the exact derivative is 0 almost everywhere. With
  `--samples K`, an antithetic **perturbed-optimiser** estimate (Berthet et al., NeurIPS 2020)
  gives a useful smoothed gradient: E[(gᵀx*(c+σZ) − gᵀx*(c−σZ)) Z] / 2σ.

It reuses the sensitivity module's reconstructed and repaired basis and the mixed-precision LU.

### Evidence
* Production model with ℓ = production_2 + 0.5·inventory_1: VJP = (0.5, 1.0, −0.5, 0) for
  (demand₁, demand₂, capacity₁, capacity₂). Hand derivation and finite differences agree exactly.
* A matrix-entry gradient matches a re-solve finite difference (test).
* **PyTorch 2.14** (scratch virtualenv, not a project dependency): autograd gradients equal
  finite differences, and SGD through the layer learns capacity₁ = 60 so that production₂
  reaches its target 50 (final loss 8e-16).

### Limits
* Continuous LPs only.
* At a degenerate basis the derivatives are one-sided; the report sets `degenerate_basis`.
* Each forward/backward is a subprocess solve, which suits planning-scale models, not batched
  deep-learning training.

---

## 10. GNN-guided branching (learned from strong branching)

### Purpose
SANKHYA-OPT replaces branching heuristics with a bipartite graph neural network. NIRYUKTI now
has its own **dependency-free** C++ GNN, a training command and trained default weights
compiled into the engine.

### Usage
```bash
niryukti solve model.json --branching gnn                               # built-in weights
niryukti train-branching weights.json train/*.json --dives 8 --epochs 120
niryukti solve model.json --branching gnn --branching-model weights.json
python3 examples/generate_milp_benchmarks.py /tmp/train --count 20 --seed 1
```

### Model (Gasse et al., NeurIPS 2019)
* **Graph:** constraint nodes and variable nodes of the node LP. Edges are nonzeros, weighted
  |a_ij| / max_k |a_ik|.
* **Variable features (10):** cost, fractionality, distance to integer, integrality, at-lower,
  at-upper, reduced cost, degree, cost/column-weight ratio, share of tight rows.
* **Constraint features (4):** normalised dual, tightness, equality, degree.
* **Layers:** constraint update h_c = ReLU(W_c [mean_j w_ij x_j , x_c]), then variable update
  h_v = ReLU(W_v [x_v , mean_i w_ij h_c,i]), then a linear score. Hidden size 16, 689 parameters.
* **Training:** imitation learning on strong-branching labels (product score
  max(Δ⁻,ε)·max(Δ⁺,ε); infeasible child = large gain). The loss is tie-aware cross-entropy:
  −log Σ_{ties} softmax. Labels come from expert-guided dives with 30% exploration, and the
  optimiser is Adam with L2 regularisation. Hand-written backprop is **gradient-checked** against
  finite differences (worst relative error 1.6e-7, in CTest `gnn`).
* **Use in branch-and-bound:** at each node the graph is built from the node LP (bounds and cuts
  included) and the highest-scoring fractional variable is branched on. Pruning, bounds and
  verification are unchanged, so the learned rule only affects search order, never correctness.

### Shipped weights (`models/branching_gnn.json`, embedded via `src/gnn_weights.inc`)
* **Training data:** 90 generated instances (set cover, capacitated facility location,
  multidimensional knapsack; seeds 1–2), giving 8,073 labelled nodes from 85,728 LP solves (20 min).
* **Validation:** top-1 agreement with strong branching 51.9% (most-fractional rule 48.1%);
  top-3 agreement 88.8%.

**Held-out test** (30 new instances, seed 99, geometric means):

| Family | Fractional nodes / time | Reliability nodes / time | **GNN nodes / time** |
|---|---|---|---|
| Set cover | 3.6 / 26.8 ms | 3.3 / 58.6 ms | **2.7 / 21.1 ms** |
| Facility location | 18.4 / 180 ms | 17.9 / 331 ms | 18.1 / 176 ms |
| Knapsack | 2775 / 406 ms | **2356 / 355 ms** | 2493 / 370 ms |

The GNN beats the default fractional rule on all three families (knapsack −10% nodes, set
cover −25%) and is the fastest on two. Reliability branching still uses fewer nodes on
knapsack. One knapsack instance hits the default 10,000-node limit with both fractional and
GNN branching. These are small synthetic instances; MIPLIB-scale benefit is not claimed.

---

## 11. Controlled-English model translator

### Purpose
SANKHYA-OPT's "Agentic SLM Studio" turns plain English into matrices with a language model.
NIRYUKTI offers an **offline, deterministic** alternative that cannot hallucinate. It is a
controlled-English grammar that echoes every statement back as algebra, and it **rejects**
anything it cannot parse instead of guessing.

### Usage
```bash
niryukti translate examples/refinery_plan.txt --output /tmp/plan.json   # Python CLI
niryukti solve /tmp/plan.json
```
```python
model, report = niryukti.translate(open("examples/refinery_plan.txt").read())
print(report["understood"])
```

### Grammar (excerpt)
```
variables light_crude, heavy_crude between 0 and 400      # also: binary x / integer y / free z
property sulfur: light_crude 0.5, heavy_crude 2.8
maximize 92 petrol - 61 light_crude - 900 run_unit       # minimize / maximise
cdu_capacity: light_crude + heavy_crude is at most 600   # at least / no more than / equals / <= >= =
petrol_demand: petrol is between 120 and 220
total: sum of a, b, c >= 10
sulfur_spec: average sulfur of light_crude, heavy_crude at most 1.4   # -> sum (q_i - 1.4) x_i <= 0
```
The `average property` form linearises a blend-quality specification
(Σ q_i x_i ≤ s Σ x_i). Undeclared variables are accepted as continuous ≥ 0 with a warning.
Unparsed lines are returned with line numbers.

### Limits
This is not a language model: free-form sentences outside the grammar are rejected by design.
Bilinear pooling specifications must still be written in the bilinear JSON format (§4).

---

## 12. Build fix: vendored Eigen with libc++ 21+

Apple clang 21 / LLVM 22 libc++ requires `operator[]` on random-access iterators used by
`std::partial_sort`. Eigen's `CompressedStorageIterator` lacked it, so `src/model.cpp` failed
to compile on this machine. This was true of the unmodified repository as well. A one-line
`operator[]` (as in later Eigen releases) was added in
`third_party/eigen/Eigen/src/SparseCore/SparseCompressedBase.h` and recorded in
`third_party/EIGEN-NOTICE.md`. The default build now compiles, and **the complete
`scripts/run_tests.sh` passes** (it previously could not build).

---

## 13. Files added or changed

| File | Change |
|---|---|
| `include/vantage/analysis.hpp` | Public API: `sensitivity_json`, `differentiate_json`, `iis_json`, `global_solve_json`, `decompose_json`, history functions |
| `include/vantage/vantage.hpp` | `Options::newton_precision`, `Options::branching_model`; mixed-precision statistics in `Result` |
| `src/sensitivity.cpp` | Basis reconstruction, Bland repair, ranging, findings, **LP differentiation** |
| `src/dense_lu.hpp` | Dense LU with FP32 factor + FP64 iterative refinement |
| `src/barrier.cpp` | **FP32 Newton factor/operator + FP64 refinement with fallback** |
| `src/cuda.cu`, `src/cuda_stub.cpp`, `src/internal.hpp` | FP32-operator flag and tolerance for GPU Newton solves |
| `src/decompose.cpp` | **Structure detection and Dantzig–Wolfe with concurrent pricing** |
| `src/gnn.hpp`, `src/gnn.cpp`, `src/gnn_weights.inc` | **Bipartite GNN, backprop, Adam training, built-in weights** |
| `src/mip.cpp`, `src/solver.cpp` | `--branching gnn`; option validation |
| `src/iis.cpp`, `src/farkas.hpp` | IIS search and exact-sign Farkas certification |
| `src/global.cpp` | Bilinear McCormick spatial branch-and-bound |
| `src/history.cpp` | Learned method selection |
| `src/io.cpp` | `mixed_precision` result block |
| `app/main.cpp` | Commands `sensitivity`, `differentiate`, `iis`, `global`, `decompose`, `train-branching`, `recommend`; flags `--newton-precision`, `--branching gnn`, `--branching-model`, `--history`, `--method learned`; `assurance` badge |
| `python/vantage/analysis.py`, `differentiable.py`, `language.py`, `__init__.py`, `python/niryukti/__init__.py`, `python/niryukti/cli.py` | Python API (`sensitivity`, `find_iis`, `solve_global`, `recommend`, `decompose`, `differentiate`, `torch_lp_layer`, `translate`) and `niryukti translate` |
| `models/branching_gnn.json` | Trained branching weights with training metrics |
| `examples/haverly_pooling.json`, `infeasible_blend.json`, `multi_refinery.json`, `refinery_plan.txt` | Demonstration models |
| `examples/generate_multi_refinery.py`, `generate_milp_benchmarks.py` | Instance generators |
| `tests/test_analysis.py`, `tests/test_advanced.py`, `tests/test_gnn.cpp` | 28 Python regression tests + C++ gradient check |
| `CMakeLists.txt`, `scripts/run_tests.sh` | New sources and tests |
| `third_party/eigen/.../SparseCompressedBase.h`, `third_party/EIGEN-NOTICE.md` | libc++ compatibility patch and notice |

---

## 14. Validation performed

* **Default build (Eigen sparse LU on):** `./scripts/run_tests.sh` passes completely (exit 0).
  That covers 5/5 CTest suites including the new `gnn` gradient check, every existing Python
  suite, `test_analysis.py` (14) and `test_advanced.py` (14). The PyTorch test is skipped
  without PyTorch and **passes** with PyTorch 2.14 in a scratch virtualenv.
* **AddressSanitizer + UBSan (Eigen build):** the GNN test, `test_advanced.py` and
  `test_analysis.py` pass with no reports from new code. The barrier emits Eigen-internal
  128-byte alignment-assumption notes in **plain FP64 mode as well**; these predate this work.
* **Non-Eigen build (`-DVANTAGE_SPARSE_LU=OFF`):** all new suites pass. The C++ `research`
  test and `test_engine_checkpoints.py` need sparse LU and fail identically on pristine `HEAD`.
* **Known solver limitation:** with Eigen enabled, the existing revised simplex reports
  `NUMERICAL_ERROR` on Netlib e226 (dense-basis builds solve it). Sensitivity and
  differentiation then report that failure rather than analysing a non-vertex point.
* **Not validated here:** the CUDA mixed-precision Newton path (no NVIDIA GPU available).
