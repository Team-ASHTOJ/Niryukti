# Implemented mathematics

## Continuous formulation

NIRYUKTI minimizes

`f(x) = cᵀx + 0.5 Σ qⱼ xⱼ² + c₀`, with `l ≤ x ≤ u` and `L ≤ Ax ≤ U`.

The current diagonal QP backend requires `qⱼ ≥ 0`. All-zero q is LP. The indicator of the row interval has a support-function conjugate. This gives the saddle problem

`min_{x in [l,u]} max_y f(x) + yᵀAx - support_[L,U](y)`.

The sign convention is y≥0 for upper-only rows, y≤0 for lower-only rows, and unrestricted y for equalities. The independent verifier uses this convention; conventional solver row prices often have the opposite sign.

## PDHG iteration

For positive steps τ, σ:

1. `x⁺ⱼ = clip((xⱼ - τ(cⱼ + (Aᵀy)ⱼ)) / (1 + τqⱼ), lⱼ, uⱼ)`.
2. `xbar = 2x⁺ - x`.
3. `z = y + σ A xbar`.
4. `y⁺ᵢ = max(0, zᵢ - σUᵢ) + min(0, zᵢ - σLᵢ)`.

The dual update is the interval-support proximal operator. Its split form makes inactive one-sided multipliers exactly zero, avoiding division-and-subtraction artifacts. Diagonal QP uses the exact separable proximal formula. Supported general sparse Q uses the forward-gradient splitting described below; it is not labeled as rAPDHG or PDHCG.

The backends evaluate `A*dx` directly and form `A*xbar = A*x + 2*A*dx`. This avoids cancellation in the line-search coupling that occurs when subtracting two separately accumulated activities. Accepted activity advances by `A*dx`; CPU monitoring checkpoints and backend restarts recompute activity. There are still two sparse products per trial (the transpose product and the displacement product). CPU and CUDA both require `limit >= 1`; the former CPU threshold of 0.5 was inconsistent with the inequality below.

Let `M = sqrt(||A||₁ ||A||∞)`, an upper bound on the spectral norm. The initial base step is `s=0.9/M` (1 for a zero matrix), with `τ=s/w` and `σ=s*w`. In fixed-step mode this preserves the conservative product bound. Adaptive mode initializes w from objective/RHS norms and changes the step factor using local trial displacements.

For a trial, form `E=||dx||²/τ + ||dy||²/σ` and `C=dyᵀA dx`. Accept a finite trial when `E >= 2|C|`. Define `limit=E/(2|C|)` (infinity for zero coupling). With k equal to accepted steps plus two, multiply the trial step factor by `min((1-k^-0.3)*limit, 1+k^-0.6)` for the next attempt, clipped to [1e-12,1e12]. Invalid arithmetic reduces it by one half. Rejected trials do not modify accepted iterates or averages. This is an independently implemented heuristic inspired by the PDLP paper's line search, with absolute coupling; it is not the complete PDLP algorithm or a claim of universal convergence.

Average accepted iterates using their step factors as weights. Check both current and averaged candidates in original units. After at least two checkpoints, restart when KKT error halves or when the epoch length reaches `max(2000,total_iterations/2)`. At restart averaging resets and primal/dual weighting is damped toward the displacement-norm ratio, clipped to [1e-4,1e4]. The CUDA backend reduces trial statistics and accepts/rejects on device. Six control scalars return per chunk; full candidate vectors return only at monitoring checkpoints. GPU chunks count attempts, so accepted iteration checkpoints can differ slightly from CPU checkpoints.

## Presolve and scaling

Fixed variables are substituted into row bounds and objective, and bounded isolated columns are minimized independently. Empty rows are removed when satisfied. Disjoint row-activity intervals can prove infeasibility. Roundoff allowances use long-double accumulation and magnitude-dependent guards; this presolve is conservative, not an exact rational proof system.

Infinity-norm diagonal equilibration alternately normalizes rows and columns by inverse square roots of their maximum coefficient magnitudes. Each per-pass factor is clipped to [1e-3,1e3]. Objectives are normalized by the largest scaled linear/diagonal coefficient. No tiny nonzero coefficient is dropped. Scaling is always reversed before stopping checks.

`--scaling combined` adds one geometric-mean row/column sweep before Ruiz and one simultaneous Pock–Chambolle sweep (alpha=1) afterward. Geometric factors use the reciprocal geometric mean of the smallest/largest nonzero magnitude, evaluated in log space. Pock–Chambolle factors are reciprocal square roots of row/column absolute sums. Extra factors are clipped to [1e-3,1e3]; each extra row/column transformation is skipped if it would overflow finite data or underflow a nonzero to zero. The existing Ruiz and objective normalization stages retain their existing arithmetic. Diagonal Q transforms by the square of the column scale. Zero scaling passes disables every stage. The solver still recomputes its conservative norm bound after scaling; clipping is not treated as a spectral guarantee.

## Experimental Halpern LP method

`--method halpern --no-adaptive` enables a CPU-only continuous LP experiment based on [Lu and Yang's Halpern PDHG update](https://arxiv.org/html/2407.16144v2). With epoch anchor `a` and zero-based epoch index k, it computes `z_next = (k+1)/(k+2) T(z) + a/(k+2)`. Both the anchored point and unanchored PDHG image are checked in original units. Restarts reset the anchor and k at the better verified candidate using VANTAGE's existing KKT progress/epoch-length heuristic. With `--no-restart`, the anchor stays fixed.

The base step is `0.9/sqrt(||A||1 ||A||inf)`. Primal/dual weight uses the existing objective/RHS norm initialization and damped epoch-displacement update at restarts; the PDHG operator stays fixed within each epoch. Adaptive trial steps, CUDA, QP and MILP are rejected for this method. This is an experimental variant, not a reproduction of the paper's fixed-point restart policy, reflected variant, or theoretical rate claims. Default PDHG is unchanged as the selected method.

## Restarted/reflected LP variants and polishing

`rhpdhg` uses the anchored update above; `r2hpdhg` substitutes `2T(z)-z` for `T(z)` in the anchor mixture. Both monitor `sqrt(E-2C)`, the PDHG fixed-point residual in the canonical metric for VANTAGE's multiplier sign convention. Restarts use the raw image `T(z)`, reset the anchor, and may update weighting between epochs. Their sufficient/necessary/artificial rules and supported combinations are specified in [research_features.md](research_features.md). These are CPU LP experiments inspired by [cuPDLPx](https://arxiv.org/html/2507.14051v2); HPR splitting is a separate unimplemented method.

The optional PID controller uses `e=log(w)+log(||dx||/||dy||)` at restarts, and updates log weight using P/I/D feedback with safeguards. Optional power iteration estimates the scaled matrix spectral norm, but never substitutes for a guaranteed bound in fixed-step methods. Adaptive acceptance remains required when the estimate is used.

LP polishing separately solves primal feasibility (zero original objective) and dual feasibility (zero objective over the multiplier and reduced-cost cones). For `g=c+Aᵀy`, a lower-only column requires `g>=0`, an upper-only column requires `g<=0`, a free column requires `g=0`, and a doubly bounded column has unrestricted g. Only recombining and verifying both candidates against the original LP can establish optimality. Auxiliary solves share the remaining iteration/time budget.

## Independent stopping checks

A reusable verifier caches the original sparse transpose and work arrays, then recomputes A*x and Aᵀ*y from the original model. Checkpoints omit the conservative pruning-bound computation; final convergence and MILP lower bounds always use it. It rejects nonfinite input or intermediate products, including overflow followed by cancellation.

- Primal error is the maximum relative row or variable-bound violation; each violated row endpoint is normalized by one plus its own absolute value, so an enormous opposite endpoint cannot hide a violation.
- Stationarity uses the box projected-gradient mapping `clip(g, x-u, x-l)`, where `g = c + Qx + Aᵀy`. The algebraic form avoids subtracting two nearly equal large x values.
- Complementarity sums the absolute products of row multipliers with endpoint slack and compatible box-gradient components with box slack.
- The reported `relative_gap` is the maximum of normalized complementarity and, where available, the primal/dual objective discrepancy. It is not always a finite two-objective duality gap. The raw bound is null when unavailable.
- `kkt_error` is the maximum of relative primal error, relative stationarity, and that gap measure.

`OPTIMAL` for continuous models means these numerical checks meet the requested tolerance; it is not exact rational optimality. Results also expose absolute violations. Iteration and time limits do not become optimal statuses.

## Lower bounds

For a correctly signed y, the Lagrangian lower bound is

`c₀ - support_[L,U](y) + Σ min_{lⱼ≤t≤uⱼ} (rⱼ t + 0.5 qⱼ t²)`, where `r=c+Aᵀy`.

For LP, the minimizing endpoint is separable. Production LP bound evaluation encloses products and additions with outward-rounded long doubles, including reduced-coefficient uncertainty, then rounds the final value downward. An incompatible infinite variable bound or multiplier sign produces -infinity. Exact-zero operations avoid unnecessary subnormal arithmetic. The bound is for the double-valued input model; it is not a certificate for a different decimal/exact interpretation. No unsafe fast-math is used.

Diagonal QP bounds use separable quadratic minimization with a roundoff allowance. Supported convex MIQP uses the affine-minorant bound described below; B&B uses conservative relaxation bounds and inherited node bounds. MILP incumbents are feasible to the requested numerical tolerances, rather than exact rational feasibility.

## MILP

The tree relaxes integrality through NIRYUKTI's own continuous solver. Integer domains are rounded inward initially. At each node, up to five propagation passes derive implied integer bounds from prefix/suffix row-activity intervals. Every arithmetic operation expands its long-double interval outward; tightening is restricted to exactly representable integer magnitudes. Continuous bounds are also tightened conservatively, with outward-rounded double endpoints; original row intervals are retained, avoiding a new dual postsolve mapping. Most-fractional branching partitions a variable at floor/ceiling; nodes are selected by best bound. Candidates are rounded and independently checked, and occasional round-and-repair solves fix integers while optimizing continuous variables. Children inherit the primal/dual warm start.

An LP primal objective is not a pruning bound. Nodes close only through presolve infeasibility or a valid Lagrangian/inherited bound relative to the incumbent and requested MIP gap. Approximately integral but unresolved relaxations remain unresolved leaves. Global bound reporting includes the queue, unresolved leaves and bounds of fathomed leaves. A standalone solution file verifies an incumbent, not a replayable tree proof.

`--branching reliability` optionally ranks fractional variables by directional pseudocosts. Verified-optimal parent/child LP objectives supply nonnegative objective improvement per unit branch distance. Each direction becomes reliable after two observations. At each node at most four most-fractional candidates receive serial probes, at most 1,000 iterations per direction and within the remaining solve time. Failed or limited probes do not train pseudocosts. Unobserved costs fall back to `max(1,abs(c_j))`; the score is `0.9*min(down,up)+0.1*max(down,up)`. These quantities only select a branch: probe objectives and pseudocosts never replace pruning bounds, close nodes, or discard children. Probe work is included in total iteration/timing counters and separately counted in `mip.strong_branch_probes`. The default remains most-fractional branching. CUDA LP probes can batch up to four candidate variables in both directions using SpMM. Each final candidate is independently verified; only OPTIMAL probes train pseudocosts, and probe estimates never substitute for pruning bounds.

## Certificates and interruptions

For a candidate row multiplier y, independently evaluate the conservative box Lagrangian bound with c=0 and objective offset=0. A strictly positive finite result contradicts the zero objective of every feasible point and certifies infeasibility. The solver periodically tries normalized current multipliers after 1,000 accepted iterations. JSON stores a `BOX_ROW_FARKAS` ray and margin; `verify` recomputes the margin rather than trusting the stored number. This is conservative and can fail to certify models with unbounded boxes or cancellation; failure to find a ray does not imply feasibility. Presolve contradictions and completed MILP trees are also supported, but do not currently emit independently replayable proof artifacts. Normalized current/alternate iterates and epoch-displacement multipliers are tested as dual-ray candidates; their extraction alone is never proof. General primal recession-ray extraction remains unimplemented. `UNBOUNDED` currently requires a checked feasible point and an isolated improving unbounded LP column. Other divergent problems return limits/unknown/numerical error. SIGINT and time limits are observed between iteration chunks; preprocessing and monitoring are not immediately interruptible.

## Mathematical references

The implementation was written independently from these mathematical ideas; no solver iteration code was transplanted.

- Chambolle and Pock, [A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging](https://doi.org/10.1007/s10851-010-0251-1).
- Applegate et al., [Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient](https://arxiv.org/abs/2106.04756). NIRYUKTI implements a smaller, distinct feature set; the paper's performance claims do not describe this prototype.
- Applegate et al., [Infeasibility detection with primal-dual hybrid gradient for large-scale linear programming](https://arxiv.org/abs/2102.04592), a future certificate-development reference.
- [MOSEK's MPS format documentation](https://docs.mosek.com/11.1/pythonapi/mps-format.html), used for format semantics, not optimization code.

## Optional cuts and primal heuristics

For a binary knapsack `sum a_j x_j <= b`, a cover C satisfying `sum_C a_j > b` gives `sum_C x_j <= |C|-1`. A clique whose every pair satisfies `a_i+a_j>b` gives `sum_C x_j<=1`. The separator uses exact integer coefficients and capacities, restricts eligible magnitudes, and checks these conditions without tolerance. These cover/clique cuts are generated at the root. Restricted MIR separation is also available below; tableau-dependent general Gomory/GMI separation remains unsupported.

The feasibility pump alternates integer rounding with minimizing L1 distance to the rounded assignment over the current LP relaxation. General integer distances use epigraph variables with `x_j-d_j<=target_j`, `-x_j-d_j<=-target_j`, and `d_j>=0`. Deterministic perturbations break repeated rounded assignments. RINS fixes integer coordinates shared by an incumbent and relaxation solution, then runs a bounded sub-MIP. Its restricted bounds never prune the global tree. Every heuristic incumbent is checked against the original model.

## September submission additions

### General sparse QP and MIQP bounds

The additional symmetric sparse matrix and diagonal shorthand contribute to the same quadratic objective. General Q uses smooth primal-dual splitting with sparse `Q*x` and conservative steps satisfying `tau*sigma*||A||² + tau*L_Q/2 < 1`; this is not a claimed rAPDHG reproduction. The existing diagonal proximal path remains available.

At any point `x`, convexity gives an affine minorant of the objective. For row multiplier `y`, let `g = c + Q*x + Aᵀ*y` (including the diagonal shorthand), and `h(y)` be the support function of the row interval. Then

```
LB = objective_offset - h(y) - 0.5*xᵀQ*x
     + sum_j min(g_j*lb_j, g_j*ub_j).
```

The implementation encloses coefficients, quadratic terms and support contributions with outward-rounded long-double intervals. If a free bound makes the interval minimization unavailable, the result is negative infinity and cannot prune a node. Numerical convexity validation remains a prerequisite; it is not a formal exact-arithmetic PSD certificate.

### Compact revised primal simplex

A two-phase standardization shifts/splits variables and introduces slack, surplus and artificial columns. Phase I minimizes the artificial-variable sum; phase II prices original objective reduced costs. Pricing columns remain sparse. Sparse Eigen LU numerical factorization and product-form updates solve basis systems; an optional own dense kernel remains available; periodic refactorization and a Bland fallback address some numerical drift/degeneracy. The sparse numerical path has a 4096 transformed-row guard; the optional dense path retains 512. Original-space verification controls `OPTIMAL`; general improving rays currently remain `UNKNOWN` until a recession certificate is available. This is not a warm-basis dual-simplex engine.

### Restricted root mixed-integer rounding

For a transformed row `sum(a_j*z_j) >= b`, nonnegative shifted variables and integral shifts for integer columns, write `f = b-floor(b)`, with `0 < f < 1`. Integer coefficients become `floor(a_j)+min(1, frac(a_j)/f)`; continuous coefficients become `max(0,a_j)/f`; the cut RHS is `ceil(b)`. Only safe finite shifts are accepted. Coefficients round upward and RHS downward to weaken the resulting inequality conservatively. Independently enumerated mixed feasible points test validity. Separation is opt-in. Up to two rounds of node-local cuts use the node bounds and are discarded when leaving that node. A capped global pool derives cuts only from root bounds, tracks efficacy/usage/age and selects at most 16 pooled cuts per node. No tableau/Gomory implementation is claimed.

### Search policies and CUDA diagnostics

Queued integer nodes store bound differences relative to root bounds. Best-bound, depth-first and best-estimate selection are ordering policies only. Reported global bounds always take the minimum over all open/unresolved nodes, regardless of queue ordering.

CUDA monitoring computes scaled current/averaged diagnostic quantities and transfers a scalar. This can defer expensive host candidate checks. It does not replace final independent original-space verification and does not imply fully device-resident restart/control or an independent GPU certificate.


## Predictor-corrector barrier and numerical Newton solve

The independent infeasible-start barrier uses affine primal/dual Newton directions, a cubic centering parameter and the affine-product correction. Slack and multiplier positivity is preserved with a 0.995 fraction-to-boundary step. The regularized sparse KKT matrix contains the objective Hessian, inequality contribution `Gᵀ diag(z/s) G`, and equality blocks. CPU uses Eigen sparse numerical LU, direction-residual checks and refinement. The CUDA option assembles KKT on CPU and runs an independently implemented sparse BiCGSTAB numerical solve on GPU; it is not a cuDSS/direct GPU factorization implementation. Dimension/fill guards reject oversized systems. Only an original-space KKT pass can produce OPTIMAL. E226 exposes barrier numerical failure; no homogeneous infeasibility embedding is claimed.

## Dual simplex and verified portfolio

Compatible warm basis indices are matched to standardized sparse structural columns through a fingerprint. Revised dual pivots restore primal feasibility only after reduced-cost dual feasibility is checked. A singular basis or changed objective that invalidates the dual basis triggers a cold primal initializer. Basis solves use sparse numerical factorization and product-form updates; final candidates are independently verified. Row multipliers are projected onto the exact interval support domain before verification, preventing tiny wrong-sign multipliers from invalidating otherwise finite bounds. Significant dual errors still fail verification.

The concurrent portfolio races supported continuous methods under cooperative local cancellation. The first independently verified optimal candidate wins; remaining workers are joined. This is an experimental robustness option, not a guarantee of faster runtime or perfectly shared global CPU thread budgets.

## Restricted conflicts and state continuation

A binary no-good conflict is learned only from established node infeasibility and unchanged nonbinary root bounds. The pool is capped and matches full fixed binary masks; there is no general implication graph or minimal dual-conflict analysis. Local branching adds an incumbent Hamming-ball neighborhood inside an isolated bounded search, and its bounds never become global pruning bounds.

First-order checkpoints serialize actual CPU/CUDA current/average/anchor states, adaptive steps, epoch counts and host restart/PID controls. MIP checkpoints include the open frontier, bound changes, compatible warm starts/bases, incumbent, pseudocosts, cuts, conflicts and closed/unresolved bounds. Model, backend and mathematical configuration must match. Time/node/iteration budgets may extend. State files are trusted local continuation files, not externally exchangeable proofs; Simplex checkpoints also save the actual phase, basis and degeneracy count; resume refactorizes the basis. Barrier checkpoints save primal/dual/slack Newton state and the best checked candidate. Concurrent portfolios use a manifest and per-engine state files; resume restarts the race without preserving wall-clock ordering. See `local_completion_20260928.md` for validation and limitations.
