# Implemented mathematics

## Continuous formulation

VANTAGE minimizes

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

The dual update is the interval-support proximal operator. Its split form makes inactive one-sided multipliers exactly zero, avoiding division-and-subtraction artifacts. QP uses the exact diagonal proximal formula, not an inner CG algorithm. General Q would require a new proximal solver or a correctly controlled forward-gradient method.

Let `M = sqrt(||A||₁ ||A||∞)`, an upper bound on the spectral norm. The initial base step is `s=0.9/M` (1 for a zero matrix), with `τ=s/w` and `σ=s*w`. In fixed-step mode this preserves the conservative product bound. Adaptive mode initializes w from objective/RHS norms and changes the step factor using local trial displacements.

For a trial, form `E=||dx||²/τ + ||dy||²/σ` and `C=dyᵀA dx`. Accept a finite trial when `E >= 2|C|`. Define `limit=E/(2|C|)` (infinity for zero coupling). With k equal to accepted steps plus two, multiply the trial step factor by `min((1-k^-0.3)*limit, 1+k^-0.6)` for the next attempt, clipped to [1e-12,1e12]. Invalid arithmetic reduces it by one half. Rejected trials do not modify accepted iterates or averages. This is an independently implemented heuristic inspired by the PDLP paper's line search, with absolute coupling; it is not the complete PDLP algorithm or a claim of universal convergence.

Average accepted iterates using their step factors as weights. Check both current and averaged candidates in original units. After at least two checkpoints, restart when KKT error halves or when the epoch length reaches `max(2000,total_iterations/2)`. At restart averaging resets and primal/dual weighting is damped toward the displacement-norm ratio, clipped to [1e-4,1e4]. The CUDA backend reduces trial statistics and accepts/rejects on device. Six control scalars return per chunk; full candidate vectors return only at monitoring checkpoints. GPU chunks count attempts, so accepted iteration checkpoints can differ slightly from CPU checkpoints.

## Presolve and scaling

Fixed variables are substituted into row bounds and objective, and bounded isolated columns are minimized independently. Empty rows are removed when satisfied. Disjoint row-activity intervals can prove infeasibility. Roundoff allowances use long-double accumulation and magnitude-dependent guards; this presolve is conservative, not an exact rational proof system.

Infinity-norm diagonal equilibration alternately normalizes rows and columns by inverse square roots of their maximum coefficient magnitudes. Each per-pass factor is clipped to [1e-3,1e3]. Objectives are normalized by the largest scaled linear/diagonal coefficient. No tiny nonzero coefficient is dropped. Scaling is always reversed before stopping checks.

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

The diagonal QP bound uses separable quadratic minimization with a roundoff allowance; it is not used by MILP because MIQP is unsupported. B&B uses only LP Lagrangian bounds and inherited node bounds. MILP incumbents are feasible to the requested numerical tolerances, rather than exact rational feasibility.

## MILP

The tree relaxes integrality through VANTAGE's own continuous solver. Integer domains are rounded inward initially. At each node, up to five propagation passes derive implied integer bounds from prefix/suffix row-activity intervals. Every arithmetic operation expands its long-double interval outward; tightening is restricted to exactly representable integer magnitudes. Continuous bounds and row intervals are retained, avoiding a new dual postsolve mapping. Most-fractional branching partitions a variable at floor/ceiling; nodes are selected by best bound. Candidates are rounded and independently checked, and occasional round-and-repair solves fix integers while optimizing continuous variables. Children inherit the primal/dual warm start.

An LP primal objective is not a pruning bound. Nodes close only through presolve infeasibility or a valid Lagrangian/inherited bound relative to the incumbent and requested MIP gap. Approximately integral but unresolved relaxations remain unresolved leaves. Global bound reporting includes the queue, unresolved leaves and bounds of fathomed leaves. A standalone solution file verifies an incumbent, not a replayable tree proof.

## Certificates and interruptions

For a candidate row multiplier y, independently evaluate the conservative box Lagrangian bound with c=0 and objective offset=0. A strictly positive finite result contradicts the zero objective of every feasible point and certifies infeasibility. The solver periodically tries normalized current multipliers after 1,000 accepted iterations. JSON stores a `BOX_ROW_FARKAS` ray and margin; `verify` recomputes the margin rather than trusting the stored number. This is conservative and can fail to certify models with unbounded boxes or cancellation; failure to find a ray does not imply feasibility. Presolve contradictions and completed MILP trees are also supported, but do not currently emit independently replayable proof artifacts. General recession-ray extraction remains unimplemented. `UNBOUNDED` currently requires a checked feasible point and an isolated improving unbounded LP column. Other divergent problems return limits/unknown/numerical error. SIGINT and time limits are observed between iteration chunks; preprocessing and monitoring are not immediately interruptible.

## Mathematical references

The implementation was written independently from these mathematical ideas; no solver iteration code was transplanted.

- Chambolle and Pock, [A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging](https://doi.org/10.1007/s10851-010-0251-1).
- Applegate et al., [Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient](https://arxiv.org/abs/2106.04756). VANTAGE implements a smaller, distinct feature set; the paper's performance claims do not describe this prototype.
- Applegate et al., [Infeasibility detection with primal-dual hybrid gradient for large-scale linear programming](https://arxiv.org/abs/2102.04592), a future certificate-development reference.
- [MOSEK's MPS format documentation](https://docs.mosek.com/11.1/pythonapi/mps-format.html), used for format semantics, not optimization code.
