# Independent optimization certificates

NIRYUKTI certificates carry candidates and independently replayable numerical evidence.
They let an auditor verify an original model without executing optimization iterations.
The lifecycle is solve → candidate → native certificate generator → JSON → native verifier.
`src/certificate.cpp` uses the existing `Verifier`; Python and the dashboard do not
implement alternative numerical checks. No optimization dependency was added.

## CLI and offline demo

```sh
build/vantage solve examples/toy.lp --device cpu --certificate certificate.json
build/vantage verify examples/toy.lp certificate.json
./scripts/run_certificate_demo.sh
```

`--certificate-out` is an alias. Verification exits 0 for valid evidence, 2 for
failed evidence, and 1 for invalid CLI input or unreadable/malformed files.
Existing result verification and solve exit codes remain available. Every CLI and
session solve result includes `verification_certificate`. Infeasible results preserve their legacy
`certificate` Farkas structure. Unsuccessful candidates may carry an INVALID report.

## Mathematical checks

The native verifier validates the model and recomputes Ax, Aᵀy, Qx, the objective,
variable bounds, row feasibility, integrality, projected box stationarity,
dual sign restrictions, complementarity and KKT residuals. LP and supported convex
QP optimality require the existing KKT checks to pass. General sparse Q uses the
existing model convexity validation. Indefinite diagonal Q cannot receive an
optimality certificate. These are floating-point numerical guarantees at the
recorded tolerances, not exact rational proofs.

Primal residual normalization is per row endpoint (1 + absolute endpoint) and
per variable (1 + absolute candidate value). Stationarity uses the existing
objective coefficient scale. Absolute primal and stationarity residuals and
unnormalized complementarity are also reported. The KKT envelope includes
feasibility, projected stationarity and the existing complementarity/dual gap.
Objective consistency allows tolerance × (1 + absolute canonical objective).
The certificate never relaxes the solver tolerance: feasibility and integrality
are capped at 1e-6, and MIP gap at 1e-4. Tighter solver settings are retained.

MILP incumbents must satisfy the original constraints, bounds and integrality.
The verifier computes an outward-rounded relaxation lower bound from the supplied
dual vector over the original variable box. Only that bound may establish
`OPTIMAL_WITHIN_VERIFIED_GAP_TOLERANCE`; otherwise the certificate says
`FEASIBLE_WITH_UNRESOLVED_GAP`, even when the solver reports OPTIMAL. Absolute gap
is max(0, incumbent − lower bound); relative gap divides by max(1, |incumbent|).
Bounds are in canonical minimization coordinates; displayed objective uses the
original sense. Optional tree telemetry is checked for arithmetic consistency,
but is explicitly not a replay of branch-and-bound, cuts, or node fathoming.

LP box/row Farkas rays use `Verifier::infeasibility_bound`. A finite, strictly
positive outward-rounded separation margin establishes infeasibility. Unsupported
infeasibility claims, including those without a replayable ray, are not certified.

## Fingerprints and tamper detection

Certificate SHA-256 fingerprints hash deterministic JSON of the canonical model:
name-sorted variables/rows, bounds, types, objective, sense, offset, sparse A and
combined q + Q. Zero coefficients are omitted and negative zero normalized.
Infinite bounds use explicit strings. Input ordering with stable names does not
change identity; renaming variables or rows does. This is not graph-isomorphism
canonicalization. Duplicate coefficients are combined by the existing sparse
reader; floating-point differences after summation represent different stored
models. Legacy checkpoint fingerprints remain unchanged.

The payload SHA-256 detects accidental edits, including changes to status or
metadata. It is **not a signature or proof of provenance**: an attacker can
recompute it. Mathematical claims and saved diagnostics are independently replayed
even after a payload is resealed. A different but valid candidate with correctly
recomputed evidence can be accepted. Solver termination is provenance metadata,
not mathematical evidence. Timestamp, if supplied, is excluded from the payload
checksum and does not affect verification. Generation omits timestamps for
reproducibility; build git commit is included when available.

## Python and dashboard

```python
from vantage import solve, verify_certificate
result = solve('examples/toy.lp', device='cpu')
certificate = result['verification_certificate']
report = verify_certificate('examples/toy.lp', certificate)
assert report['valid']
```

`Model` objects and model dictionaries are also accepted. `NativeSession` provides
`verify_certificate(certificate)` through the C ABI; ordinary `SolverSession`
solves include the same certificate. CPU and CUDA candidates use exactly the
same original-model native verification path. Bit-identical solutions are not
required. CUDA tests skip when no usable CUDA backend exists.

The dashboard shows certificate status, residuals, View Certificate and Explain
Verification panels, with certificate and verification report downloads. The
explanations are fixed text attached to actual check results. Archived runs without
certificates retain their legacy diagnostics.

## Schema and robustness

See [schema](schema/certificate.schema.json). Version 1.0 requires solver identity,
model fingerprint/dimensions/type/sense, termination metadata, configuration,
tolerances, candidate kind, verification report and payload digest. Solutions
require named primal/dual maps and finite objective; infeasibility requires a named
ray. Bounds and RHS are referenced through the fingerprint and original model,
not duplicated. Unknown schema versions are rejected. Reports must match a fresh
replay; missing fields, wrong dimensions, nonfinite candidates, malformed input,
wrong models and inconsistent claims fail. File inputs are limited to 64 MiB and
32 nesting levels. Very large certificates require a future streaming format.

Statuses are OPTIMAL (continuous KKT), FEASIBLE (continuous feasibility only),
OPTIMAL_WITHIN_VERIFIED_GAP_TOLERANCE (integer feasibility plus replayed bound),
FEASIBLE_WITH_UNRESOLVED_GAP, INFEASIBLE (Farkas), and INVALID. A VALID certificate
can establish only feasibility. No tree proof, cryptographic signer authentication,
exact arithmetic guarantee, or unsupported infeasibility proof is implied.
