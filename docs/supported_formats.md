# Supported formats and result semantics

## Native JSON

```json
{
  "name": "example",
  "sense": "min",
  "variables": [
    {"name": "x", "lb": 0, "ub": 10, "type": "continuous"},
    {"name": "y", "lb": 0, "ub": 1, "type": "binary"}
  ],
  "objective": {"linear": [1, 2], "quadratic_diagonal": [0, 0], "offset": 0},
  "constraints": [
    {"name": "demand", "lb": 2, "ub": null, "coefficients": {"x": 1, "y": 3}}
  ]
}
```

Bounds omitted for variables default to [0,+infinity]; explicitly null lower/upper bounds mean -infinity/+infinity respectively. Row bounds default to unbounded. Strings `-inf` and `inf` are also accepted. Do not put nonstandard JSON NaN/Infinity numeric literals in files. Linear coefficients and optional diagonal Q follow variable order; row coefficient objects refer to variable names. Unknown variables, bad dimensions, NaNs, infinite coefficients and invalid bounds are rejected.

Only diagonal quadratic objectives are accepted. Their entries are Q's diagonal in `0.5*xᵀQ*x`, not coefficients lacking the half factor. Integer quadratic problems and nonconvex objectives return `UNSUPPORTED`.

## MPS

Supported: `NAME`, `OBJSENSE`, `ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`, `ENDATA`; `INTORG`/`INTEND` markers; `LO UP FX FR MI PL BV LI UI` bounds; diagonal `QMATRIX` and `QUADOBJ`.

`INTORG` variables with no BOUNDS record default to [0,1]. An explicit BOUNDS record switches to ordinary defaults before its value is applied, matching the comparison adapter’s parser convention.

Free whitespace-separated fields and conventional fixed-format lines without embedded spaces in names are accepted. Blank repeated column/set fields are accepted where unambiguous. Section headers must begin in column one, and data lines must be indented. Duplicate matrix coefficients are summed. The first/default objective is the single N row; multiple N rows, multiple RHS/bound/range sets, `OBJNAME`, `QSECTION`, quadratic constraints, SOS and other extensions are explicitly unsupported. `QMATRIX` and `QUADOBJ` off-diagonal terms are rejected rather than misinterpreted.

This is a documented useful MPS subset, not a complete implementation of every historical dialect. The public suite is intended to expose additional parser needs. MPS export accepts the canonical linear/diagonal-quadratic representation; fully free rows cannot be exported to this subset.

## LP text

Section headings: `Minimize`/`Maximize`, `Subject To`, `Bounds`, `Binary`/`Binaries`, `General`/`Integer` (plural variants), and `End`. Variable names start with a letter or underscore and can include letters, digits, underscores and dots. Signed/scientific numerical coefficients, optional row/objective labels, constants, and linear terms on both sides of a row are accepted. Backslash comments are supported.

Examples of bounds: `0 <= x <= 10`, `x >= -2`, `x = 4`, `x free`. Quadratic expressions and nonlinear operators are unsupported in LP text; use native JSON or diagonal quadratic MPS.

Objective lines can span lines. A constraint can span lines until its comparison is present, but a complete row/RHS should be on that final line. Ranged LP rows, continuation after an already complete row, quoted names, SOS and indicator syntax are not supported. Use MPS/JSON for those representable canonical constraints.

## Results

Result files are JSON, including files written with `--solution-out`. They contain primal and row-dual arrays, model fingerprint and ordered names, objective in the original objective sense, original-space accuracy, backend and timings. Nonfinite or unavailable numeric diagnostics serialize as null. `accuracy.dual_bound` uses canonical minimization units; `mip.best_bound` uses the original objective sense.

The fingerprint is a deterministic FNV-1a hash of canonical numeric data and integrality, not a security checksum or proof. Benchmark datasets and binaries additionally use SHA-256. A changed fingerprint is rejected for verification. Parametric warm starts must explicitly allow a model change and preserve ordered row/variable names.

Continuous KKT metrics on a MILP incumbent describe the original continuous formulation; they do **not** measure integer-program optimality. Use `mip.relative_gap` and the tree status for MILP, and independently verify incumbent feasibility.
