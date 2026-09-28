# NIRYUKTI — sparse optimization for Python

An independently implemented C++20 optimization engine with a Python model API,
native sessions, a command-line interface, authenticated HTTP service and offline
HTML reporting. No existing optimization solver is used to solve your model.

## Install

```bash
pip install niryukti
niryukti devices
```

Python 3.10+ is required. Linux x86_64 wheels bundle the FP64 CPU executable and
native library. Source installations require CMake 3.24+ and a C++20 compiler.
CUDA is optional and is not bundled in CPU wheels. Version 0.2.1 includes the service and report additions described below.

## Build and solve a model

```python
from niryukti import Model, NativeSession

model = Model()
model.add_var("x", ub=10)
model.add_constraint({"x": 1}, ">=", 2)
model.set_objective({"x": 3})
result = model.solve(device="auto", method="auto", time_limit=60)
print(result["status"], result["objective"])

# Keep model data in a native session for repeated requests.
with NativeSession(model) as session:
    print(session.solve())
```

LP, supported convex sparse QP, MILP and convex MIQP are available. Sparse
convexity certification and advanced integer algorithms have documented limits;
this is a research optimizer, not a validated replacement for every industrial
solver. Nonconvex global optimization is unsupported. Always inspect status and
verification diagnostics rather than assuming a returned point is optimal.

## CLI, verification and reports

```bash
niryukti inspect model.mps
niryukti solve model.mps --method auto --device auto --time-limit 60 --json-out result.json
niryukti verify model.mps result.json
niryukti report result.json --output report.html
```

Automatic selection considers structure and GPU memory; explicit methods remain
available. Reports are portable, printable HTML derived from saved telemetry.
Rendering a report does not independently certify its input.

## HTTP API

```bash
export NIRYUKTI_API_TOKEN="your-long-random-secret"
niryukti serve --host 127.0.0.1 --port 8090 --workers 2 --max-time 300
```

Send `Authorization: Bearer <token>` to `/v1/health`, `/v1/solve` or
`/v1/report`. Solve requests contain a native JSON model and optional solver
settings. The service bounds request size, concurrency and solve time, and does
not accept client executable paths. It is synchronous; disconnecting does not
cancel computation. Put remote deployments behind HTTPS and appropriate access
controls. Full examples are in the repository's `docs/api.md`.

## CUDA and source builds

Build the repository with `-DNIRYUKTI_CUDA=ON`, then set `NIRYUKTI_BINARY` to
its executable and `NIRYUKTI_LIBRARY` to `libniryukti_c.so` when using native
sessions. Legacy VANTAGE environment variables remain compatibility aliases.
Build distributions with `python -m build`; CPU wheels do not download GPU code.

## License

AGPL-3.0-only for original project code, with third-party notices retained.
Copying and commercial use are permitted under the license terms. Covered
modified distributions retain AGPL; modified network deployments must offer
corresponding source to users. See the bundled LICENSE and NOTICE.

## Version 0.2.2

Adds checkpoint/resume for simplex, barrier and concurrent portfolios, guarded
sparse singular-PSD ordering, general-integer bound conflicts and integer-lattice
cuts. CUDA source builds also include continuous bound derivation replay, dual
reconstruction and GPU CSR compaction. CPU distributions include the same
mathematical core, but require a separate CUDA build for GPU execution.

Integer incumbent verification is distinct from replaying a full search-tree
proof. Large/difficult PSD recognition, general dual conflict analysis and direct
GPU barrier factorization remain restricted. No universal speedup is claimed.
