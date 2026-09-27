# Document review and implemented extensions — 28 September 2026

Reviewed the supplied Markdown and extracted PDF text of *SIH26119 Competitor Deep-Dive & Above-and-Beyond Technical Roadmap*. They contain the same substantive material. Their internal title uses SIH25119, unlike this project's SIH26119. The text mixes competitor assertions with proposed research and contains no reproducible source links for the claimed implementations/speedups. This review does not establish competitor capabilities, category, license prices or competition requirements.

## Extraction and disposition

| Distinct idea / improvement | Project comparison | Disposition |
| --- | --- | --- |
| Native C FFI for Python | Python previously required a persistent subprocess or separate CLI invocation | Implemented shared C ABI and in-process `NativeSession`, reusing the same session logic as the CLI |
| Hash-linked trust certificates | Benchmarks already retain hashes; standalone portable solve package was missing | Implemented ZIP evidence bundles with original input, canonical model, solution, options, binary hash, verification and member manifest; fresh verification on import |
| Sensitivity analysis for planners | Basis/warm starts existed, but no measured parameter response helper | Implemented two-sided RHS perturbation/re-solve reports, including distinct left/right slopes at kinks; no claim of exact derivatives or crossover |
| Structural dispatcher | Existing `explain`, automatic method/device selection and numerical recovery | Already present; no duplicate implementation |
| Restarted/adaptive PDHG, simplex, IPM | Existing solver paths and original-space verification | Already present; validation is more useful than adding duplicate names |
| Ruiz scaling and FP64 final checks | Existing scaling, mixed matrix precision and verifier | Already present; FP16/BF16 Tensor Core implementation and speed claims not added |
| Original-model feasibility / Farkas evidence | Existing verifier and verified-ray diagnostics | Retained; no promise of a ray for every infeasible integer/nonlinear model |
| Local telemetry and separated UI/engine | Existing dashboard and worker processes | Already present |
| Pinned asynchronous transfers / overlapping GPU compute | Existing GPU implementation needs device profiling to identify profitable changes | Hardware-gated follow-up; no untested kernel changes in this pass |
| GPU crossover to a basic solution | No general crossover engine | Separate numerical project requiring basis construction, degeneracy handling and validation; not implemented |
| Custom multi-GPU sparse factorization | Not supported | Separate hardware/numerical project; not implemented |
| Learned branching via GNN | Reliability branching exists; no training data, model or inference contract supplied | Not implemented; learned scores may choose branch order, never justify pruning by themselves |
| Local SLM operator interface | Structured named model updates exist | No model weights or validated units/formulation system supplied; not implemented. Multipliers must come from optimization, not language-model invention |
| Differentiable OptNet/KKT layer | No autodifferentiation contract | Not implemented; measured finite changes do not replace implicit differentiation or degeneracy treatment |
| Nonconvex blending via ADMM | Supported QP is convex | Not implemented; selecting ADMM alone does not supply a general global-optimality certificate |
| C++23 / mdspan migration | C++20 already supports the needed interfaces | Not adopted as a feature; changing language version alone supplies neither GPU zero-copy execution nor memory safety |
| Audit proprietary binaries for telemetry | Dependency separation is documented and source-scanned | No claim of proving arbitrary binary telemetry absence; scanner now includes C headers/sources |
| 89/89 Netlib, A100/L4 scaling, 4–8×/13× speedups | Not established by supplied raw evidence | Treat as unverified targets/claims, not implemented capabilities |

## Native interface

```sh
./scripts/build.sh
PYTHONPATH=python python3 - <<'PY'
from vantage import NativeSession
with NativeSession('examples/refinery.json') as session:
    first = session.solve(device='cpu', method='auto')
    print(first['status'], first['objective'])
PY
```

`libvantage_c.dylib` (macOS) or `libvantage_c.so` (Linux) is built alongside the CLI. Override discovery using `VANTAGE_LIBRARY` or the `library=` constructor argument. It uses Python's standard `ctypes`, with no numerical Python dependency and no solver subprocess. This removes process launch/IPC from repeated calls; no speedup is inferred from architecture alone. Model loading remains file-based and request/results use JSON; this is not a zero-copy array binding.

The C header is `include/vantage/c_api.h`: ABI version, opaque open/close handle, JSON request and thread-local error retrieval. Open/request translate exceptions to null and a diagnostic. Returned result strings are borrowed until the next request on that handle. Callers must use valid handles and serialize handle access/destruction. Native open/request calls are globally serialized because the solver retains process-global interrupt state. No asynchronous cancellation API is claimed. CLI subprocess isolation remains available.

Both native and subprocess sessions use `src/session.hpp`; named updates are transactional, preserve structural matrices and reuse compatible primal/dual/basis state. Unknown settings and malformed update objects are rejected rather than silently ignored. GPU behavior is not validated on this Mac.

## Evidence bundle

```sh
PYTHONPATH=python python3 -m vantage.evidence create examples/refinery.json /tmp/refinery-evidence.zip --method auto
# Retain the printed SHA-256 externally, then supply it when checking:
PYTHONPATH=python python3 -m vantage.evidence verify /tmp/refinery-evidence.zip --expected-sha256 DIGEST_FROM_CREATE
```

Creation refuses to overwrite an existing bundle. Verification rejects unexpected/duplicate ZIP members, oversized uncompressed payloads and mismatched hashes. It never executes code or command lines from the archive. It invokes only the locally selected solver's verifier against the bundled model and solution. Failed numerical verification returns CLI exit code 2. Stored verification reports are historical evidence, not trusted substitutes for rechecking.

SHA-256 is an integrity checksum, **not a signature**. Without an externally retained archive digest, an attacker can replace both payloads and the manifest; the fresh numerical checker still detects an invalid solution, but cannot authenticate the intended model or author. A verified integer incumbent does not independently certify the search tree. Optional detached signing has subsequently been added: see [signed evidence](trust_and_stable_planning.md). Key management remains external.

## Measured sensitivity

```python
from vantage import Model, NativeSession, rhs_sensitivity
m = Model()
m.add_var('production', ub=100)
m.add_constraint({'production': 1}, '>=', 20, name='demand')
m.set_objective({'production': 5})
report = rhs_sensitivity(m, 'demand', 0.1, session_type=NativeSession)
print(report['slopes'])  # left/right objective change per unit: both 5
```

The helper shifts both finite bounds of one named row by ±delta and solves all three models. It exposes complete results/statuses and only calculates slopes when all three report optimality. Integer models are rejected. This is measured finite-perturbation response; it is not an analytic gradient, a basis-validity interval or a differentiable training layer. Distinct left/right responses can reveal a kink. It uses the existing solver's original-space final checks; it is not a second implementation of the verifier.

## Tests

`tests/test_extensions.py` checks native execution with subprocess launches prohibited, updates, rollback, malformed JSON/null arguments, handle lifecycle, analytical slopes and a kink, integer rejection, externally pinned hashes, byte tampering, rehashed corrupt objectives and archive path rejection. Existing planning/CLI/core/research tests exercise the shared-session refactor. GPU runtime and Windows ABI testing remain pending.

Validation on the development Mac: `cmake --build build --parallel 4` and `bash scripts/run_tests.sh build` passed, including core/research CTest, CLI/Python, 4 planning tests, 5 extension tests, benchmark evidence, 17 dashboard API tests and dependency policy. The refinery bundle passed fresh `VERIFIED_OPTIMAL` verification; the analytical sensitivity example returned left/right slopes of 5.0. Local generated artifacts are in `results/competitor-extensions-20260928/` (ignored by Git). No GPU or performance superiority claim follows from these checks.
