# NIRYUKTI implementation record — 2026-09-28

## Implemented in this round

- Pure structural algorithm/backend advisor with injectable hardware information:
  compact LP simplex, basis reoptimization, guarded small coupled-QP barrier,
  sparse first-order fallback, GPU free-memory guard, and explicit overrides.
- Auto recovery reserves time for independently verified PDHG after numerical or
  limit outcomes from simplex/barrier. First-order controls and checkpoints stay
  on a compatible method.
- Python HTTP API with bearer authentication, inline model requests, bounded
  payloads, worker slots and solve budgets. Synchronous execution; disconnects
  do not cancel existing requests.
- Printable offline Python/Node HTML reports with escaped text and explicit
  distinction between displaying results and verifying them.
- Public NIRYUKTI C++ include and C ABI names, native library filename, CLI and
  environment variables. Legacy internal names remain compatibility aliases.
- Comprehensive Python/npm README examples, root API/report documentation and
  reproducible NVIDIA laptop validation script.
- Docker service profile and correct benchmark files included in images.

## Validation

The complete CPU suite and CUDA-enabled suite passed after advisor policy
corrections. API tests exercise authenticated requests, actual LP solves,
resource-policy rejection, reporting and text escaping. Node solve/report and
Python model examples were smoke tested. Docker Compose configuration validates.
The NVIDIA laptop script passed locally: all four demo models were solved and
independently verified on CPU, CUDA and auto; evidence is retained in
`results/niryukti_improvements_20260928/`. A second physical laptop and
standard NVIDIA-container runtime remain external validation requirements.

Public registries still contain 0.2.0. Source metadata is 0.2.1; new source
features are not available from those registries until the next release passes
portable packaging checks and is published.

## Remaining work, not completed claims

- Fully device-resident adaptive restart/control and complete GPU presolve
  compaction; current acceleration is restricted.
- Direct GPU barrier factorization; current experimental GPU Newton path uses
  CPU KKT assembly and iterative sparse solves.
- General implication/dual conflict analysis and unrestricted global/tableau
  cut machinery; current cuts/conflicts have restricted applicability.
- Unrestricted large singular PSD certification and hard large-MIP performance.
- Full simplex/barrier/concurrent checkpoint state; available first-order and
  MIP checkpoint implementations do not imply coverage of every engine.
- Portable 0.2.1 release checks/publication, standard NVIDIA container runtime,
  and reproduction on a second physical laptop.

## Next development sequence

1. Run the laptop validation script on a second NVIDIA machine and retain its
   manifest, verification results and failures. Validate container runtime there.
2. Publish tested packages; compare structural auto selection against explicit
   engines with equal time/thread budgets and retained unsuccessful cases.
3. Profile device monitoring/control before changing CUDA kernels. Implement
   device restart and GPU presolve incrementally with differential tests.
4. Strengthen PSD certification and integer proof/cut infrastructure, verifying
   each transformation independently before enabling it by default.
5. Improve barrier factorization and checkpoint coverage only with reproducible
   numerical regression evidence. Keep experimental methods opt-in where needed.

No universal speedup or complete industrial-solver parity is claimed.
