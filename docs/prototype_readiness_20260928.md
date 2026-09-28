# Prototype readiness audit — 28 September 2026

## Latest changes reviewed

`main` was already synchronized with commit `58db13d` (Auto Solver analysis,
dry-run output, configuration telemetry and benchmark reporting). No remote
changes were discarded. Certificate replay from the preceding changes remains
part of the regression suite.

## Confirmed issues corrected

| Issue | Correction | Evidence |
|---|---|---|
| GitHub sanitizer job aborted when Python loaded the native library | Preload linked ASan and C++ runtimes for native Python hosts; use portable `ldd`/`awk` discovery | Standalone C++ sanitizer checks remain enabled; GitHub matrix rerun |
| Analyze/dry-run accepted invalid numeric and enum options | Shared validation with the solver | Regression cases for NaN tolerance, negative budget, unknown method/scaling, zero threads |
| Concurrent engines exceeded small requested CPU thread budgets | Cap engine count, divide the budget, initialize OpenMP inside each worker | One-thread portfolio regression plus existing LP/QP portfolio cases |
| Malformed API authorization/numeric inputs could terminate a handler | Byte-based constant-time comparison and finite JSON number validation | Unicode header returns 401; exponent overflow returns 400; service remains alive |
| Report renderer failed on malformed accuracy sections | Treat non-object accuracy as unavailable | Report regression with `accuracy: null` |
| Generated browser dependencies were committed | Untrack `node_modules`; preserve local files and lockfile | Ignore rule already present; reproducible install remains possible |
| Distribution checks did not run on main pushes | Add path-filtered main build/install smoke jobs | Wheel/native/API/report and npm clean-install workflow |

The C++ sanitizer tests retain ordinary leak checks. Leak reporting is disabled
only for Python processes hosting the instrumented library, because CPython
allocations are not the native-library leak oracle; ASan/UBSan instrumentation
remains active there.

## Prioritized remaining work

### Prototype readiness tasks

1. **Maintain green CI and distribution smoke jobs.** The GitHub CPU matrix
   for commit `734adfb` passed both sanitizer OFF and ON (run `36397746913`).
   Distribution smoke results are recorded below; a local build alone is not
   portable-registry evidence.
2. **Publish tested 0.2.1 Python/npm packages.** Public 0.2.0 predates the API,
   reports, and latest advisor changes. Keep examples/version claims aligned.
   Publishing is separate from build-only CI; npm trusted-publisher setup may
   still require owner action.
3. **Reproduce on a second NVIDIA laptop.** Run
   `scripts/validate_cuda_laptop.sh`; retain hardware manifest, verification
   results and unsuccessful runs. Test the normal `--gpus all` Docker runtime
   with NVIDIA Container Toolkit. Local GPU success is not second-host evidence.
4. **Freeze an honest numerical demo campaign.** E226 and the large refinery LP
   fail 1e-6 verification within the friend's recorded five-second campaign.
   That is a convergence/performance limitation under that budget, not evidence
   that every solve fails. The follow-up E226 solve below reached verified
   optimality after raising the iteration cap. Establish outcomes with realistic limits and
   retain explicit failures. No blanket competitiveness claim versus HiGHS.
5. **Maintain the distinction between MIP solver status and replayable proof.**
   The standalone certificate can verify incumbents and available relaxation
   bounds; it does not replay a complete branch-and-cut tree. Some telemetry
   OPTIMAL results therefore have only independently replayable FEASIBLE evidence.
   Keep this visible in submission material and reports.
6. **Validate presentation/platform paths after final changes.** Refresh browser
   and mobile live-solve/cancellation checks, offline demo and updated container
   images. Synchronous HTTP disconnect does not cancel an API solve; dashboard
   job cancellation is a separate feature.

### Known capability limits / larger implementation projects

- Complete GPU presolve compaction and reduction families with reversible primal
  and dual postsolve provenance. Continuous GPU bounds currently initialize LP/QP
  points rather than silently changing their dual box; MIP nodes use valid tightening.
- Fully device-resident restart/weight/control decisions. Device trial acceptance,
  monitoring and resident restart copies exist; host control still participates.
- General implication-graph/dual conflicts and unrestricted tableau separators.
  Current replay-minimized binary clauses and MIR/cover/clique cuts have defined scope.
- Unrestricted large singular PSD QP certification and difficult large MILP/MIQP
  performance; current supported sparse convex classes are guarded.
- Direct GPU sparse barrier factorization and high-accuracy robustness. Current
  barrier is experimental and has size/fill guards and CPU KKT assembly.
- Checkpoint state for simplex/barrier/concurrent engines. MIP tree and first-order
  state saving do not imply that every algorithm supports exact resume.
- Strict cancellation/time budgeting inside every preprocessing/factorization
  operation. Individual synchronous operations can exceed a soft solver budget.
- Broad external benchmarking, calibration of auto thresholds, and measured
  end-to-end speedups rather than assuming an advisor choice is always fastest.

AMD/HIP is intentionally deferred at the user's request.

## Assessment

The repository contains a functioning independent optimization prototype with
CPU/CUDA solving, model analysis, supported LP/QP/MIP algorithms, verification,
HTTP API, CLI, reports, dashboard, packages and validation scripts. It is not a
finished commercial solver. The items above separate evidence/release tasks from
research algorithms, so prototype submission need not depend on claiming that
all future research work is complete.


## Follow-up evidence

- Local CPU: 15,189 research assertions; CUDA RTX 4060: 16,382 research
  assertions. All CTest/CLI/Python/API/dashboard/certificate/conflict checks passed.
- Headless Chrome browser check passed desktop/mobile navigation, search,
  details, live solve, import and download with no page errors. The local browser
  installation remains present despite untracking generated dependencies.
- E226 with `--method auto --device cpu --threads 1 --time-limit 15` and the
  default 100,000-iteration cap returned ITERATION_LIMIT with KKT approximately
  1.61e-4. With `--time-limit 30 --iterations 1000000`, the same current solver
  returned OPTIMAL with KKT 9.701436868e-7 after 170,241 total iterations.
  Standalone verification returned VERIFIED_OPTIMAL. This is one local numerical
  recovery test, not evidence of baseline speed superiority.

```bash
./build/niryukti solve datasets/e226.mps --method auto --device cpu \
  --threads 1 --time-limit 30 --iterations 1000000 --json-out e226-result.json
./build/niryukti verify datasets/e226.mps e226-result.json
```

Raw local audit logs and model-result verification are retained under
`results/prototype_audit_20260928/` (ignored run artifacts).

GitHub distribution smoke run `36397747006` on `734adfb` also passed: wheel
build/metadata check, clean installation, subprocess/native APIs, API/report
imports and npm source installation/smoke tests. The CI failures described above
are resolved for this code revision. This build/install validation did not
publish a new registry release.
