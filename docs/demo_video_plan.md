# NIRYUKTI prototype demo video

The prototype is ready to record a working SIH demonstration on the validated
NVIDIA laptop. This is readiness for a research-prototype demo, not completion of
every roadmap item or proof of commercial-solver performance. Check the official
submission limit before exporting; this plan targets about 3 minutes 30 seconds.

## Prepare before recording

1. Build the current release source; run `bash scripts/run_tests.sh build`.
2. Run `NIRYUKTI_DEMO_DIR=results/video-take ./scripts/recording_demo.sh`.
   This produces actual solutions, independent verification, HTML reports,
   CUDA telemetry and a barrier checkpoint/resume example without network access.
3. Launch the dashboard with `NIRYUKTI_BINARY="$PWD/build/niryukti" python3
   dashboard/server.py --host 127.0.0.1 --port 8080`.
4. Open the refinery model, report and solver source beforehand. Use a readable
   browser zoom and hide credentials, account approval screens and unrelated tabs.
5. Keep the locally verified files available if a live solve stalls. Identify
   saved results as saved runs; do not depict a recording as a live solve.
6. Preinstall packages; npm compiles C++ and should not consume recording time.

## Shot sequence and suggested narration

| Time | Screen/action | Narration |
|---|---|---|
| 0:00–0:20 | Title and refinery planning overview | “NIRYUKTI is an independently implemented sparse optimization engine for linear, supported convex quadratic and mixed-integer models.” |
| 0:20–0:40 | Brief source architecture view | “The solver core implements its own optimization algorithms. CPU and NVIDIA CUDA paths share sparse models and independent verification. External solvers are used only for comparison.” |
| 0:40–1:20 | Dashboard refinery model; run automatic selection, then explicit CUDA solve | “This synthetic refinery blending case uses availability, capacity and product constraints. Automatic selection can choose CPU for small models. Here we explicitly demonstrate GPU sparse computation and presolve.” |
| 1:20–1:45 | Actual status, objective, residuals; verify and open/download report | “Every continuous result is checked against the original model. This screen shows measured residuals and the objective; the report retains them for inspection.” |
| 1:45–2:10 | Supply-chain or scheduling integer demo | “The integer engine branches, propagates bounds and separates cuts. We show the incumbent and search gap. Independent incumbent feasibility is distinct from replaying a complete optimality proof.” |
| 2:10–2:35 | Terminal: barrier limited run, then resume saved state | “The checkpoint stores actual solver state. Resuming continues that state and the final candidate is independently verified.” |
| 2:35–2:55 | Python/Node usage and CLI/report commands | “The same core is available through pip, npm, the CLI and a local authenticated API, with portable offline reports.” |
| 2:55–3:15 | Optional frozen comparison table, with version clearly shown | “Our frozen earlier-version campaign retains successes and failures. GPU is not always faster, and these figures do not measure the latest update.” Omit this shot if the time limit is tighter. |
| 3:15–3:30 | Final architecture and limitations | “This is a transparent, extensible research prototype. The demonstrated paths are tested; broader device control and production numerical hardening remain future work.” |

## Evidence to show

For the latest recorded CUDA refinery run, original-space normalized KKT error was
1.685e-8, with 60 bound-tightening derivations and 16 removed rows. Use the values
from the actual video take if they differ. Do not show continuous LP KKT as a
proof of integer optimality. Do not label synthetic examples as actual MRPL data.

Local validation includes 17,461 CUDA research assertions, the full regression
suite, CPU sanitizer checks and engine/tree checkpoint tests. Do not claim
second-laptop or NVIDIA-container validation, unrestricted nonconvex support,
complete GPU presolve/control, or superiority over mature solvers.

## Recording checklist

- Readable model name, selected method/device, status and residuals.
- One real CUDA solve and original-model verification.
- One feasible integer plan with honest proof/gap scope.
- One checkpoint/resume and one generated report.
- No installation/authentication interruptions or fabricated benchmark graphics.
- Trim idle terminal output; preserve the order and meaning of live actions.
