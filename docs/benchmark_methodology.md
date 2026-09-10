# Benchmark methodology

`benchmark/run.py` launches every configured solver in a separate process. The CPU and GPU comparisons use the same VANTAGE algorithm, tolerance, time limit and thread setting. HiGHS receives the same thread/time settings and approximately corresponding feasibility tolerances. Its algorithm and stopping conventions differ.

Default measurement is three runs, with a discarded CUDA warmup process before the three GPU measurements. Every process constructs a new CUDA context; the warmup warms device/OS state, not a persistent solver session. Reports retain warmup raw files separately. Process isolation includes runtime setup in end-to-end timing and prevents solver imports from entering the production API.

Previous reports are archived automatically when an output directory is reused. The measured VANTAGE executable is saved with the raw files, alongside its hash.

Stored data:

- Per-run status, objective, residuals, iterations, dimensions and nonzeros.
- Internal timing, end-to-end timing including parsing/runtime/device setup, and whole-process wall time including executable/Python startup.
- Commands, return codes, stdout, stderr and full structured outputs.
- SHA-256 of each dataset and executable, platform/device information, parameters and baseline version.
- An uncommitted marker when no git commit exists.

Iteration time includes a CUDA chunk synchronization, but excludes monitoring and setup. Monitoring copies full current and average vectors to the CPU and recomputes diagnostics. `transfer_setup_seconds` covers backend construction/upload; monitoring downloads are included in verification time. These are not CUDA-event per-kernel profiles. Peak GPU memory is estimated before execution, not measured continuously. HiGHS RSS comes from its separate process.

Every selected instance appears, including parse errors, unavailable backends, numerical failures, process timeouts and solver limits. Medians include recorded times of unsuccessful runs; they are not solved-only performance scores. Read statuses before comparing bars. Do not turn a timeout into a solved-instance speed ratio. Do not merge exploratory runs made during compilation with controlled final runs.

`accuracy.relative_gap` is VANTAGE's normalized complementarity/gap diagnostic; read the mathematical definition. Approximate primal objectives can fall below the exact optimum due to small feasibility errors. Baseline vectors are checked through the independent original-space verifier. MILP dual/KKT numbers are not integer optimality certificates; inspect the incumbent, bound and MIP gap.

The cached Netlib set is a **small smoke suite**, not a complete benchmark campaign. See `datasets/manifest.json` for pinned upstream sources and checksums. Synthetic refinery instances measure sparse throughput and application plumbing; repeated periods are largely independent and are not representative of difficult coupled industrial scheduling. No real MRPL data is included.

For reports, use an idle machine, a fixed Release binary, identical resource limits and enough repetitions to identify noise. The current report provides per-instance bars and tables. Shifted geometric means, Dolan–Moré profiles, memory profiling, CUDA-event breakdowns, full MIPLIB/QPLIB campaigns and benchmark caching remain Phase 2 work.
