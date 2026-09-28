# Benchmark methodology

`benchmark/run.py` launches every configured solver in a separate process. The CPU and GPU comparisons use the same NIRYUKTI algorithm, tolerance, time limit and thread setting. HiGHS receives the same thread/time settings and approximately corresponding feasibility tolerances. Its algorithm and stopping conventions differ.

Default measurement is three runs, with a discarded CUDA warmup process before the three GPU measurements. Every process constructs a new CUDA context; the warmup warms device/OS state, not a persistent solver session. Reports retain warmup raw files separately. Process isolation includes runtime setup in end-to-end timing and prevents solver imports from entering the production API.

Previous reports are archived automatically when an output directory is reused. The measured NIRYUKTI executable is saved with the raw files, alongside its hash.

Stored data:

- Per-run status, objective, residuals, iterations, dimensions and nonzeros.
- Internal timing, end-to-end timing including parsing/runtime/device setup, and whole-process wall time including executable/Python startup.
- Commands, return codes, stdout, stderr and full structured outputs.
- SHA-256 of each dataset and executable, platform/device information, parameters and baseline version.
- An uncommitted marker when no git commit exists.

Iteration time includes a CUDA chunk synchronization, but excludes monitoring and setup. Monitoring copies full current and average vectors to the CPU and recomputes diagnostics. `transfer_setup_seconds` covers backend construction/upload; monitoring downloads are included in verification time. These are not CUDA-event per-kernel profiles. Peak GPU memory is estimated before execution, not measured continuously. HiGHS RSS comes from its separate process.

Every selected instance appears, including parse errors, unavailable backends, numerical failures, process timeouts and solver limits. Medians include recorded times of unsuccessful runs; they are not solved-only performance scores. Read statuses before comparing bars. Do not turn a timeout into a solved-instance speed ratio. Do not merge exploratory runs made during compilation with controlled final runs.

`accuracy.relative_gap` is NIRYUKTI's normalized complementarity/gap diagnostic; read the mathematical definition. Approximate primal objectives can fall below the exact optimum due to small feasibility errors. Baseline vectors are checked through the independent original-space verifier. MILP dual/KKT numbers are not integer optimality certificates; inspect the incumbent, bound and MIP gap.

The cached Netlib set is a **small smoke suite**, not a complete benchmark campaign. See `datasets/manifest.json` for pinned upstream sources and checksums. Synthetic refinery instances measure sparse throughput and application plumbing; repeated periods are largely independent and are not representative of difficult coupled industrial scheduling. No real MRPL data is included.

For reports, use an idle machine, a fixed Release binary, identical resource limits and enough repetitions to identify noise. The current report provides per-instance bars and tables. Shifted geometric means, Dolan–Moré profiles, memory profiling, CUDA-event breakdowns, full MIPLIB/QPLIB campaigns and benchmark caching remain Phase 2 work.

## Phase 2 before/after comparison

Keep the previous executable before rebuilding. Run the same explicit model list, repetitions, threads, tolerance and time limit against the preserved executable and then against the updated executable. Do not compile or run correctness/browser tests concurrently with the measured campaigns.

The current nine-case list is `examples/refinery.json examples/dispatch.json examples/supply_chain.json examples/scheduling.json datasets/afiro.mps datasets/adlittle.mps datasets/israel.mps datasets/e226.mps datasets/refinery_large.mps`.

```bash
# Use the same model list for both commands below.
.venv/bin/python benchmark/run.py MODEL_LIST \
  --binary results/phase2/before/vantage --solvers cpu,cuda \
  --runs 3 --threads 4 --time-limit 20 --output results/phase2-baseline
.venv/bin/python benchmark/run.py MODEL_LIST \
  --solvers cpu,cuda,highs --runs 3 --threads 4 --time-limit 20 \
  --output results/phase2-final
python3 benchmark/compare.py results/phase2-baseline results/phase2-final
```

Replace `MODEL_LIST` with the explicit list above. The generated `docs/phase2_results.md` includes all cases and links to raw artifacts. Comparison validates equal dataset checksums and common measurement settings. The dashboard uses a completed `phase2-final` campaign when present and otherwise retains the earlier campaigns. A summary file marks completion; partial folders must not replace the displayed comparison.

For submission archiving, preserve both result directories and the source tree corresponding to each executable. A dirty git commit ID alone does not identify exact source contents. The retained executable hash identifies what was measured, even when uncommitted source changes exist.

An explicitly separate iteration-budget experiment can be included without changing the standard statuses:

```bash
.venv/bin/python benchmark/run.py datasets/e226.mps --runs 3 --threads 4 \
  --iterations 200000 --time-limit 20 --output results/phase2-e226-extended
python3 benchmark/compare.py results/phase2-baseline results/phase2-final \
  --extended results/phase2-e226-extended
```


## Automatic solver comparison

The benchmark runner accepts `--solvers auto,cpu,cuda,highs`. `auto` uses the production advisor with `method=auto` and `device=auto`; CPU and CUDA use the `--method` argument and force their named backends. All attempts are retained, and NIRYUKTI results receive separate-process original-model verification. Compare status and verification before timing. Current checked-in datasets form a small smoke suite, CUDA may be unavailable, and no universal performance claim follows from it. See [Auto Solver benchmark instructions and limitations](auto_solver.md).
