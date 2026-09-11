# Phase 2 measured results

Generated from retained raw runs by `benchmark/compare.py`. All selected cases appear, including limits and regressions.

Settings: 3 measured runs per case/backend, one separate CUDA warmup, 4 CPU threads, tolerance 1e-06, time limit 20.0 s, 100,000 continuous iterations per solve. Times below are median end-to-end solver time, including parsing, setup, transfer and verification. Each MILP node has its own continuous iteration cap.

CPU: Intel(R) Core(TM) i7-14650HX. GPU: name, driver_version, memory.total [MiB], power.limit [W] / NVIDIA GeForce RTX 4060 Laptop GPU, 610.57.04, 8188 MiB, [N/A]

These measurements were taken on a shared development laptop, not an isolated performance laboratory. Small timings and GPU startup times are sensitive to system activity. They do not establish industrial competitiveness.

| Instance | Backend | Before status | Before s | After status | After s | Before/after speedup¹ |
| --- | --- | --- | ---: | --- | ---: | ---: |
| refinery.json | cpu | OPTIMAL | 0.028150 | OPTIMAL | 0.001688 | 16.68× |
| refinery.json | cuda | OPTIMAL | 0.348018 | OPTIMAL | 0.126260 | 2.76× |
| dispatch.json | cpu | OPTIMAL | 0.000865 | OPTIMAL | 0.000384 | 2.25× |
| dispatch.json | cuda | OPTIMAL | 0.128934 | OPTIMAL | 0.109875 | 1.17× |
| supply_chain.json | cpu | OPTIMAL | 0.535395 | OPTIMAL | 0.007192 | 74.44× |
| supply_chain.json | cuda | OPTIMAL | 5.197347 | OPTIMAL | 0.217504 | 23.90× |
| scheduling.json | cpu | UNKNOWN | 9.045273 | OPTIMAL | 0.031140 | — |
| scheduling.json | cuda | TIME_LIMIT | 20.026283 | OPTIMAL | 0.350435 | — |
| afiro.mps | cpu | OPTIMAL | 0.017994 | OPTIMAL | 0.002980 | 6.04× |
| afiro.mps | cuda | OPTIMAL | 0.273493 | OPTIMAL | 0.132353 | 2.07× |
| adlittle.mps | cpu | OPTIMAL | 0.107737 | OPTIMAL | 0.095760 | 1.13× |
| adlittle.mps | cuda | OPTIMAL | 0.781229 | OPTIMAL | 1.821139 | 0.43× |
| israel.mps | cpu | ITERATION_LIMIT | 0.815087 | OPTIMAL | 0.384378 | — |
| israel.mps | cuda | ITERATION_LIMIT | 2.882654 | OPTIMAL | 2.382684 | — |
| e226.mps | cpu | ITERATION_LIMIT | 0.806558 | ITERATION_LIMIT | 0.790133 | — |
| e226.mps | cuda | ITERATION_LIMIT | 3.579817 | ITERATION_LIMIT | 4.228184 | — |
| refinery_large.mps | cpu | OPTIMAL | 18.128155 | OPTIMAL | 1.297431 | 13.97× |
| refinery_large.mps | cuda | OPTIMAL | 11.785054 | OPTIMAL | 1.065444 | 11.06× |

¹Speedup is shown only when every repetition in both campaigns is optimal. Values below one are regressions. A limit time is not a solve time.

| Solver | Before: cases optimal in every run | After: cases optimal in every run |
| --- | ---: | ---: |
| cpu | 6/9 | 8/9 |
| cuda | 6/9 | 8/9 |
| highs | Not measured | 9/9 |

Worst relative objective difference from the matched HiGHS reference among optimal VANTAGE runs: **1.26488e-07**.

For LP/diagonal QP, optimal status also requires original-space feasibility, stationarity and gap checks at the requested tolerance. For MILP, it requires a feasible integer incumbent and conservative global-bound gap. A rounded incumbent’s continuous KKT value is not an integer optimality test.

## Provenance

- Before: [results/phase2-baseline](../results/phase2-baseline/index.html), timestamp `2026-09-11T10:49:40+0530`, executable SHA-256 `fce0862a20d6d16aab1438d9f2961c4cb4821dbe57e72a98d367afdbafeada7e`. Raw commands, per-run JSON, stdout/stderr, model checksums, hardware and build metadata are retained in that directory.
- After: [results/phase2-final](../results/phase2-final/index.html), timestamp `2026-09-11T10:54:12+0530`, executable SHA-256 `e406b434e217ff1c3e8353f678f552c4c41e174bd23f4f8c43219589d82e1144`. Raw commands, per-run JSON, stdout/stderr, model checksums, hardware and build metadata are retained in that directory.

## Separate extended-budget experiment

The same executable was also measured with 200,000 VANTAGE iterations, 4 threads, tolerance 1e-06, 20.0 s and 3 repetitions. This changes the iteration budget, so these results do **not** replace the standard-budget statuses above. [Raw extended campaign](../results/phase2-e226-extended/index.html).

| Instance | Solver | Status across repetitions | Median end-to-end s | Median accepted iterations |
| --- | --- | --- | ---: | ---: |
| e226.mps | cpu | OPTIMAL | 2.236620 | 138,100 |
| e226.mps | cuda | OPTIMAL | 6.159601 | 144,479 |
| e226.mps | highs | OPTIMAL | 0.019490 | — |

## Interpretation and remaining work

This is a selected nine-case prototype suite, not a representative estimate of industrial solve rate. Synthetic refinery periods are largely independent. The primary improvements in 0.2 are local adaptive step control, cached verification, conservative integer-bound propagation and replayable box/row infeasibility certificates. General sparse QP, stronger restart merit functions, wider public benchmark coverage, persistent GPU contexts, cuts and general recession certificates remain future work.
