# Phase 1 validation snapshot

Local validation on 2026-09-11. This is a prototype smoke/scalability campaign, not comprehensive industrial validation.

Hardware: Intel Core i7-14650HX, NVIDIA RTX 4060 Laptop GPU, approximately 16 GB host RAM. Release C++/CUDA build; FP64.

Core: 160 assertions with CUDA, including 25 small MILPs compared against exhaustive enumeration. CPU-only and address/undefined-behavior sanitizer suites passed. CLI, malformed-input, Python API, verification and warm-start integration tests passed. Test logs are in `results/validation/`.

Measured executable SHA-256: `fce0862a20d6d16aab1438d9f2961c4cb4821dbe57e72a98d367afdbafeada7e`. All reports below refer to this executable; raw folders preserve a copy.

| Suite | Instance | Backend | Optimal runs | Median end-to-end seconds | Statuses |
|---|---|---|---:|---:|---|
| demo | refinery.json | cpu | 3/3 | 0.027446 | OPTIMAL |
| demo | refinery.json | cuda | 3/3 | 0.260335 | OPTIMAL |
| demo | refinery.json | highs | 3/3 | 0.001686 | OPTIMAL |
| demo | dispatch.json | cpu | 3/3 | 0.001188 | OPTIMAL |
| demo | dispatch.json | cuda | 3/3 | 0.137174 | OPTIMAL |
| demo | dispatch.json | highs | 3/3 | 0.001256 | OPTIMAL |
| demo | supply_chain.json | cpu | 3/3 | 0.371452 | OPTIMAL |
| demo | supply_chain.json | cuda | 3/3 | 4.997558 | OPTIMAL |
| demo | supply_chain.json | highs | 3/3 | 0.016787 | OPTIMAL |
| demo | afiro.mps | cpu | 3/3 | 0.012184 | OPTIMAL |
| demo | afiro.mps | cuda | 3/3 | 0.250951 | OPTIMAL |
| demo | afiro.mps | highs | 3/3 | 0.002254 | OPTIMAL |
| netlib | afiro.mps | cpu | 3/3 | 0.016585 | OPTIMAL |
| netlib | afiro.mps | cuda | 3/3 | 0.235784 | OPTIMAL |
| netlib | afiro.mps | highs | 3/3 | 0.002100 | OPTIMAL |
| netlib | adlittle.mps | cpu | 3/3 | 0.062688 | OPTIMAL |
| netlib | adlittle.mps | cuda | 3/3 | 0.613142 | OPTIMAL |
| netlib | adlittle.mps | highs | 3/3 | 0.003642 | OPTIMAL |
| netlib | israel.mps | cpu | 0/3 | 0.768968 | ITERATION_LIMIT |
| netlib | israel.mps | cuda | 0/3 | 3.066576 | ITERATION_LIMIT |
| netlib | israel.mps | highs | 3/3 | 0.009711 | OPTIMAL |
| netlib | e226.mps | cpu | 0/3 | 0.918477 | ITERATION_LIMIT |
| netlib | e226.mps | cuda | 0/3 | 3.256076 | ITERATION_LIMIT |
| netlib | e226.mps | highs | 3/3 | 0.033458 | OPTIMAL |
| netlib | scheduling.json | cpu | 0/3 | 7.314454 | UNKNOWN |
| netlib | scheduling.json | cuda | 0/3 | 10.025897 | TIME_LIMIT |
| netlib | scheduling.json | highs | 3/3 | 0.031266 | OPTIMAL |
| scalability-final | refinery_large.mps | cpu | 3/3 | 11.162786 | OPTIMAL |
| scalability-final | refinery_large.mps | cuda | 3/3 | 7.707421 | OPTIMAL |
| scalability-final | refinery_large.mps | highs | 3/3 | 0.794403 | OPTIMAL |

Main demo: 36 measured runs, all optimal. The broader suite retains VANTAGE iteration limits on `israel` and `e226`; scheduling retains unresolved bounds / time limits. HiGHS solves these cases.

The large synthetic instance has 46,720 variables and 186,880 nonzeros. It is largely separable by period. Three measured runs, one discarded CUDA warmup, and four CPU threads for both VANTAGE and HiGHS. Read residuals with objectives: approximate feasible objectives can fall slightly below the exact optimum.

On this synthetic instance the median own-CPU/GPU end-to-end ratio was 1.45x. HiGHS remained faster (0.794s versus 7.707s CUDA). This is not a claim about general industrial performance.

The 5% demand-change demo used 3600 iterations cold and 1700 warm, with independently verified and agreeing objectives.

Reports: [main demo](../results/demo/index.html), [public and harder cases](../results/netlib/index.html), [synthetic scalability](../results/scalability-final/index.html). Raw files, exact commands, residuals and per-instance CSV are linked from each report. Earlier experimental runs were archived rather than erased.
