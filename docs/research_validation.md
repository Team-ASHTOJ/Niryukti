# Research validation snapshot

CPU measurements on 2026-09-27, AppleClang Release, one thread, tolerance 1e-6, 100,000 iterations and two seconds per solve. Each configuration ran three times, sequentially. Every run is retained, including limits. These are small regression/ablation measurements, not general solver speed claims. MILP iteration counts include all node, repair and probe solves; continuous KKT does not certify MILP optimality.

Baseline source commit: `2045b07cf0e4b484f0e29b96b7465a981825171c`. The baseline executable was built before edits. New executables contain uncommitted changes; exact binary hashes are recorded below and in local manifests.

## Default and scaling comparison

All three runs had the same status in each cell. Counts are accepted iterations from the first run; time-limited counts can vary.

| Instance | Original default | Corrected default | Combined scaling |
| --- | --- | --- | --- |
| adlittle.mps | OPTIMAL / 46,200 | OPTIMAL / 59,900 | OPTIMAL / 10,000 |
| afiro.mps | OPTIMAL / 1,200 | OPTIMAL / 1,200 | OPTIMAL / 1,200 |
| e226.mps | ITERATION_LIMIT / 100,000 | ITERATION_LIMIT / 100,000 | OPTIMAL / 29,200 |
| israel.mps | OPTIMAL / 57,600 | OPTIMAL / 53,700 | OPTIMAL / 58,700 |
| refinery_large.mps | OPTIMAL / 1,500 | OPTIMAL / 1,500 | TIME_LIMIT / 3,400 |
| dispatch.json | OPTIMAL / 100 | OPTIMAL / 100 | OPTIMAL / 100 |
| refinery.json | OPTIMAL / 600 | OPTIMAL / 500 | OPTIMAL / 200 |
| scheduling.json | OPTIMAL / 13,600 | OPTIMAL / 14,700 | OPTIMAL / 11,700 |
| supply_chain.json | OPTIMAL / 3,300 | OPTIMAL / 3,300 | OPTIMAL / 3,300 |

The original and corrected defaults both solved 8/9 cases on every repetition. Combined scaling also solved 8/9, but changed which case failed: e226 reached verified optimality in 29,200 iterations (KKT 7.42e-7), while refinery_large reached the time limit. Combined scaling therefore remains opt-in. The stricter default acceptance is a correctness change and does not improve iteration counts on every instance.

## Other experiments

| Configuration | Instances | Result across three repetitions |
| --- | --- | --- |
| Halpern + combined scaling | adlittle, afiro, e226, israel, refinery | 3/5 solved each run; e226 and israel hit iteration limits |
| Reliability branching + default scaling | scheduling, supply_chain | 2/2 solved each run; probe overhead included |

Halpern uses norm-based initial weighting and damped weighting at epoch restarts. It remains experimental: this suite does not justify selecting it over PDHG. Earlier exploratory runs, including unit-weight Halpern and concurrently launched experiments, are archived locally and are excluded from this serial summary.

## Checks

- 240 existing core assertions and 191 research assertions passed on CPU.
- Release and ASan/UBSan builds passed CTest, CLI/Python and dependency-policy checks.
- Seven existing dashboard HTTP tests passed; their temporary localhost listener required running outside the restricted sandbox.
- Python compile checks and whitespace checks passed.
- No CUDA compiler/device or OpenMP runtime was available for execution validation. CUDA source changes and conditional parity tests still need a CUDA machine.

## Artifact identity

Raw local artifacts are in `results/research-serial-<configuration>/` (git-ignored). Manifests include dataset checksums, options, platform, compiler and binary hashes.

| Configuration | Binary SHA-256 |
| --- | --- |
| baseline | `1d570fd5bedaea6a877293379ae889a8ebfd5fa9cad512d010613454fe931295` |
| default | `ca57a4e2ce040d87addee9a045d7bff29a91b76d5e4c4b69148c5a3205027a8c` |
| combined | `ca57a4e2ce040d87addee9a045d7bff29a91b76d5e4c4b69148c5a3205027a8c` |
| halpern | `147a5acaa6cf799e8df52dab450efe64244afacf2a12bd7eec06d86bcf0a8199` |
| reliability | `ca57a4e2ce040d87addee9a045d7bff29a91b76d5e4c4b69148c5a3205027a8c` |

These tests reduce regression risk; they do not establish bug-free behavior or validate the unfinished features listed in [the integration record](research_features.md).

## Extended implementation measurements

The subsequent extension passed 240 core assertions, **1,495 research assertions**, CLI/Python tests, two performance-profile tests, and dependency checks in both Release and ASan/UBSan builds. CUDA remains unavailable. The dashboard tests above predate this extension; dashboard code was not changed.

The following serial measurements used the same 1e-6 tolerance, 100,000-iteration/two-second limits and three repetitions. The five LP instances are adlittle, afiro, e226, israel and refinery. Bundled settings are listed explicitly; improvements cannot be attributed to any single component without further ablations.

| Configuration | Solved per repetition | e226 accepted iterations / KKT |
| --- | --- | --- |
| Current default | 4/5 LPs | 100,000 / 3.88e-6 (limit) |
| rHPDHG + combined scaling + PID | 3/5 LPs | 100,000 / 1.03e-3 (limit) |
| Reflected r2HPDHG + combined scaling + PID | 5/5 LPs | 80,733 / 8.84e-7 |
| PDHG + combined scaling + PID + power(20) + polishing | 5/5 LPs | 33,200 / 3.00e-7 |
| Reliability + cuts + pump/RINS | 2/2 MILPs | Not applicable |

Across the complete previous nine-instance regression selection, the current default again solved 8/9 on all repetitions; e226 remains the sole limit. The large refinery case was measured separately after the eight other cases and solved on every run. No default algorithm selection was changed.

The polishing bundle spent 2,700 of its 33,200 e226 iterations in auxiliary solves. The MILP bundle increased scheduling work from 14,700 to 75,800 iterations and supply-chain work from 3,300 to 8,800. Those two examples generated no cuts because they lack eligible knapsack rows; cut generation and validity were separately exercised in research tests. This suite supports keeping all experiments opt-in.

Artifacts: `results/research-serial-{default2,rhpdhg,r2hpdhg,polishing,mip2}/`, `results/research-default2-large/`, and a five-LP comparison profile in `results/research-profile-lp/profile.html`. Every repetition and limit is retained; profile selection is explicit in `selection.json`. Measured binary SHA-256: `db8d3b140bb8e9465dd474bd027608a64e0f1ffb5ae6940e627201f6b6a82ed7`.
