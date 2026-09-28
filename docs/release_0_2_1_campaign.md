# 0.2.1 frozen prototype campaign

Measured on 28 September 2026; source revision `0d93546`. This is a fixed nine-model regression campaign, not a broad industrial performance claim.

Budget: 30 seconds per solve; 1,000,000 NIRYUKTI iterations; one CPU thread; FP64; tolerance 1e-6; three repetitions; one discarded CUDA warmup per instance. All 108 measured rows are retained.

| Configuration | Verified continuous optimal runs | Incumbent-only runs | Other retained results |
|---|---:|---:|---|
| auto | 18 | 9 | {} |
| cpu | 18 | 9 | {} |
| cuda | 18 | 9 | {} |
| highs | 18 | 6 | {'NOT_SET': 3} |

The six continuous models passed independent KKT checks in every configuration/repetition. All three integer models yielded independently verified incumbents in NIRYUKTI. No complete integer search tree was independently replayed. HiGHS 1.15.1 retained `NOT_SET` on all three integer-dispatch MIQP repetitions; those failures remain in performance-profile denominators.

## E226 median end-to-end seconds

| Configuration | Seconds |
|---|---:|
| auto | 1.865687 |
| cpu | 2.014600 |
| cuda | 7.427836 |
| highs | 0.021340 |

Every E226 measured run passed independent optimality verification. HiGHS is substantially faster on this instance. CPU/CUDA configurations select methods automatically, so these measurements compare deployed configurations rather than identical kernels.

## Reproduce and inspect

Run the command in [campaign instructions](../benchmark/campaigns/README.md). Local raw evidence, offline HTML and performance profiles are under `results/prototype_0_2_1/`. That generated directory is intentionally excluded from Git; the configuration, runner and this summary are tracked. The binary and source are copied into each campaign, with input, source, hardware and command metadata retained.

Evidence checksum-manifest SHA-256: `93e3d1f85f6adb8aea92ce784fd176f1e75b8e13a3a2e8a6ef86d956a592a342`.

PyPI 0.2.1 is published: https://pypi.org/project/niryukti/0.2.1/. Release workflow 36406280403 built and tested the portable wheel/source distribution and npm artifact. npm publishing awaits account browser approval; do not treat it as published yet.
