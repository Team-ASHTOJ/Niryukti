# NIRYUKTI 0.2.2 release

Release source commit: `3930284`, incorporating solver changes from `460b703`
and the preceding checkpoint/sparse-PSD source updates.

## Included changes

- Actual simplex basis, barrier Newton-state and portfolio-child checkpoints.
- Guarded AMD ordering for supported singular sparse PSD validation.
- GPU-proposed continuous bounds with independent derivation replay and original
  dual reconstruction; CUDA retained-CSR count/scan/scatter compaction.
- Replay-proved general bound conflicts with checkpoint persistence.
- Integer-lattice cuts in the existing cut-pool pipeline.
- Service version metadata aligned with the Python package.
- Updated Python/npm documentation and a locally verified recording plan.

Python wheels bundle the CPU executable and native library. The npm distribution
contains source and builds locally. Neither default CPU package silently installs
or enables CUDA; use an explicit CUDA source build for GPU paths.

## Build and publication evidence

[Release workflow 36429315322](https://github.com/shoaib2000857/Vantage/actions/runs/36429315322)
built and passed Python wheel/source checks and the npm install/build/smoke test.
PyPI publication succeeded and public metadata exposes 0.2.2.
The npm publishing job failed because its relative tarball path was interpreted
as a Git package reference. The workflow path is corrected to `./npm-dist/*.tgz`.
Local publication of the exact tested artifact requires fresh npm account 2FA.

Tested npm tarball SHA-256:
`4408feb370af0438c034d305e71a7e7a7cc55a66972ed720faf03c993d4930d3`.

CPU correctness workflow passed for normal and sanitizer builds. Local CUDA
CTest passed 4/4, and the preceding solver validation passed 17,461 CUDA research
assertions plus regression/checkpoint checks. The frozen benchmark campaign
remains the earlier 0.2.1 source; its timings are not relabeled as 0.2.2 results.

## Install

```bash
python -m pip install --upgrade niryukti==0.2.2
# After npm publication is approved and publicly visible:
npm install niryukti@0.2.2
```

## Demo readiness

[Video plan and recording checklist](demo_video_plan.md). The validated laptop
is ready to demonstrate the tested research-prototype paths. Fully device-resident
control, general dual-proof conflicts and production GPU barrier remain pending;
second-host/container validation is not implied by package publication.
