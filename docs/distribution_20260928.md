# Distribution and integration — 28 September 2026

## Integrated upstream

Working branch: `main`. Fast-forwarded local `a21567f` to `4c0f066` after fetching
all remotes. Friend's commits `84bcc7e` and `4c0f066` were on `origin/main`;
`origin/frontend` and `origin/master` were older. No merge conflicts or local
changes were discarded. Replanning, C ABI, native sessions, measured sensitivity,
stable-plan helpers and evidence bundles are retained.

## Distribution architecture

* Python: scikit-build-core compiles our CPU engine and C ABI. Platform wheels
  include the executable, shared library and numerical dependency notices.
  `NativeSession` loads that library directly; the existing `Model.solve` API
  invokes the bundled executable. An installed `vantage` command forwards CLI
  arguments. Explicit binary/library overrides remain supported for CUDA.
* npm: asynchronous `solve(pathOrModel, options)` with TypeScript declarations,
  AbortSignal cancellation, temporary-file cleanup and bounded error output.
  The source tarball includes only required C++ sources and numerical headers;
  local CMake compilation replaces unverified remote binary downloads. Some npm
  versions require explicit install-script approval or running `install.js`.
* Docker: CPU/CUDA recipes now include the C library and Python API, with native
  paths configured. Existing Studio compose configuration remains available.
* Arch: a draft local `PKGBUILD` builds main from source. It has not been uploaded
  to AUR, built with makepkg, or added to an official pacman repository. There is
  no public apt repository. These are secondary distribution tasks.
* CI: a manually runnable/PR-triggered Linux workflow builds Python wheel/sdist,
  checks metadata, performs out-of-checkout installed API solves, stages npm,
  compiles its engine, checks cancellation, and uploads artifacts. No automated
  registry publishing or credentials are embedded.

The package default is FP64 CPU with OpenMP off to avoid an extra wheel runtime
dependency. Source users may opt into CUDA/OpenMP. Linux host wheels are not
manylinux releases: a portable public wheel campaign needs manylinux builds,
ABI/dependency audits and additional supported-platform validation. Source npm
installation needs CMake 3.24+, a C++20 compiler and Node 18+.

## Release gates

The project now uses AGPL-3.0-only following the user’s request to prevent
covered closed-source redistribution; vendored dependencies retain their own notices.
No PyPI/npm upload has occurred. npm authentication is absent on this machine.
Configure registry ownership and publishing credentials/trusted publishing only
after reviewing concrete artifacts. Do not describe
`pip install vantage-opt` or `npm install vantage-opt` as publicly available yet.

## Reproduce

```sh
python -m pip install build twine
python -m build --outdir /tmp/vantage-dist
python -m twine check /tmp/vantage-dist/*
python -m pip install /tmp/vantage-dist/*.whl
# Run from outside the checkout to prove no source-tree engine dependency:
cp tests/packaging/python_smoke.py /tmp/python_smoke.py
(cd /tmp && python python_smoke.py && vantage devices)
python scripts/package_npm.py --output /tmp/vantage-npm-dist
mkdir /tmp/vantage-node-consumer
(cd /tmp/vantage-node-consumer && npm install /tmp/vantage-npm-dist/*.tgz &&
 node node_modules/vantage-opt/install.js && node node_modules/vantage-opt/smoke.cjs)
docker build -f Dockerfile.cpu -t vantage:cpu .
docker run --rm vantage:cpu devices
```

Use new output directories for npm staging. Keep artifacts outside the checkout
when the workspace volume is nearly full. Registry publication, portable binary
builds, AUR publication, apt hosting and second-laptop reproduction remain future
steps; none follow automatically from a successful local build.

## Solver readiness remains bounded

The completed ten-model demonstration campaign has 30/30 optimal runs on CPU
and 30/30 on CUDA; HiGHS has 27/30, with the three unsupported MIQP runs retained.
This is demonstration evidence, not general MIPLIB completeness. The separate
one-million-variable HiGHS-IPM screening solved in 24.49 seconds (one repetition),
so claims based on default-HiGHS timeouts must include this alternative baseline.
General difficult MILPs, guarded singular sparse PSD acceptance, complete GPU
presolve/compaction, fully device-side restart/control, general implication/dual
conflicts, tableau cuts, full simplex/barrier checkpointing and second-machine
validation remain pending. AMD/HIP remains explicitly deferred.

Dashboard persistence was corrected to retain full solution vectors on disk;
response previews can still truncate large arrays without damaging downloads.

## Local validation outcome

CPU core: 241 assertions; CPU research: 15,182 assertions. Full CLI/Python,
planning, native extensions, trust/stability, benchmark-adapter, 17 dashboard API
tests and dependency policy passed. Wheel and sdist passed `twine check`; an
installed wheel solved a tiny LP through both subprocess and C ABI from `/tmp`.
The npm package built independently outside the checkout, solved the same LP,
rejected a missing input and honored a pre-aborted AbortSignal. Archive inspection
confirmed no datasets/results are included.

Both updated Docker images built. CPU native QP returned OPTIMAL, objective
2833.309965865842. CUDA native QP returned OPTIMAL, objective 2833.309965865844
with explicit NVIDIA device and driver-library passthrough. Standard `--gpus all`
still fails with “failed to discover GPU vendor from CDI”; this is a host runtime
configuration gap, not a claim that standard NVIDIA-container deployment works.
No host runtime configuration was changed.

Artifacts and checksums persist in ignored `results/distribution_20260928/`: CPU
Linux wheel, Python source tarball, npm source tarball and validation manifest.
The current wheel was built on this Arch Linux host and needs a manylinux rebuild
for broad Linux portability. The Linux distribution workflow has been authored
but not executed on GitHub in this session.
