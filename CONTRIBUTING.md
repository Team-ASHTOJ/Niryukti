# Contributing

Build a CPU version and run `./scripts/run_tests.sh` before proposing a change. CUDA changes also need actual device tests. Use clang-format with the repository configuration. Keep solver-library dependencies separate from external benchmark adapters.

Numerical changes should include a case that demonstrates why the change is valid, verified original-space accuracy and any affected benchmark outcomes. Never discard unsuccessful runs or loosen stopping criteria to claim convergence. Document mathematical assumptions and unsupported classes explicitly.

The current reference implementation prioritizes correctness and inspectability. Add performance improvements with measurements of both runtime and solution quality. See `docs/phase2.md` for the next development work.

## License of contributions

By submitting original contributions, you agree to license them under AGPL-3.0-only. You retain your copyright. Only submit code you have the right to license; identify any third-party material and preserve its notices. Existing numerical dependencies retain their original licenses.
