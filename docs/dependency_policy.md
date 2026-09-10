# Dependency and provenance policy

VANTAGE's solving path is independently implemented. It does not call, link, embed or wrap HiGHS, SCIP, CBC/CLP, GLPK, Gurobi, CPLEX, Xpress, OR-Tools, OSQP, SCS, ECOS, CVXPY or SciPy optimization functions.

Production dependencies are the C++ standard/runtime libraries, optional OpenMP, optional CUDA runtime/cuSPARSE, and the vendored nlohmann JSON parser. `third_party/json.hpp` is upstream nlohmann/json v3.12.0 with its MIT license retained in `third_party/JSON-LICENSE.MIT`. This is data-format infrastructure, not a numerical optimizer. GPU sparse products use NVIDIA cuSPARSE under the installed toolkit's terms; all optimization iteration and control logic lives in VANTAGE source.

Only `benchmark/adapters/highs.py` imports HiGHS. It runs as an external comparison process; the benchmark Python environment is separate from the product API. `scripts/check_dependencies.py` scans production source/build/package definitions for accidental optimization-engine references. This is a regression guard, not a substitute for source review.

The mathematical references are listed in `docs/algorithms.md`. No solver source was transplanted or renamed. Cached public benchmark files were obtained from a pinned HiGHS test-data mirror; these are model data, with source URLs/checksums and mirror license retained under `datasets/`. All generated industrial examples are labeled synthetic.
