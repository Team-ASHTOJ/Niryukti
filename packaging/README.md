# VANTAGE distribution

`vantage-opt` packages the independently implemented C++ optimization engine.
Python wheels contain the CPU executable and native C ABI library; no existing
optimization solver is used. Python 3.10+ is required. Source installations need
CMake 3.24+ and a C++20 compiler. Default wheels use FP64 and do not require CUDA.

```python
from vantage import Model, NativeSession
m = Model()
m.add_var("x", ub=10)
m.add_constraint({"x": 1}, ">=", 2)
m.set_objective({"x": 3})
with NativeSession(m) as session:
    print(session.solve())
```

Build: `python -m build`. Install a generated wheel with `pip install dist/*.whl`.
The installed `vantage` command exposes the engine CLI. CUDA builds can be used
through `VANTAGE_BINARY` and `VANTAGE_LIBRARY`; CUDA is not bundled into CPU wheels.

License: **AGPL-3.0-only**. Modified covered redistributions must retain AGPL;
modified network deployments must offer corresponding source to their users.
Copying and commercial use remain permitted. Third-party licenses are retained.
See LICENSE and NOTICE in each distribution.

Registry publication needs account ownership and trusted-publisher setup. Only
tested host platforms should be advertised. The release workflow builds repaired
manylinux x86_64 wheels plus a corresponding-source tarball; other platforms may
build from source. CUDA builds are not bundled in the initial CPU release.
