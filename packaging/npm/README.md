# VANTAGE for Node.js

An asynchronous API to our independent native optimizer. Source installation
builds the bundled engine locally using CMake >=3.24 and a C++20 compiler.
There are no runtime downloads or external optimization-library dependencies.
Node.js >=18 is required. CPU is the default; set VANTAGE_BINARY to use a CUDA build.

```js
const {solve} = require('vantage-opt');
const result = await solve('model.mps', {method:'auto', timeLimit:60});
console.log(result.status, result.objective);
```

`solve` accepts a native JSON model object too. `signal` accepts an AbortSignal
for cancellation. Nonoptimal outcomes remain explicit in `result.status`.
The `vantage` command forwards CLI arguments to the native engine.

Licensed AGPL-3.0-only, with third-party notices retained. Public publication
is pending registry access. The initial source package is validated on Linux, not every OS.

Some npm versions block dependency install scripts by default. Review and approve
this package’s build script, or run `node node_modules/vantage-opt/install.js`
explicitly after installation. No engine is available until this build completes.
