# NIRYUKTI for Node.js

An asynchronous interface to an independently implemented sparse C++ optimizer,
with cancellation and portable HTML reports. No external optimization engine or
runtime download is used.

## Installation

```bash
npm install niryukti
```

Requires Node.js 18+, CMake 3.24+ and a C++20 compiler. Installation builds the
included CPU engine locally. If npm blocks build scripts, review and approve the
script or run `node node_modules/niryukti/install.js`. Linux is validated; other
platforms require their own validation. CUDA requires a separately built engine.

## Solve and report

```js
const fs = require('node:fs');
const { solve, renderReport } = require('niryukti');

async function main() {
  const result = await solve('model.mps', {
    method: 'auto', device: 'auto', timeLimit: 60,
    tolerance: 1e-6, threads: 2
  });
  console.log(result.status, result.objective);
  fs.writeFileSync('report.html', renderReport(result));
}
main().catch(console.error);
```

`solve` accepts an MPS/LP/JSON path or native JSON model object. `signal` accepts
an AbortSignal for subprocess cancellation. `binary` or `NIRYUKTI_BINARY` can
select your compiled CUDA executable. Legacy VANTAGE_BINARY is accepted.
Automatic selection is structural and memory-aware, not a universal speed
promise. Nonoptimal statuses remain explicit; inspect accuracy and status before
using a candidate. Report rendering does not certify the supplied result.

## CLI

```bash
npx niryukti solve model.mps --method auto --device auto --json-out result.json
npx niryukti verify model.mps result.json
npx niryukti report result.json --output report.html
```

The authenticated HTTP service is provided by the Python package (`pip install
niryukti`, then `niryukti serve`), not the Node CLI. See repository `docs/api.md`.
The published 0.2.0 release predates `renderReport`; this documentation describes
the next source release.

## Supported scope and license

LP, supported convex sparse QP, MILP and convex MIQP. Large singular PSD
certification and advanced integer performance remain restricted. No general
nonconvex global optimization. Original code is AGPL-3.0-only; commercial use and
copying are permitted under its terms. LICENSE and third-party NOTICE are bundled.
