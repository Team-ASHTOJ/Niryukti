# VANTAGE dashboard

Run from the repository root:

```bash
./scripts/run_dashboard.sh
# http://127.0.0.1:8080
```

Use `--port 8081` to choose another port. Python's standard library is the only server dependency. The interface and its fonts are local; no frontend build, CDN, or network access is required at runtime.

The dashboard provides an overview, filterable benchmark comparisons, per-instance accuracy drawers, an industrial/public model library, and a live solver workspace. It launches the existing C++ binary in a separate process, supports CPU/CUDA/auto, exposes tolerance/thread/time controls, imports text MPS/LP/JSON models, streams real iteration logs, and saves downloadable solutions and run history under `results/dashboard/`.

Benchmark data uses the completed `results/phase2-final` campaign when available, otherwise `results/demo`, `results/netlib`, and `results/scalability-final`. Each model is counted once. The methodology panel reads thread/time/tolerance settings from the campaign manifest. All statuses are retained. No timings or accuracy figures are fabricated. Run `./scripts/run_demo.sh` if benchmark data is missing; generate the large model with the instructions in the main README. Reload the page after a new benchmark campaign.

Python and JavaScript coordinate execution and present the independent C++ solver’s results. The Phase 2 solver changes and benchmark evidence are described in [the documentation index](../docs/README.md). One local solve runs at a time; uploads are limited to 5 MB and execution requests to 120 seconds, plus a cleanup allowance. The server binds to loopback and is intended for your local workspace, not an internet deployment. Mutation endpoints require a session token; arbitrary filesystem paths and command strings cannot be supplied by the client.

Fonts: Instrument Sans and IBM Plex Mono, locally bundled with their OFL notices in `static/fonts/`.

Browser validation (development-only npm dependency):

```bash
npm ci --prefix dashboard/browser-tests
node dashboard/browser-tests/check.cjs
```

The browser check uses `/opt/google/chrome/chrome` by default; override `CHROME_PATH` and `DASHBOARD_URL` for another installation. It checks desktop/mobile layouts, navigation, filtering, details, a real solve, model import and solution download. `python3 -m unittest discover -s dashboard/tests` checks the API and benchmark aggregation.
