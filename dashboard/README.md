# NIRYUKTI dashboard

Run from the repository root:

```bash
./scripts/run_dashboard.sh
# http://127.0.0.1:8080
```

Use `--port 8081` to choose another port. Python's standard library is the only server dependency. The interface and its fonts are local; no frontend build, CDN, or network access is required at runtime.

The dashboard provides an overview, filterable benchmark comparisons, per-instance accuracy drawers, an industrial/public model library, and a live solver workspace. It launches the existing C++ binary in a separate process, supports CPU/CUDA/auto, exposes tolerance/thread/time controls, imports text MPS/LP/JSON models, streams real iteration logs, and saves downloadable solutions and run history under `results/dashboard/`.

Benchmark data uses the completed `results/phase2-final` campaign when available, otherwise `results/demo`, `results/netlib`, and `results/scalability-final`. Each model is counted once. The methodology panel reads thread/time/tolerance settings from the campaign manifest. All statuses are retained. No timings or accuracy figures are fabricated. Run `./scripts/run_demo.sh` if benchmark data is missing; generate the large model with the instructions in the main README. Reload the page after a new benchmark campaign.

Python and JavaScript coordinate execution and present the independent C++ solver’s results. The Phase 2 solver changes and benchmark evidence are described in [the documentation index](../docs/README.md). One local solve runs at a time; uploads are limited to 5 MB and execution requests to 120 seconds, plus a cleanup allowance. The server binds to loopback and is intended for your local workspace, not an internet deployment. Mutation endpoints require a session token; arbitrary filesystem paths and command strings cannot be supplied by the client.

Fonts: Sora (display and figures), Instrument Sans (interface) and IBM Plex Mono (numeric readouts), locally bundled with their OFL notices in `static/fonts/`.

Browser validation (development-only npm dependency):

```bash
npm ci --prefix dashboard/browser-tests
node dashboard/browser-tests/check.cjs
```

The browser check uses `/opt/google/chrome/chrome` by default; override `CHROME_PATH` and `DASHBOARD_URL` for another installation. It checks desktop/mobile layouts, navigation, filtering, details, a real solve, model import and solution download. `python3 -m unittest discover -s dashboard/tests` checks the API and benchmark aggregation.

## Instrumentation interface

The dashboard follows the NIRYUKTI brand: charcoal ink and champagne gold. Light theme is the default brand sheet; the dark theme is graphite with a brighter champagne. The theme follows the system preference until the sun/moon button saves a choice. Colours are tokens in `style.css`, so charts, effects and the logo recolour together. The interface uses locally bundled fonts, CSS perspective geometry and SVG plots. No WebGL renderer or new runtime dependencies are required. The **Effects** button persists a local preference that disables decorative motion and the computational background. System reduced-motion preferences also disable animations.

Navigation includes run history, import, verification, hardware, diagnostics and in-app documentation alongside the original overview, solver, library and benchmark views. The solver cockpit displays real result fields and a logarithmic envelope of logged primal, dual and gap values. MILP relaxation logs are not plotted as global convergence. Refinery and branch-tree diagrams are explicitly illustrative; dispatch bars use the returned solution vector. GPU utilization, memory, search topology and separate independent recomputation are marked unavailable or omitted where the API does not expose them.

Benchmark plots use actual recorded medians. Cactus and scatter plots include optimal results only; performance profiles retain unsolved cases in their denominator. Empty datasets remain empty rather than receiving example values. Expanded browser validation covers all routes at 390, 1440, 1920 and 2560 pixel widths, effect preferences, reduced motion, and real LP/QP/MILP runs.

The NIRYUKTI mark and wordmark are inline SVG symbols in `index.html` (`#nk-mark`, `#nk-word`), redrawn as vectors from the supplied logo so they recolour per theme. The mark is an N on a coordinate frame crossed by a gold optimization path; the overview hero draws that path once on load. `favicon.svg` adapts to the browser's colour scheme, and `brand.svg` is the full lockup with tagline for use outside the app.

The overview’s asymmetric grid, diamond textures, hatched comparison bars and one-time tile entrances adapt the supplied [Zepa UI any-grid](https://zepa.design/components/any-grid) reference. Attribution is retained in source. Layout and action highlights stay stable, measurements never scramble, and entrance effects use an intersection observer without a continuous rendering loop. These shared treatments also apply to the model library and analysis panels.

The sticky top navigation adapts the supplied [Zepa UI cohort-navbar](https://zepa.design/components/cohort-navbar) reference, with NIRYUKTI routes, a Workspace mega menu, hover/focus previews, and a collapsible tablet/mobile menu. Dropdowns support click, keyboard entry, Escape, and outside-click dismissal. No React or external image dependencies were added.

The overview computation stage includes a decorative cursor-responsive wave grid inspired by Zepa UI wave-hero (@franky-adl). It uses 256 Canvas 2D blocks at up to 24 updates per second with a fixed 640×360 backing surface and at most six pointer ripples. It stops offscreen, in hidden tabs, during local solver runs, with reduced motion, or with Effects off. The effect adds no WebGL, Three.js, GSAP, or network dependencies.

The global PixelSnow background uses the supplied React Bits shader, adapted to native WebGL2 without adding React or Three.js. It uses the supplied square-flake settings, a backing canvas capped at 480×600, a 24 Hz timer, and low-power context preference. Motion pauses with reduced motion, Effects off, hidden tabs, or an active local solve. If WebGL2 initialization fails or its context is lost, the ordinary gradient/grid background remains available.
