# Refinery disruption planning

Run the CPU demonstration:

```sh
./scripts/build.sh
python3 examples/refinery_replanning.py --output results/refinery-replanning
```

The output directory contains the three model snapshots, warm and cold solutions, standalone verifier outputs, a JSON comparison table, and an infeasibility diagnostic and explicit repair proposal for a supply failure. A successful process exit means the experiment completed, not that every scenario solved; inspect status and verifier outcomes.

## Formulation and provenance

All data are invented demonstration coefficients, not MRPL data or an externally validated refinery formulation. Seven daily periods are linked by crude and product inventories. Flows/inventory use tonnes; costs use illustrative currency units, not claimed rupee savings. Clean and sour feeds contain 0.5% and 2% sulfur; the blended product limit is 1%. The model uses linear mass-weighted sulfur balance and assumes unit product yield. Nonlinear yields, viscosity, distillation and unit startup decisions are outside this example.

Each day offers up to 40 tonnes of each crude, production capacity is 100 tonnes/day, demand is 60 tonnes/day, crude tank capacity is 100 tonnes per crude and product storage is 200 tonnes. Initial clean/sour inventory is 40/20 tonnes, valued as sunk cost. Purchase costs are 70/45 currency units per tonne; daily closing inventory costs 0.1 per crude tonne and 0.2 per product tonne. No terminal inventory target is imposed. Baseline optimum is 22,200: sulfur requires at least 280 clean tonnes for total demand of 420, so net purchases are 240 clean and 120 sour tonnes. A just-in-time plan attains this lower bound with zero holding cost.

The delayed-delivery scenario removes day-1 clean supply and adds it to day 3. The outage scenario sets day-3 production capacity to zero. Each is solved from the original initial state, independently of the other scenario. These are advance replanning scenarios, not execution-aware rolling-horizon control: past decisions are not locked. Safety/quality constraints remain hard in the repair example; only named demand rows may be relaxed. The supply-failure case has no further crude purchases.

## Persistent sessions

```python
from vantage import SolverSession

with SolverSession('results/refinery-replanning/baseline.json') as session:
    original = session.solve()
    session.update_row_bounds({'capacity_3': {'ub': 0}})
    revised = session.solve(warm_start=True)
```

Set `PYTHONPATH=python` and optionally `VANTAGE_BINARY` when importing from the checkout. A session owns one sequential C++ worker and retains the parsed model. Updates accept existing row/variable names only; matrix structure and variable types remain fixed. Updates are transactional and validated. `None` removes a bound. Objective coefficients are in the original min/max objective convention. Compatible primal/dual vectors and simplex basis metadata are passed to subsequent solves; the existing basis compatibility checks decide whether a basis is reusable. Presolve and iterate workspaces are still rebuilt. Existing per-thread GPU matrix caching can survive within this worker, but reuse depends on unchanged processed matrices and has not been validated on this Mac.

The constructor waits for a ready response after parsing. Example timings cover solve request/response after readiness for both warm and cold sessions; they exclude session startup/loading and scenario-update time. They are single-run diagnostics, not a speedup claim or full operational latency. Cold solves use a separate process and disable warm starts.

## Infeasibility and repair

`diagnose_infeasibility(model)` runs the solver and invokes a separate verifier process. `conflict_verified` is true only for accepted Farkas evidence. Named participating rows plus original variable bounds form the certificate context; it is not a minimal irreducible infeasible subsystem. An infeasible status without a verified ray does not establish a verified conflict explanation.

`propose_repair(model, {'demand_0': 1, ...})` adds nonnegative violation variables solely to explicitly named row bounds and minimizes their positive weighted sum. Every other row, variable bound and integrality restriction stays hard. Lower/upper sides use separate slacks. Operating cost is excluded; weights have currency-per-unit-violation interpretation only if supplied that way. An accepted repair verifies the relaxed model, never the original one. The returned artifact includes that model, its solution and verification, plus proposed bound violations. Integer repair optimality still relies on solver tree bounds; standalone verification checks only incumbent feasibility.

Use `violation_limits={'demand_0': {'lb': 60, 'ub': 0}}` to cap permitted changes. The refinery demonstration caps each day's unmet demand at 60 tonnes and prohibits relaxing the other side of the demand equality. This prevents a repair from proposing negative demand or introducing material through relaxed balance equations. Unspecified limits are unbounded; callers must choose physically meaningful limits for their application.

## Validation

`tests/test_planning.py` checks a known linear optimum, persistent updates, failed-update rollback, minimum repair size, preservation of hard constraints, and refinery inventory/sulfur balances. GPU testing and domain-expert review remain pending.
