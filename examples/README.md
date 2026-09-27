# Industrial examples

All coefficients are synthetic and illustrative, not MRPL data.

- `refinery.json`: flow allocation across crudes/products/periods, feed availability, capacity, demand balance and linear mass-weighted sulfur specifications. Periods are independent. Cost includes illustrative feed and processing cost. API gravity, viscosity and nonlinear yield interactions are not modeled.
- `scheduling.json`: multi-period flow, binary unit on/off and startup, minimum run rate, tank inventory balance and holding cost. This intentionally exercises basic MILP; a limited solve may return an incumbent with an unresolved gap.
- `supply_chain.json`: facility-open binaries, transportation flows, capacities, demand and fixed/variable cost.
- `dispatch.json`: three-generator economic dispatch with convex quadratic generation costs, capacity bounds and a power-balance equality. Q's diagonal is twice the coefficient of the squared term.
- `toy.lp`: tiny LP with objective 2 for a fast installation check.

`generate.py` reconstructs the synthetic cases. `warm_resolve.py` increases blending demand by 5%, solves cold and warm, verifies their objective agreement and saves both full outputs. The report makes no promise that a warm start is always faster.

Additional submission cases: `coupled_dispatch.json` uses symmetric off-diagonal quadratic costs; `integer_dispatch.json` demonstrates convex MIQP; `production.json` links production and inventory across periods. All are illustrative synthetic data, not MRPL measurements.
