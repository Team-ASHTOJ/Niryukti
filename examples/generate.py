#!/usr/bin/env python3
"""Deterministic synthetic industrial cases. Coefficients are illustrative, not MRPL data."""
import argparse
import json
import random
from pathlib import Path


def refinery(crudes=4, products=2, periods=4, seed=42, demand_multiplier=1.0):
    rng = random.Random(seed)
    sulfur = [0.2 + 2.5 * i / max(1, crudes - 1) for i in range(crudes)]
    cost = [65 - 7 * sulfur[i] + rng.uniform(-1, 1) for i in range(crudes)]
    variables, objective, constraints = [], [], []
    def row(name, terms, lb=None, ub=None):
        constraints.append(dict(name=name, coefficients=terms, lb=lb, ub=ub))
    for t in range(periods):
        for p in range(products):
            demand = (80 + 10 * p) * demand_multiplier
            terms, quality = {}, {}
            for c in range(crudes):
                name = f'flow_{t}_{p}_{c}'
                variables.append(dict(name=name, lb=0, ub=120))
                objective.append(cost[c] + 2)
                terms[name] = 1
                quality[name] = sulfur[c] - (0.8 + 0.2 * p)
            # Linear mass-weighted sulfur specification; no claim of nonlinear assay fidelity.
            row(f'demand_{t}_{p}', terms, lb=demand, ub=demand)
            row(f'sulfur_{t}_{p}', quality, ub=0)
        for c in range(crudes):
            row(f'availability_{t}_{c}', {f'flow_{t}_{p}_{c}': 1 for p in range(products)}, ub=120 * products)
        row(f'cdu_{t}', {f'flow_{t}_{p}_{c}': 1 for p in range(products) for c in range(crudes)}, ub=150 * products)
    return dict(name='synthetic_refinery_blending', sense='min', variables=variables,
                objective=dict(linear=objective), constraints=constraints,
                provenance=dict(synthetic=True, seed=seed, units='illustrative tonnes and cost units'))


def scheduling(periods=4):
    variables, costs, rows = [], [], []
    for t in range(periods):
        for name, lb, ub, typ, cost in [(f'flow_{t}', 0, 100, 'continuous', 4),
                                      (f'on_{t}', 0, 1, 'binary', 30),
                                      (f'start_{t}', 0, 1, 'binary', 20),
                                      (f'inventory_{t}', 0, 100, 'continuous', 0.2)]:
            variables.append(dict(name=name, lb=lb, ub=ub, type=typ)); costs.append(cost)
        rows.append(dict(name=f'capacity_{t}', coefficients={f'flow_{t}':1, f'on_{t}':-100}, ub=0))
        rows.append(dict(name=f'min_run_{t}', coefficients={f'flow_{t}':1, f'on_{t}':-20}, lb=0))
        balance={f'flow_{t}':1, f'inventory_{t}':-1}
        if t: balance[f'inventory_{t-1}']=1
        rows.append(dict(name=f'balance_{t}', coefficients=balance, lb=40, ub=40))
        startup={f'start_{t}':1, f'on_{t}':-1}
        if t: startup[f'on_{t-1}']=1
        rows.append(dict(name=f'startup_{t}', coefficients=startup, lb=0))
    return dict(name='synthetic_refinery_schedule',sense='min',variables=variables,objective=dict(linear=costs),constraints=rows)


def supply_chain():
    variables=[dict(name=f'open_{i}',lb=0,ub=1,type='binary') for i in range(2)]
    costs=[30,20]
    for i in range(2):
        for j in range(3):
            variables.append(dict(name=f'ship_{i}_{j}',lb=0,ub=30));costs.append(1+abs(i-j))
    rows=[]
    for i in range(2):
        terms={f'ship_{i}_{j}':1 for j in range(3)};terms[f'open_{i}']=-50
        rows.append(dict(name=f'capacity_{i}',coefficients=terms,ub=0))
    for j in range(3):rows.append(dict(name=f'demand_{j}',coefficients={f'ship_{i}_{j}':1 for i in range(2)},lb=20,ub=20))
    return dict(name='synthetic_supply_chain',sense='min',variables=variables,objective=dict(linear=costs),constraints=rows)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['refinery','scheduling','supply-chain'],default='refinery')
    p.add_argument('--crudes',type=int,default=4);p.add_argument('--products',type=int,default=2);p.add_argument('--periods',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',default='examples/refinery.json')
    a=p.parse_args()
    if min(a.crudes,a.products,a.periods)<1:p.error('dimensions must be positive')
    model=refinery(a.crudes,a.products,a.periods,a.seed) if a.kind=='refinery' else scheduling(a.periods) if a.kind=='scheduling' else supply_chain()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(model,indent=2)+'\n')
