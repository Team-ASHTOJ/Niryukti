"""Random MILP families used to train and evaluate learned branching: weighted set cover and
capacitated facility location (as in Gasse et al. 2019) plus multidimensional knapsack,
which is hard for branch-and-bound at small sizes.

    python3 examples/generate_milp_benchmarks.py OUTPUT_DIR --count 20 --seed 0
"""
import argparse
import json
import random
from pathlib import Path


def set_cover(rng, rows=70, columns=140, density=0.05, name='set_cover'):
    variables = [{"name": f"pick_{j}", "type": "binary"} for j in range(columns)]
    cost = [rng.randint(1, 5) for _ in range(columns)]
    constraints = []
    for i in range(rows):
        members = {j for j in range(columns) if rng.random() < density}
        while len(members) < 2:
            members.add(rng.randrange(columns))
        constraints.append({"name": f"cover_{i}", "lb": 1,
                            "coefficients": {f"pick_{j}": 1 for j in sorted(members)}})
    return {"name": name, "sense": "min", "variables": variables,
            "objective": {"linear": cost}, "constraints": constraints}


def facility_location(rng, facilities=10, customers=25, name='facility_location'):
    demand = [rng.randint(5, 35) for _ in range(customers)]
    capacity = [rng.randint(40, 110) for _ in range(facilities)]
    while sum(capacity) < 1.2 * sum(demand):
        capacity = [c + 10 for c in capacity]
    fixed = [rng.randint(100, 300) for _ in range(facilities)]
    variables, cost, constraints = [], [], []
    for f in range(facilities):
        variables.append({"name": f"open_{f}", "type": "binary"})
        cost.append(fixed[f])
    for f in range(facilities):
        for c in range(customers):
            variables.append({"name": f"serve_{f}_{c}", "lb": 0, "ub": 1})
            cost.append(round(demand[c] * rng.uniform(1, 10), 2))
    for c in range(customers):
        constraints.append({"name": f"demand_{c}", "lb": 1, "ub": 1,
                            "coefficients": {f"serve_{f}_{c}": 1 for f in range(facilities)}})
    for f in range(facilities):
        row = {f"serve_{f}_{c}": demand[c] for c in range(customers)}
        row[f"open_{f}"] = -capacity[f]
        constraints.append({"name": f"capacity_{f}", "ub": 0, "coefficients": row})
        for c in range(customers):
            constraints.append({"name": f"link_{f}_{c}", "ub": 0,
                                "coefficients": {f"serve_{f}_{c}": 1, f"open_{f}": -1}})
    return {"name": name, "sense": "min", "variables": variables,
            "objective": {"linear": cost}, "constraints": constraints}


def knapsack(rng, items=35, dimensions=5, name='knapsack'):
    weight = [[rng.randint(5, 60) for _ in range(items)] for _ in range(dimensions)]
    profit = [sum(weight[d][j] for d in range(dimensions)) // dimensions + rng.randint(0, 20)
              for j in range(items)]
    constraints = [{"name": f"capacity_{d}", "ub": sum(weight[d]) // 2,
                    "coefficients": {f"take_{j}": weight[d][j] for j in range(items)}}
                   for d in range(dimensions)]
    return {"name": name, "sense": "max",
            "variables": [{"name": f"take_{j}", "type": "binary"} for j in range(items)],
            "objective": {"linear": profit}, "constraints": constraints}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for k in range(args.count):
        for family, build in (('setcover', set_cover), ('facility', facility_location),
                              ('knapsack', knapsack)):
            name = f"{family}_{args.seed}_{k}"
            (out / f"{name}.json").write_text(json.dumps(build(rng, name=name)) + "\n")


if __name__ == '__main__':
    main()
