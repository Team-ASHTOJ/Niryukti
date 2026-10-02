"""Synthetic block-angular planning LP: several refineries share a crude import jetty and
national product demand. Each refinery's own balances form a block; the shared rows link
them. Used to demonstrate Dantzig-Wolfe decomposition (`niryukti decompose`)."""
import json
import random
import sys


def build(refineries=4, crudes=3, products=2, seed=7):
    rng = random.Random(seed)
    variables, constraints, cost = [], [], []
    yields = [[round(rng.uniform(0.2, 0.6), 3) for _ in range(products)] for _ in range(crudes)]
    price = [round(rng.uniform(55, 80), 2) for _ in range(crudes)]
    for r in range(refineries):
        cdu = rng.choice([180, 220, 260])
        for c in range(crudes):
            variables.append({"name": f"crude_{r}_{c}", "lb": 0, "ub": 200})
            cost.append(price[c] + round(rng.uniform(0, 6), 2))
        for p in range(products):
            variables.append({"name": f"make_{r}_{p}", "lb": 0, "ub": 400})
            cost.append(round(rng.uniform(1, 4), 2))
            # Production limited by crude yields at this refinery.
            coefficients = {f"make_{r}_{p}": 1}
            for c in range(crudes):
                coefficients[f"crude_{r}_{c}"] = -yields[c][p]
            constraints.append({"name": f"yield_{r}_{p}", "ub": 0, "coefficients": coefficients})
        constraints.append({"name": f"cdu_{r}", "ub": cdu,
                            "coefficients": {f"crude_{r}_{c}": 1 for c in range(crudes)}})
    for c in range(crudes):
        constraints.append({"name": f"jetty_{c}", "ub": 65 * refineries,
                            "coefficients": {f"crude_{r}_{c}": 1 for r in range(refineries)}})
    for p in range(products):
        constraints.append({"name": f"demand_{p}", "lb": 45 * refineries,
                            "coefficients": {f"make_{r}_{p}": 1 for r in range(refineries)}})
    return {"name": f"multi_refinery_{refineries}", "sense": "min", "variables": variables,
            "objective": {"linear": cost}, "constraints": constraints}


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    output = sys.argv[2] if len(sys.argv) > 2 else "examples/multi_refinery.json"
    with open(output, "w") as f:
        json.dump(build(count), f, indent=1)
        f.write("\n")
