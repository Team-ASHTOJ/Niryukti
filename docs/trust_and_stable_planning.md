# Signed evidence and stable replanning

## Stable replanning

```sh
./scripts/build.sh
python3 examples/stable_refinery_plan.py
```

The demonstration uses the existing synthetic coupled refinery LP. It first obtains a baseline plan, then removes day-3 production capacity. Two explicitly chosen penalty policies illustrate the trade-off between operating cost and deviation from that plan. Day-2 feed changes receive ten times the weight of other days. These policy weights are synthetic, not empirically calibrated operating costs. Complete augmented models, solutions, changes and standalone verifier outputs are saved under `results/stable-refinery-plan`.

For minimization, the new objective is `original_cost + sum(weight[j] * abs(x[j] - reference[j]))`. For maximization, the same nonnegative penalty is subtracted. Each deviation uses a continuous nonnegative auxiliary variable with two linear inequalities. Original constraints, bounds and integer types remain intact. Positive weights make the auxiliary variable equal the absolute deviation at an exact optimum. The report computes actual deviations directly from the returned plan, so a limited incumbent is not assumed to have tight auxiliaries.

```python
from vantage import replan
report = replan(model, reference_values, {'production_day_2': 5},
                locked=['production_day_0'], device='cpu', method='auto')
```

`reference_values` maps names to previously approved values. `locked` fixes listed variables to those exact values; use it for completed decisions. Inconsistent locks return an error or infeasible result, rather than silently changing the executed plan. Locks must be integral for integer variables and within current bounds. Linear objectives only are currently supported. Supply a full consistent set of executed decisions for a rolling-horizon application; the demonstration assumes advance knowledge of the outage and does not claim to replay actual refinery operations.

The report separates operating objective, weighted change penalty, signed variable changes, solver status and numerical verification. `VERIFIED_PLAN` means the augmented model's incumbent passed the standalone checker, not necessarily that the trade-off is optimal; inspect `result.status` and MIP gap. This is a weighted objective, not lexicographic minimization. Penalties do not relax safety or quality limits.

## Detached signatures

The optional signing path uses standard [OpenSSL SHA-256 digest signing and verification](https://docs.openssl.org/3.6/man1/openssl-dgst/), with caller-managed RSA or EC PEM keys. No custom cryptographic algorithm or private-key generation is embedded in the solver. OpenSSL is required only for signing/verifying signatures, not for numerical solving or unsigned bundles.

```sh
PYTHONPATH=python python3 -m vantage.evidence create examples/refinery.json /tmp/refinery.zip
PYTHONPATH=python python3 -m vantage.evidence sign /tmp/refinery.zip /tmp/refinery.sig --private-key /secure/path/team-private.pem
PYTHONPATH=python python3 -m vantage.evidence verify /tmp/refinery.zip --signature /tmp/refinery.sig --public-key /trusted/path/team-public.pem
```

The private key must already exist and be accessible without an interactive passphrase prompt. Keep it outside the repository; this implementation does not manage key storage, rotation, revocation or identity certification. Signature creation refuses to overwrite a sidecar. Share the public key through a trusted channel separately from the bundle. A key provided by an untrusted bundle would establish no useful identity, so the verifier never selects a key from archived metadata.

Signatures cover the exact ZIP bytes. Verification snapshots those bytes, verifies the detached signature, and then independently checks the bundled mathematical solution. Wrong keys, changed bytes and changed signatures fail before the numerical check. A correctly signed but mathematically invalid solution still fails numerical verification. Signature validity establishes control of the matching private key under the caller's trust choice; it does not establish model appropriateness or prove an integer search tree.

## Validation and remaining scope

Tests cover analytical minimum/maximum trade-offs, hard locks, infeasible locked plans, correct/wrong public keys, changed bundles and correctly signed but invalid objectives. Temporary test keys are generated and deleted inside the test directory; no production signing identity was created.

The CPU simplex path already supports compatible-basis reoptimization. General first-order/barrier-to-simplex crossover needs a separate rank/feasibility-aware basis construction algorithm and is not introduced by these features. GPU improvements remain assigned to supported-device profiling and validation.
