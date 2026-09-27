# Submission engineering round — 27 September 2026

This is a working record, not a declaration that every proposed solver algorithm exists.

## Repository integration

Created `main` from `master`, fast-forward merged `frontend` at `c41ae7b`, pushed it and changed the GitHub default branch to `main`. Older branches are retained. Subsequent changes are developed and validated on `main`. The user's frontend design and NIRYUKTI display branding are preserved; the numerical executable/API remain `vantage`.

## New mathematical and execution work

- General symmetric sparse convex QP participates in fixed-variable substitution, reversible presolve and scaling. Cross terms update linear costs and objective offsets. Large Q matrices currently retain conservative diagonal-dominance validation; column scaling is restricted for that class.
- Convex MIQP uses the existing branch-and-bound engine and our continuous QP relaxations. Pruning uses outward-rounded affine convex minorants over variable bounds, not approximate primal objectives. Unavailable bounds remain unavailable. Numerical convexity checks are not exact-arithmetic PSD proofs.
- A from-scratch two-phase revised **primal** simplex path standardizes compact LPs, stores the model sparsely and uses sparse Eigen LU numerical factorization with product-form updates. It is capped at 4096 transformed rows. `VANTAGE_SPARSE_LU=OFF` retains the own dense numerical kernel with a 512-row guard. It is not dual simplex, does not retain warm bases and does not yet certify general recession directions. `--method simplex` selects it explicitly; `--method auto` considers it for compact CPU LP relaxations. The existing `pdhg` default is retained.
- Opt-in root MIR separation complements existing binary cover/clique cuts. It is restricted to rows with finite lower bounds and safe integer shifts. No tableau/Gomory separator or commercial cut management is claimed.
- MILP nodes store bound differences relative to root bounds. Large-model queued warm vectors are discarded to control memory. Best-bound, depth-first and best-estimate selection share a global minimum open-node bound calculation; estimates never authorize pruning.
- Opt-in `--gpu-monitor` calculates current/averaged scaled diagnostic residuals on CUDA and downloads a scalar. This gates expensive full candidate checks. The final original-space CPU verifier remains authoritative; monitoring is not an independent certificate and full checkpoints/restarts still transfer vectors.
- `--certificate-out` exports the full structured result and verification scope. Standalone verification rejects incorrect reported objectives and claimed optimality with invalid continuous KKT metrics. MILP certificate replay verifies an incumbent, not the complete search tree proof.
- Native `.qplib` reading targets continuous LP/convex QP with linear or box constraints. Unsupported discrete/nonlinear variants are explicit errors.

## Evidence policy

Tests and campaigns are recorded after execution. Timing must be measured without concurrent builds/tests. Keep unsuccessful runs and exact executable, configuration, hardware and data checksums. Do not substitute historical nine-instance results for current broad coverage.

The public downloader cached all 91 MPS models in the pinned COIN Netlib mirror revision `f1cc423067407d55d579c9c35fb01edf860dbc24`, with the upstream licence and per-file SHA-256 manifest. This is the available mirror subset, not a claim to cover every Netlib variant.

## Competitor review

[SANKHYA](https://sankhya-solver.vercel.app/) and [IGAOS](https://pypi.org/project/igaos/) publish solver features/results. Their numbers have not been independently reproduced here, so hardware/configuration-independent superiority is not claimed. The reviewed [third repository](https://github.com/mohitsaitummalapalli-tech/SIH26119-Indigenous-GPU-Accelerated-Optimization-Solver) describes a foundation-stage implementation. No competitor solving code was copied or linked.

## Still pending beyond this round

GPU barrier/IPM; warm-basis revised dual simplex; batched GPU strong branching/OBBT; conflict analysis; advanced flow/tableau cuts; fully GPU-resident presolve/restart/control; HIP and multi-GPU; rigorous full MIP proof replay; unrestricted large sparse PSD validation; robust general NLP/MINLP; exact search-tree checkpoint restoration. These are separate substantial numerical engineering tasks, not submission-ready features.

## References

[Condat splitting](https://lcondat.github.io/publis/Condat-optim-JOTA-2013.pdf), [Netlib](https://www.netlib.org/lp/data/), [COIN mirror](https://github.com/coin-or-tools/Data-Netlib), [QPLIB format](https://qplib.zib.de/doc.html), [MIPLIB](https://miplib.zib.de/). Solver mathematics is implemented independently; existing optimization engines are external benchmark comparators only.

## Sparse numerical factorization addition

The initial 91-model, two-second, loaded-machine run solved 28 cases versus HiGHS's 84. Its seven numerical errors and parser failures directly motivated preprocessing/scaling for simplex, a Bland degeneracy fallback, cleanup time checks and fixed-format MPS name/set handling. The frozen executable and source archive remain under `results/submission-netlib/`.

Eigen 5.0.0 is now vendored only for ordinary sparse basis-system LU factorization. The optimization formulation, phase I/II, pricing, pivots, branch-and-bound, cuts and verification remain independently implemented. No Eigen unsupported optimization modules were included. Upstream licences and immutable archive SHA-256 are preserved in `third_party/eigen/provenance.json`. See [official SparseLU documentation](https://libeigen.gitlab.io/eigen/docs-5.0/classEigen_1_1SparseLU.html). This addition must be assessed by fresh tests/campaigns, not assumed faster.
