# GPU presolve, bound proofs and general conflicts — 2026-09-28

This is an unreleased source update after public package 0.2.1. Build the current
source to use it; the immutable published packages do not include these changes.

## What changed

### GPU proposals become continuous reductions

Directed-rounding CUDA propagation proposes tighter LP/QP bounds. Previously,
continuous solves used them only for initialization because their multipliers
could not be reconstructed against the original bound box.

`src/bound_postsolve.cpp` now independently replays row-based derivations with
outward-rounded interval arithmetic. Each accepted bound records its source row,
row multiplier, and dependencies on earlier bounds. The acyclic derivation record
is traversed backwards to substitute implied-bound multipliers into original row
duals. QP stationarity includes both diagonal and sparse Hessian contributions.
Current and averaged candidates, and candidate infeasibility rays, pass through
this reconstruction before the independent original-model verifier evaluates them.
GPU proposals alone do not authorize a status or a pruning bound.

Proof replay uses up to five passes, at most 200,000 derivations and two million
dependencies. Rows over 4,096 entries are skipped in proof replay to bound dense
host work. Skipped reductions preserve original bounds. This is a hybrid pipeline:
propagation runs on GPU; proof construction and final verification remain on CPU.

### CUDA CSR compaction

After host presolve selects retained rows and columns, CUDA counts retained entries,
uses CUB's 64-bit exclusive scan for row offsets, and gathers coefficients and
mapped column indices into compact CSR. Fixed-column objective/RHS substitutions,
parallel-row decisions and reversible metadata stay on CPU. Empty retained rows,
zero columns and merged/fixed-column cases are tested against host construction.
The compact matrix returns to the host for existing scaling and solver setup;
this does not eliminate every presolve transfer or move every reduction family
to the GPU. The normal CPU build retains its host sparse construction path.

### General bound conflicts

A learned clause excludes a conjunction of bound literals such as
`x >= 3 AND y <= 2`. Clauses can now contain general-integer and continuous bounds,
not only binary assignments. Learning requires an infeasibility contradiction
replayed through original-row bound propagation. LP limit outcomes are never
accepted as proof. Deletion filtering shortens explanations, syntactic subsumption
removes redundant clauses, and a bounded pool limits storage.

Unit propagation enforces the complement of the remaining literal. Integral
thresholds in the exact representable range use a one-unit integer complement.
Continuous literals use a conservative closed relaxation of their strict
complement, without a guessed epsilon. These clauses and their counters persist
in MIP checkpoints. Existing binary no-good linearization remains separate;
general disjunctions are propagated as domains, not advertised as arbitrary linear
cuts. This is not an implication-graph or general dual-proof conflict engine.

### Integer-lattice cuts

For pure-integer rows with exactly represented integral coefficients, the GCD of
coefficients defines the activity lattice. Dividing by that GCD and outward-rounding
finite endpoints yields valid strengthening rows. Separation checks violations,
exact row fingerprints prevent repeated insertion, and generated rows participate
in the existing root/node/global cut pools with aging and efficacy selection.
The coefficient/endpoint range is guarded. This supplements MIR, cover and clique
cuts; it is not a general tableau/Gomory separator.

## Verification and demo

Tests include original-model LP/QP KKT checks after bound reconstruction,
row/column sign transformations, direct CPU/CUDA compact-matrix comparison,
exhaustive integer-point validity for lattice cuts, integer and continuous conflict
propagation, and saved general-integer clauses with tree resume. Complete CLI,
Python, native API, service, dashboard, certificate and checkpoint regression tests
are run with the CUDA build. Raw logs and recording artifacts are retained locally.

```bash
./scripts/build.sh
./build/niryukti solve examples/refinery.json --method pdhg --device cuda \
  --gpu-presolve --gpu-monitor --cuda-graphs --json-out refinery.json
./build/niryukti verify examples/refinery.json refinery.json
./build/niryukti solve examples/supply_chain.json --method auto --device cuda \
  --cuts --branching reliability --gpu-presolve --json-out supply-chain.json
./scripts/recording_demo.sh
```

No new universal speedup claim is made. Benchmark timings from the frozen 0.2.1
campaign describe that earlier source revision. Current MIP verification checks an
incumbent and available bounds; it does not replay a full search-tree proof.

## Still outside this update

Fully device-resident restart/weight/limit control; all GPU presolve reduction
families; general implication-graph/dual conflicts; unrestricted tableau cuts;
production-quality direct GPU barrier factorization; unrestricted difficult PSD
recognition; standard NVIDIA Docker runtime configuration and second-host tests.
Those limits remain explicit rather than being silently relabeled complete.

Final local validation: **17,461 CUDA research assertions**, all four CTest
programs and the complete regression script passed. ASan/UBSan CPU CTest passed
4/4, and both binary/general conflict checkpoint tests passed under sanitizers.
Five engine-checkpoint/SIGINT tests passed on the CUDA build, including actual
GPU-presolve first-order state continuation and original-space dual verification.
Logs and source checksums: `results/gpu_presolve_bound_conflicts_20260928/`.

Offline recording artifacts: `results/recording-gpu-presolve-20260928/`.
The GPU refinery run applied 60 bound-tightening derivations, removed 16 rows,
and returned OPTIMAL with original-space normalized KKT error **1.685e-8**.
Supply-chain/scheduling outputs contain feasible integer incumbents; their
independent verifier evidence remains distinct from a complete tree proof.
