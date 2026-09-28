# GPU control, presolve and conflicts — 2026-09-28

## Changes in this round

- GPU restarts select the resident current or averaged candidate with device
  copies. Averaging state, anchors and row activities are reset on-device. The
  host still chooses the candidate and controls weighting, limits and proof checks.
- Candidate activity rebasing no longer uploads the primal vector back to the GPU.
- Directed-rounding GPU propagation now tightens continuous bounds as well as
  integer bounds. Continuous endpoints remain outward rounded; integer endpoints
  are rounded only within the guarded representable range. LP/QP first-order
  initialization can use `--gpu-presolve`; continuous solver bounds remain the
  original bounds until dual postsolve provenance is implemented. MIP node bounds
  can use the tightening directly. CPU reversible compaction remains in use.
  GPU contradictions are not independently exposed as certificates.
- Binary no-good conflicts now perform fixed-point unit propagation and, with
  cuts enabled, generate globally valid relaxation rows. Conflicts are learned
  only from proved infeasibility, not from limit outcomes.
- Pooled MIR and conflict rows are appended in one sparse rebuild per node,
  rather than repeatedly rebuilding the complete sparse matrix for each row.
- Checkpoint conflicts reject empty clauses and duplicate variables.
- Fixed a string-content comparison in the newly pulled certificate diagnostic.

## Verification

New tests check continuous feasible boundary preservation, actual LP GPU-presolve
solves, resident versus host-reset continuation, cascaded binary conflict
propagation, contradiction detection, and exhaustive binary feasible assignments.
The full CPU/CUDA suites passed after implementation. CUDA ran on an NVIDIA
RTX 4060 Laptop GPU and passed 16,376 research assertions. All four CTest
programs passed, as did CLI/Python/API/dashboard/certificate checks. CPU
certificate tests skip their CUDA-only case. Logs are retained in
`results/gpu_conflicts_20260928/`.

## Supported scope and remaining work

This is not full device-resident optimization: host decisions, final original-space
verification and some candidate downloads remain. GPU presolve does not perform
complete variable/row compaction or all reduction families. Conflict learning is
binary-conjunction based, not general implication-graph/dual-proof analysis.
Current MIR/cover/clique cuts are not an unrestricted tableau separator catalog.
These remaining algorithms must be implemented and validated incrementally;
there is no claim of commercial-solver completeness or universal speedup.

## Run the new paths

```bash
./build/niryukti solve model.mps --device cuda --method pdhg \
  --gpu-presolve --gpu-monitor --cuda-graphs --json-out result.json
./build/niryukti solve integer-model.mps --device cuda --method auto \
  --branching reliability --cuts --gpu-presolve --json-out mip-result.json
./build/niryukti verify model.mps result.json
```

For a different NVIDIA laptop use `scripts/validate_cuda_laptop.sh`; the full
research suite includes the new CUDA regression cases. Final correctness remains
an original-model check. No new speed ratios are reported without measurements.
