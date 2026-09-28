# Frozen 0.2.1 prototype campaign

Run from the repository root with a Python environment containing `highspy`:

```sh
python benchmark/freeze_campaign.py --binary build/vantage --output results/prototype_0_2_1
```

The committed configuration declares all nine models before execution. Four configurations run three measured repetitions each: automatic selection, CPU, CUDA and external HiGHS. Each receives a 30-second solve budget, one CPU thread and target tolerance 1e-6. NIRYUKTI receives a one-million-iteration ceiling so E226 is not constrained by the earlier 100,000-iteration default. CUDA receives one discarded warmup per instance. Parsing and verification have separate safety timeouts. CPU/CUDA use method auto; these compare deployed configurations, not identical algorithm kernels.

All 108 measured rows remain, including unavailable devices, failures and limits. Source, executable, input checksums, machine state, commands and raw verifier output are saved. `checksums.json` seals all saved files, including the exact configuration. Existing campaigns are archived rather than overwritten. HTML reports and performance profiles work offline.

`reported_status` records each solver's claim. Evidence `status=OPTIMAL` requires independent continuous KKT verification or a verified integer incumbent plus an independently recomputed relaxation bound closing its gap. `FEASIBLE_UNPROVEN` means the solver reports optimality but only incumbent feasibility was independently established. No MIP search tree is replayed; `tree_proof_verified` remains false. This applies equally to NIRYUKTI and external baselines. This small frozen suite is a prototype regression campaign, not proof of broad commercial competitiveness.
