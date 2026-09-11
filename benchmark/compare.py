#!/usr/bin/env python3
"""Generate a complete before/after evidence table from archived benchmark runs."""
import argparse
import json
import math
import os
import statistics
from pathlib import Path


def load(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    if not (folder / 'summary.json').exists():
        raise ValueError(f'Incomplete campaign: {folder}')
    groups = {}
    for path in sorted((folder / 'raw').glob('*.json')):
        raw = json.loads(path.read_text())
        record = raw.get('record', {})
        if record.get('warmup') or record.get('run', -1) < 0:
            continue
        groups.setdefault((record['instance'], record['solver']), []).append(raw)
    return manifest, groups


def summary(runs):
    records = [r['record'] for r in runs]
    states = sorted({r['status'] for r in records})
    times = [r['end_to_end_seconds'] for r in records if r.get('end_to_end_seconds') is not None]
    return '/'.join(states), statistics.median(times) if times else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path, default=Path('docs/phase2_results.md'))
    parser.add_argument('--extended', type=Path, help='Separate higher-iteration-budget campaign; never merged into the standard table')
    args = parser.parse_args()
    before_meta, before = load(args.before)
    after_meta, after = load(args.after)
    for key in ('threads', 'tol', 'time_limit', 'runs', 'iterations'):
        if before_meta['arguments'].get(key,100000) != after_meta['arguments'].get(key,100000):
            raise ValueError(f'Campaign settings differ: {key}')
    if set(k for k in before if k[1] in ('cpu','cuda')) != set(k for k in after if k[1] in ('cpu','cuda')):
        raise ValueError('Before/after instance selections differ')
    for key in before:
        if key in after and {r['record']['dataset_sha256'] for r in before[key]} != {r['record']['dataset_sha256'] for r in after[key]}:
            raise ValueError(f'Dataset mismatch: {key}')
    config = after_meta['arguments']
    lines = ['# Phase 2 measured results', '',
        'Generated from retained raw runs by `benchmark/compare.py`. All selected cases appear, including limits and regressions.', '',
        f"Settings: {config['runs']} measured runs per case/backend, one separate CUDA warmup, {config['threads']} CPU threads, tolerance {config['tol']}, time limit {config['time_limit']} s, {config.get('iterations',100000):,} continuous iterations per solve. Times below are median end-to-end solver time, including parsing, setup, transfer and verification. Each MILP node has its own continuous iteration cap.", '',
        f"CPU: {after_meta.get('cpu_info')}. GPU: {after_meta.get('gpu_state','').strip().replace(chr(10),' / ')}", '',
        'These measurements were taken on a shared development laptop, not an isolated performance laboratory. Small timings and GPU startup times are sensitive to system activity. They do not establish industrial competitiveness.', '',
        '| Instance | Backend | Before status | Before s | After status | After s | Before/after speedup¹ |',
        '| --- | --- | --- | ---: | --- | ---: | ---: |']
    for key, runs in before.items():
        if key[1] not in ('cpu','cuda'): continue
        old, old_t = summary(runs); new, new_t = summary(after[key])
        speed = f'{old_t/new_t:.2f}×' if old == new == 'OPTIMAL' and old_t and new_t else '—'
        fmt = lambda v: f'{v:.6f}' if v is not None else '—'
        lines.append(f'| {key[0]} | {key[1]} | {old} | {fmt(old_t)} | {new} | {fmt(new_t)} | {speed} |')
    lines += ['', '¹Speedup is shown only when every repetition in both campaigns is optimal. Values below one are regressions. A limit time is not a solve time.', '', '| Solver | Before: cases optimal in every run | After: cases optimal in every run |', '| --- | ---: | ---: |']
    for solver in ('cpu','cuda','highs'):
        counts=[]
        for groups in (before,after):
            selected=[r for k,r in groups.items() if k[1]==solver]
            counts.append(f'{sum(summary(r)[0]=="OPTIMAL" for r in selected)}/{len(selected)}' if selected else 'Not measured')
        lines.append(f'| {solver} | {counts[0]} | {counts[1]} |')
    errors=[]
    for (instance,solver),runs in after.items():
        if solver == 'highs' or summary(runs)[0] != 'OPTIMAL': continue
        reference = after.get((instance,'highs'),[])
        if not reference or summary(reference)[0] != 'OPTIMAL': continue
        objective = statistics.median(r['record']['objective'] for r in reference)
        errors.extend(abs(r['record']['objective']-objective)/max(1,abs(objective)) for r in runs)
    lines += ['', f'Worst relative objective difference from the matched HiGHS reference among optimal VANTAGE runs: **{max(errors):.6g}**.' if errors else 'Objective agreement unavailable.', '',
        'For LP/diagonal QP, optimal status also requires original-space feasibility, stationarity and gap checks at the requested tolerance. For MILP, it requires a feasible integer incumbent and conservative global-bound gap. A rounded incumbent’s continuous KKT value is not an integer optimality test.', '',
        '## Provenance', '']
    for label,folder,meta in [('Before',args.before,before_meta),('After',args.after,after_meta)]:
        link=os.path.relpath(folder,args.output.parent)
        lines += [f"- {label}: [{folder}]({link}/index.html), timestamp `{meta['timestamp']}`, executable SHA-256 `{meta['binary_sha256']}`. Raw commands, per-run JSON, stdout/stderr, model checksums, hardware and build metadata are retained in that directory."]
    if args.extended:
        extra_meta, extra = load(args.extended)
        if extra_meta['binary_sha256'] != after_meta['binary_sha256']:
            raise ValueError('Extended campaign used a different executable')
        for key,runs in extra.items():
            if key not in after or {r['record']['dataset_sha256'] for r in runs} != {r['record']['dataset_sha256'] for r in after[key]}:
                raise ValueError(f'Extended dataset mismatch: {key}')
        settings=extra_meta['arguments']
        link=os.path.relpath(args.extended,args.output.parent)
        lines += ['', '## Separate extended-budget experiment', '',
            f"The same executable was also measured with {settings['iterations']:,} VANTAGE iterations, {settings['threads']} threads, tolerance {settings['tol']}, {settings['time_limit']} s and {settings['runs']} repetitions. This changes the iteration budget, so these results do **not** replace the standard-budget statuses above. [Raw extended campaign]({link}/index.html).", '',
            '| Instance | Solver | Status across repetitions | Median end-to-end s | Median accepted iterations |',
            '| --- | --- | --- | ---: | ---: |']
        for (instance,solver),runs in extra.items():
            state,seconds=summary(runs)
            iterations=[r['record']['iterations'] for r in runs if r['record'].get('iterations') is not None]
            iteration_text=f'{statistics.median(iterations):,.0f}' if iterations else '—'
            seconds_text=f'{seconds:.6f}' if seconds is not None else '—'
            lines.append(f'| {instance} | {solver} | {state} | {seconds_text} | {iteration_text} |')
    lines += ['', '## Interpretation and remaining work', '',
        'This is a selected nine-case prototype suite, not a representative estimate of industrial solve rate. Synthetic refinery periods are largely independent. The primary improvements in 0.2 are local adaptive step control, cached verification, conservative integer-bound propagation and replayable box/row infeasibility certificates. General sparse QP, stronger restart merit functions, wider public benchmark coverage, persistent GPU contexts, cuts and general recession certificates remain future work.', '']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines))
    print(args.output)

if __name__ == '__main__':
    main()
