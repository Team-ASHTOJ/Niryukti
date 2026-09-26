#!/usr/bin/env python3
"""Dolan–Moré profiles: failures stay in the denominator at infinite ratio.

Inputs are runs.csv files from benchmark/run.py, optionally LABEL=PATH. A solver/instance is successful
only when every measured repetition is OPTIMAL with a positive finite time.
No missing cells or unsuccessful repetitions are silently discarded.
"""
import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path


def profiles(rows, metric='end_to_end_seconds'):
    rows = [r for r in rows if str(r.get('warmup', False)).lower() != 'true']
    instances = sorted({(r.get('dataset_sha256') or r['instance']) for r in rows})
    solvers = sorted({r['solver'] for r in rows})
    groups = {}
    for row in rows:
        key = (row.get('dataset_sha256') or row['instance'], row['solver'])
        groups.setdefault(key, []).append(row)
    times = {}
    for key, group in groups.items():
        values = []
        for row in group:
            try:
                value = float(row[metric])
            except (ValueError, TypeError, KeyError):
                break
            if row['status'] != 'OPTIMAL' or not math.isfinite(value) or value <= 0:
                break
            values.append(value)
        if len(values) == len(group):
            times[key] = statistics.median(values)
    ratios = {s: {} for s in solvers}
    for instance in instances:
        best = min((times.get((instance, s), math.inf) for s in solvers), default=math.inf)
        for solver in solvers:
            value = times.get((instance, solver), math.inf)
            ratio = value / best if math.isfinite(value) and math.isfinite(best) else math.inf
            ratios[solver][instance] = ratio if math.isfinite(ratio) else None
    thresholds = sorted({1.0} | {r for values in ratios.values() for r in values.values() if r is not None})
    curves = {s: [[t, sum(r is not None and r <= t for r in ratios[s].values()) / len(instances)]
                  for t in thresholds] for s in solvers} if instances else {}
    return dict(metric=metric, instances=instances, ratios=ratios, curves=curves,
                success_rule='Every recorded repetition must be OPTIMAL with positive finite time; missing/failing cells have infinite ratio (JSON null).')


def write_report(data, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'profile.json').write_text(json.dumps(data, indent=2, allow_nan=False))
    maximum = max([2.] + [p[0] for c in data['curves'].values() for p in c])
    def px(t):
        return 65 + 660 * math.log2(t) / math.log2(maximum)
    colors = ['#0369a1', '#b45309', '#047857', '#be123c', '#7e22ce', '#334155']
    plot = '<rect width="900" height="440" fill="white"/>'
    for fraction in (0, .25, .5, .75, 1):
        y = 365 - 320 * fraction
        plot += f'<path d="M65 {y} H725" stroke="#ddd"/><text x="20" y="{y+5}">{fraction:g}</text>'
    for tick in sorted({1., 2., maximum}):
        plot += f'<text x="{px(tick)}" y="390">{tick:.3g}</text>'
    for i, (solver, curve) in enumerate(data['curves'].items()):
        color = colors[i % len(colors)]
        path = 'M65 365'
        for threshold, fraction in curve:
            path += f' H{px(threshold):.4f} V{365-320*fraction:.4f}'
        path += ' H725'
        plot += f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
        plot += f'<text x="745" y="{60+25*i}" fill="{color}">{html.escape(solver)}</text>'
    plot += '<text x="210" y="425">Time / fastest successful solver (log scale)</text>'
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 440" role="img" aria-label="Performance profiles">{plot}</svg>'
    (output / 'profile.svg').write_text(svg)
    (output / 'profile.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Performance profiles</title>'
        '<body style="font:16px system-ui;max-width:1100px;margin:32px auto"><h1>Performance profiles</h1>'
        f'<p>Fraction of all {len(data["instances"])} selected instances solved within a time ratio. '
        'Median time across repetitions; every recorded repetition must succeed. Failures and missing solver cells '
        'remain in the denominator. Status-based results require inspecting raw objective and feasibility agreement '
        'before interpreting speed differences.</p>' + svg + '<p><a href="profile.json">Ratios and curves (JSON)</a> · '
        '<a href="profile.svg">SVG</a> · <a href="https://arxiv.org/abs/cs/0102001">Dolan–Moré paper</a></p></body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', help='CSV path or LABEL=CSV path; multiple inputs become separate configurations')
    parser.add_argument('--metric', choices=['end_to_end_seconds', 'process_wall_seconds'], default='end_to_end_seconds')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = []
    first_path = None
    for source in args.runs:
        label, separator, filename = source.partition('=')
        path = Path(filename if separator else source)
        if first_path is None:
            first_path = path
        prefix = label if separator else path.parent.name if len(args.runs)>1 else ''
        with path.open(newline='') as stream:
            group = list(csv.DictReader(stream))
        for row in group:
            if prefix:
                row['solver'] = prefix + '/' + row['solver']
        rows.extend(group)
    data = profiles(rows, args.metric)
    if not data['instances']:
        parser.error('No measured instances')
    write_report(data, args.output or first_path.parent)



if __name__ == '__main__':
    main()
