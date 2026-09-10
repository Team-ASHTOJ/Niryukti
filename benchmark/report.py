#!/usr/bin/env python3
"""Offline report: no CDN, invented values, filtering of failures, or aggregate-only claims."""
import csv
import html
import json
import sys
from pathlib import Path

root=Path(sys.argv[1]);rows=list(csv.DictReader((root/'runs.csv').open()));summary=json.loads((root/'summary.json').read_text())
esc=lambda x:html.escape(str(x))
body=''
for r in rows:
    cls='ok' if r['status']=='OPTIMAL' else 'limit'
    body+='<tr>'+''.join(f'<td class="{cls if k=="status" else ""}">{esc(r[k])}</td>' for k in ['instance','solver','run','status','objective','primal_residual','kkt_error','end_to_end_seconds','process_wall_seconds'])+'</tr>'
finite=[r['median_end_to_end_seconds'] for r in summary if r['median_end_to_end_seconds'] is not None]
maximum=max(finite,default=1) or 1
bars=''
for i,r in enumerate(summary):
    val=r['median_end_to_end_seconds'];y=30+i*30
    width=0 if val is None else 470*val/maximum
    bars+=f'<text x="8" y="{y+14}">{esc(r["instance"]+" / "+r["solver"])}</text><rect x="275" y="{y}" width="{width}" height="20" fill="{("#2dd4bf" if r["optimal_runs"]==r["total_runs"] else "#fbbf24")}"/><text x="{285+width}" y="{y+14}">{"unavailable" if val is None else f"{val:.5f}s"}</text>'
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VANTAGE benchmark report</title><style>
body{background:#0c1322;color:#e2e8f0;font:15px system-ui;margin:32px}h1{letter-spacing:.12em}p{max-width:1000px;line-height:1.6}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;text-align:left;border-bottom:1px solid #334155}th{background:#172033;position:sticky;top:0}.ok{color:#2dd4bf}.limit{color:#fbbf24}svg{background:#172033;border-radius:12px;width:100%;max-width:1100px}svg text{fill:#e2e8f0;font:12px system-ui}a{color:#67e8f9}.scroll{overflow:auto}</style>
<h1>VANTAGE</h1><p>Independent sparse optimization · reproducible prototype measurements</p>
<p>Every selected measured run appears below, including failures and limits. CUDA receives one discarded warmup process per instance. Timings include parsing and device setup/transfers; process wall time additionally includes process startup. Small instances can favor CPU solvers. Amber bars include runs that did not reach optimal status.</p>
<p>LP/QP KKT values are independently recomputed in original units. For MILP, continuous KKT values do not certify tree optimality: inspect the raw incumbent, global bound, and MIP gap. Baseline tolerance conventions differ; objective and feasibility agreement must be read alongside timing.</p>
<p><a href="runs.csv">All measured runs (CSV)</a> · <a href="summary.json">Median summaries</a> · <a href="manifest.json">Machine and configuration</a></p>
'''
page+=f'<h2>Median end-to-end time</h2><svg role="img" aria-label="Per-instance median solver times" viewBox="0 0 900 {60+30*len(summary)}">{bars}</svg><h2>Per-instance results</h2><div class="scroll"><table><thead><tr>'+''.join(f'<th>{h}</th>' for h in ['Instance','Solver','Run','Status','Objective','Primal residual','KKT','End-to-end s','Process s'])+'</tr></thead><tbody>'+body+'</tbody></table></div></html>'
(root/'index.html').write_text(page);print(root/'index.html')
