"""Portable, escaped HTML reports from saved solver results (no solve required)."""
import html
import json
from pathlib import Path

def render_report(result, *, title="NIRYUKTI solve report"):
    if not isinstance(result, dict) or not isinstance(result.get("status"), str):
        raise ValueError("Report input must be a saved solve result with a status")
    rows = []
    certificate = result.get("verification_certificate") or {}
    verification = certificate.get("verification", {}) if isinstance(certificate, dict) else {}
    is_mip = result.get("problem_type") in ("MILP", "MIQP") or bool(result.get("mip"))
    if is_mip:
        rows.append("<section><h2>Recorded integer-solution evidence</h2><p>Certificate status: " +
                    html.escape(str(verification.get("status", "not supplied"))) +
                    ". Incumbent feasibility and relaxation-bound gap evidence are distinct from a full tree proof. "
                    "The complete search tree has not been independently replayed.</p></section>")
    for section in ("accuracy", "performance", "selection", "hardware", "mip", "model"):
        fields = result.get(section, {})
        if not isinstance(fields, dict):
            continue
        values = "".join("<tr><th>" + html.escape(str(k).replace("_", " ")) + "</th><td>" +
                         html.escape(json.dumps(v, ensure_ascii=False, allow_nan=False)) + "</td></tr>"
                         for k, v in fields.items() if k not in ("primal", "dual"))
        if values:
            rows.append("<section><h2>" + html.escape(section.title()) + "</h2><table>" + values + "</table></section>")
    accuracy = result.get("accuracy", {})
    objective = result.get("objective", accuracy.get("objective") if isinstance(accuracy, dict) else None)
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>''' + html.escape(title) + '''</title><style>
body{background:#f4f1e9;color:#242821;font:16px/1.6 system-ui;margin:0}main{max-width:1100px;margin:auto;padding:40px 24px}
h1{font-size:clamp(30px,5vw,54px);line-height:1.1}header{border-bottom:2px solid #a78b55;padding-bottom:24px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:22px}section{background:#fffdf7;padding:24px;border:1px solid #dedace;border-radius:12px;margin-top:24px;overflow:auto}
th{text-align:left;font-weight:500;padding:8px 16px 8px 0}td{padding:8px;overflow-wrap:anywhere}p{max-width:80ch}small{color:#5f6459}table{width:100%;border-collapse:collapse}tr{border-bottom:1px solid #e9e5da}
@media print{body{background:white}main{padding:0}section{break-inside:avoid}}
</style><main><header><small>NIRYUKTI · INDEPENDENT SPARSE OPTIMIZATION</small><h1>''' + html.escape(title) + "</h1><p>Status: <strong>" + html.escape(result["status"]) + "</strong> · Objective: " + html.escape(str(objective)) + '''</p></header>
<p>This report summarizes the supplied result. Rendering is not a new independent verification, an authenticity signature, or a replay of an integer search proof. Keep the original model and result JSON for verification.</p><div class="grid">''' + "".join(rows) + "</div></main></html>"

def write_report(result_path, output, *, title="NIRYUKTI solve report"):
    result = json.loads(Path(result_path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    Path(output).write_text(render_report(result, title=title), encoding="utf-8")
