'use strict';
function escape(value) {
  return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function renderReport(result, {title='NIRYUKTI solve report'}={}) {
  if (!result || typeof result.status !== 'string') throw new TypeError('Expected a saved solve result');
  const verification=result.verification_certificate?.verification || {};
  const evidence=(['MILP','MIQP'].includes(result.problem_type)||result.mip)?`<section><h2>Recorded integer-solution evidence</h2><p>Certificate status: ${escape(verification.status || 'not supplied')}. Incumbent feasibility and relaxation-bound gap evidence are distinct from a full tree proof. The complete search tree has not been independently replayed.</p></section>`:'';
  const sections=['accuracy','performance','selection','hardware','mip','model'].map(section=>{
    const values=result[section];if (!values || typeof values!=='object' || Array.isArray(values)) return '';
    return `<section><h2>${escape(section)}</h2><table>${Object.entries(values).map(([k,v])=>`<tr><th>${escape(k)}</th><td>${escape(JSON.stringify(v))}</td></tr>`).join('')}</table></section>`;
  }).join('');
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(title)}</title><style>body{background:#f4f1e9;color:#242821;font:16px/1.6 system-ui;margin:0}main{max-width:1000px;margin:auto;padding:32px}section{padding:24px;background:#fffdf7;border:1px solid #dedace;border-radius:12px;margin:20px 0;overflow:auto}th{text-align:left;padding:8px 24px 8px 0}td{padding:8px;overflow-wrap:anywhere}table{width:100%}@media print{body{background:white}section{break-inside:avoid}}</style><main><h1>${escape(title)}</h1><p>Status: <strong>${escape(result.status)}</strong></p><p>This report summarizes supplied data; it is not a new independent verification or an authenticity signature. Keep the model and original result for verification.</p>${evidence}${sections}</main></html>`;
}
module.exports={renderReport};
