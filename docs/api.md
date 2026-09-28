# HTTP API and solve reports

The installed Python CLI includes `niryukti serve` and `niryukti report`.
The engine's native C++ CLI remains focused on solve/inspect/explain/verify;
from a checkout use `PYTHONPATH=python python -m niryukti` for service/report commands.

```sh
export NIRYUKTI_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
niryukti serve --port 8090 --workers 2 --max-time 300
```

Requests use `Authorization: Bearer $NIRYUKTI_API_TOKEN`. The service binds to
127.0.0.1 by default. For remote deployments, put HTTPS and access control in
front of it. Tokens/model bodies are not logged. Payloads are limited to 8 MiB;
worker slots and per-solve time limits are bounded. There is no remote binary
selection, filesystem-model loading, shell execution or arbitrary option forwarding.

| Route | Body | Response |
|---|---|---|
| GET `/v1/health` | none | service/version/resource limits |
| POST `/v1/solve` | `model` (native JSON), optional `options` | full solver result |
| POST `/v1/report` | `result`, optional `title` | standalone HTML |

Solve options: `method`, `device`, `tol`, `threads`, `time_limit`, `iterations`.
Default `method=auto`, `device=auto`. Requests are synchronous and solver
processes are isolated. Disconnecting an HTTP client does not cancel an existing
solve; its time limit still applies. Use the dashboard's job API for live job
monitoring/cancellation. This is a bounded local service, not a multi-user scheduler.

```sh
curl http://127.0.0.1:8090/v1/health \
  -H "Authorization: Bearer $NIRYUKTI_API_TOKEN"
# request.json: {"model": {...}, "options": {"method":"auto","device":"auto"}}
curl http://127.0.0.1:8090/v1/solve \
  -H "Authorization: Bearer $NIRYUKTI_API_TOKEN" \
  -H 'Content-Type: application/json' --data-binary @request.json > result.json
niryukti report result.json --output report.html
niryukti verify model.json result.json
```

Reports contain status, objective, feasibility/KKT, timing, algorithm/device
selection and available MIP data. They escape supplied text, work offline, print
cleanly and require no JavaScript. Rendering a supplied result is not independent
verification or authentication. Keep the original model/result JSON and use
`verify` or evidence bundles when verification matters. Limit statuses remain
visible; a report never relabels a candidate as optimal.
