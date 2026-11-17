---
name: compute-costs
description: Track and attribute compute spend (OpenWeights, RunPod, LiteLLM, Tinker, Modal, Hugging Face storage) to projects at https://costs.nielsrolf.com, and protect models/checkpoints from automatic cleanup. Use when starting GPU/LLM/Tinker work for a project, when asked what something costs, or before/after pushing models or checkpoints that must be kept.
---

# Compute costs: costs.nielsrolf.com

One dashboard for all our compute spend, by **month × provider × project**, plus the
**retention** systems that delete unused Hugging Face models and Tinker checkpoints.

- Web: https://costs.nielsrolf.com with a personal account (ask Niels for an invite link)
- JSON API: same host, `Authorization: Bearer $COSTS_TOKEN`, a personal API token you create
  on the Account tab (shown once; it acts as you)
- MCP: `https://costs.nielsrolf.com/mcp` (streamable HTTP, same bearer token)
- This file: https://costs.nielsrolf.com/skill.md

```bash
claude mcp add --transport http costs https://costs.nielsrolf.com/mcp \
  --header "Authorization: Bearer $COSTS_TOKEN"
```

## How spend reaches a project

Each provider account is a **connection** (an API key stored on the server). Every cost
line carries attributes, and the first of these that applies decides its project:

1. **A rule** `attribute = value` (or a `*` pattern) → one project, or split by share
   (e.g. 50 % A / 50 % B). Exact values beat patterns.
2. **A naming convention** (below): if the line names an existing project, it goes there.
3. **A whole-connection assignment** (fallback for everything else in that account).

Anything left over is **Unassigned**; the dashboard's Unassigned tab lists it with
one-click assignment. Rules apply to all past months as soon as they are added.

## What each of us should do

### OpenWeights
- Each OpenWeights org belongs to a project. Assign the org once (Connections tab, or a
  rule on `ow_org`): that maps its GPU workers, **its RunPod pods**, and **every Hugging
  Face model its jobs push**.
- Nothing else needed: org keys and their RunPod accounts are imported centrally.

### Tinker
- **Use Tinker projects.** Create one per dashboard project and start sessions in it:
  `tinker.ServiceClient(project_id="<project>")` or `export TINKER_PROJECT_ID=<project>`.
  If the Tinker project id equals the dashboard project's name, it is mapped with no
  rule; otherwise add a rule `tinker_project = <id>`. Alternatively tag the session:
  `ServiceClient(user_metadata={"project": "<project>"})`.
- **Per person:** every Tinker cost line names the user who created the session, so a rule on
  "Tinker user in account" (e.g. `CLR: Jane Doe` → `my-project`) maps one person's usage in one
  Tinker org. Checkpoint storage is billed to the org, so it is split between users by their
  share of stored checkpoints (an estimate).
- **Add your Tinker API key** on the Connections tab (it needs billing view access in
  the Tinker org), then assign it to your project if all of its usage belongs to one.
- **Checkpoints are deleted automatically** once the cleanup is switched on (see
  Retention). Protect what you need *before* that.

### RunPod (outside OpenWeights)
- **Add your RunPod API key** on the Connections tab (read access is enough).
- **Name pods `<project>/<anything>`**, e.g. `srf/vllm-eval`. The part before `/` is the
  project. Names are recorded every 5 minutes while a pod is alive (RunPod's bill
  only has pod ids), so a pod living under 5 minutes may stay unnamed.
- Or assign the whole account to a project on the Connections tab.

### Modal
- Add a token (`{"token_id": "...", "token_secret": "..."}`) on the Connections tab.
- Tag apps `project=<project>`; or assign the workspace / a rule on `modal_env`, `modal_app`.

### LiteLLM
- Every LiteLLM team automatically becomes a project of the same name the first time it
  has spend. To group teams, add a rule (e.g. `litellm_team = *-srf` → `srf`) first.

### Hugging Face (`longtermrisk` org)
- **Every model must belong to a project** so storage has an owner. Models pushed by an
  OpenWeights job inherit the project of that org automatically. Others need a rule on
  `hf_repo` (e.g. `longtermrisk/myproj-*` → `myproj`) or an "assign" click on the
  Retention page.
- Storage is an *estimate* (HF has no billing API): size × $/TB-month from a daily snapshot.

## VAT

We are a UK non-profit and cannot reclaim VAT, so the dashboard shows amounts **including
VAT** by default (toggle "excl. VAT" to compare with provider consoles). Provider APIs report
net amounts; each billing supplier has a VAT rate (Connections → VAT; `GET/PUT /api/vat/{supplier}`)
that is added on top. For LiteLLM the supplier is the upstream provider; OpenWeights compute
is billed by RunPod. Manual entries say whether their amount already includes VAT
(`"includes_vat": true` for a bill, `false` for net usage).

## Retention (automatic deletion)

**Hugging Face**: every night at 04:00 a `longtermrisk` model is deleted when it had
**0 downloads in the last 30 days** and is **≥ 4 GB and older than 14 days** or
**< 4 GB and older than 60 days**. Never deleted: names containing `noautodelete`, and
anything on the keep list (optionally only until a date). Deletion is permanent.

**Tinker**: nightly at 04:30, per account: training checkpoints older than 14 days and
sampler checkpoints older than 60 days are deleted, except keep-list matches, anything
with `noautodelete` in its run id/path/metadata, public checkpoints, checkpoints with a
Tinker TTL, and each run's newest checkpoint per type. A global **Auto-clean Tinker**
switch must be on (it starts off) or every run is a dry run; the Retention page shows
what a dry run would delete.

```python
import os, requests
H = {"Authorization": f"Bearer {os.environ['COSTS_TOKEN']}"}
B = "https://costs.nielsrolf.com"

# after push_to_hub: keep this model (optionally until a date)
requests.post(f"{B}/api/retention/hf/keep", headers=H, timeout=15,
              json={"pattern": "longtermrisk/my-model", "keep_until": "2027-03-01", "note": "paper"}).raise_for_status()

# keep a Tinker checkpoint, or a whole run, or a whole Tinker project
requests.post(f"{B}/api/retention/tinker/keep", headers=H, timeout=15,
              json={"pattern": "tinker://RUN-ID/sampler_weights/final", "note": "final model"}).raise_for_status()
requests.post(f"{B}/api/retention/tinker/keep", headers=H, timeout=15,
              json={"pattern": "project:my-tinker-project", "keep_until": "2026-12-31"}).raise_for_status()
```

## API reference

All endpoints take/return JSON and need `Authorization: Bearer <personal API token>`. Admin-only:
user management (`/api/users`), the Tinker auto-clean switch and cleanup policies.

| Method & path | Purpose |
|---|---|
| `GET /api/summary?rows=project\|provider&month_from=YYYY-MM&month_to=&vat=incl\|excl` | spend per group and month |
| `GET/POST /api/manual` `{month, provider, usd, includes_vat, project?, note?}`, `PATCH/DELETE /api/manual/{id}` | manual entries (bills without an API) |
| `GET /api/vat`, `PUT /api/vat/{supplier}` `{rate, status, note}` | VAT rate per billing supplier |
| `GET /api/lines?project=&provider=&month_from=&q=&limit=` | cost lines with attributes |
| `GET /api/unassigned` | unassigned spend grouped by attribute value |
| `GET /api/fields` | attributes rules can use, with values and spend |
| `GET/POST /api/projects`, `DELETE /api/projects/{name}` | projects |
| `GET /api/rules`, `POST /api/rules` `{field, pattern, splits:[{project, share}], note}` | rules (same field+pattern replaces) |
| `PATCH/DELETE /api/rules/{id}`, `GET /api/rules/preview?field=&pattern=` | edit, delete, preview |
| `GET/POST /api/connections` `{provider, name, secret}` | provider accounts (keys write-only) |
| `PUT /api/connections/{id}/assign` `{splits:[...]}` | assign a whole account ([] = unassign) |
| `GET /api/retention/hf/storage` | HF storage by project / OW org / model, autodelete status |
| `GET/POST/DELETE /api/retention/hf/keep` `{pattern, keep_until?, note?}` | HF keep list |
| `GET /api/retention/tinker/storage` | Tinker checkpoint storage by account / project / user |
| `GET/POST/DELETE /api/retention/tinker/keep`, `GET .../tinker/check?path=` | Tinker keep list |
| `GET /api/retention/tinker/actions?action=would_delete` | latest cleanup run |

Rule attributes: `ow_org`, `litellm_team`, `litellm_key`, `litellm_user`, `tinker_project`,
`tinker_user`, `tinker_account_user` (`"<account>: <user>"`), `tinker_meta.<key>`, `modal_env`, `modal_app`, `modal_tag.<key>`,
`runpod_account`, `runpod_pod_name`, `hf_repo`, `project_hint`, `connection`, `provider`.
