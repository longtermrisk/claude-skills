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

## Nested projects and budgets

A project can sit under another one (e.g. one project per agent workspace or per run, under
the project that pays for it). Spend is booked on the most specific project and **adds up
the tree**: the dashboard shows top-level projects, each including its sub-projects, and
opens a project to show them.

A project can have a **lifetime budget** in USD that covers itself and everything below it.
The dashboard only reports it (`remaining_usd` is the tightest of the project's own budget
and those above it); whoever hands out the keys enforces it. While a budget is open, LiteLLM,
Tinker and Modal are re-read every 10 minutes instead of every 6 hours; OpenWeights keeps the
6-hour cycle. Numbers still arrive with a delay (`synced_at`, and Tinker's own data lags
about an hour), so a limit can be overshot by what is spent in between. Mark a finished
project `closed` so it is no longer watched.

```python
requests.post(f"{B}/api/projects", headers=H, json={"name": "backlog-run-42", "parent": "backlog",
                                                    "budget_usd": 1000}).raise_for_status()
s = requests.get(f"{B}/api/projects/backlog-run-42/spend", headers=H).json()
# {"spent_usd": 61.0, "budget_usd": 1000, "remaining_usd": 939.0, "over_budget": false,
#  "by_provider": [{"provider": "tinker", "usd": 43.0, ...}, {"provider": "litellm", "usd": 18.0, ...}],
#  "limits": [...], "children": [...], "synced_at": {"tinker": "...", "litellm": "..."}}
```

To send spend to a sub-project with no rule, name things after it: Tinker session metadata
`ServiceClient(user_metadata={"project": "<project>"})` (Tinker's own project ids are UUIDs,
so they need a rule), Modal `MODAL_ENVIRONMENT=<project>` or tag `project=<project>`,
RunPod pod `<project>/...`. Credentials minted for one project are that project's ahead of
every rule: a LiteLLM key with `metadata: {"costs_project": "<project>"}`, Tinker session
metadata `{"costs_project": "<project>"}` (r-docs sets it for budgeted workspaces), and an OpenWeights
API token **named after the project** (an org's spend is divided between its tokens by what
their jobs cost; jobs submitted without a token stay with the org).

### If `COSTS_PROJECT` is set in your environment

You are working in an r-docs workspace with a compute budget, and `COSTS_PROJECT` is its
project here. Nothing to do for the usual providers, as long as you use the credentials and
settings from your environment as they are:

- **LiteLLM, OpenWeights**: the workspace has its own key / token, counted and limited.
- **Tinker**: requests go through r-docs (`TINKER_BASE_URL`), which tags every session with
  the project and refuses them once the budget is used up.
- **Modal**: `MODAL_ENVIRONMENT` is the workspace's own Modal environment. Everything in it is
  counted for the workspace, and **every app in it is stopped when the budget is used up**
  (volumes and secrets stay). Do not pass another `environment_name=` / `--env`: spend
  elsewhere is counted against nobody and is not stopped. Volumes, secrets and deployed apps
  of other environments are not visible from yours; create what you need in your own.

Only where one of these is missing (no `TINKER_BASE_URL`, no `MODAL_ENVIRONMENT`) is the key a
shared one, and you tag the usage yourself:

```python
import os, tinker, modal
project = os.environ["COSTS_PROJECT"]
service = tinker.ServiceClient(user_metadata={"project": project})
app = modal.App("my-app", tags={"project": project})
```

RunPod pods: name them `f"{project}/<anything>"`. The workspace's budget and spend so far:
ask r-docs (`get_workspace_budget`), or `GET /api/projects/$COSTS_PROJECT/spend` here.

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
- Or give a project its own Modal environment: `POST /api/projects/<project>/modal-environment`
  creates it (named after the project). Run with `MODAL_ENVIRONMENT=<project>`; everything in
  it is the project's, and `POST .../modal-environment/stop` stops every app in it.

### LiteLLM
- Every LiteLLM team automatically becomes a project of the same name at the next LiteLLM
  sync (every 6 h; or Connections → litellm → Sync to do it right away). To group teams, add a rule (e.g. `litellm_team = *-srf` → `srf`) first.

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
| `GET /api/summary?rows=project\|provider&month_from=YYYY-MM&month_to=&vat=incl\|excl&project=&rollup=top\|none` | spend per group and month; project rows include sub-projects, `project=X` shows X and its direct sub-projects |
| `GET/POST /api/manual` `{month, provider, usd, includes_vat, project?, note?}`, `PATCH/DELETE /api/manual/{id}` | manual entries (bills without an API) |
| `GET /api/vat`, `PUT /api/vat/{supplier}` `{rate, status, note}` | VAT rate per billing supplier |
| `GET /api/lines?project=&provider=&month_from=&q=&limit=&exact=` | cost lines with attributes (`project` includes sub-projects unless `exact=true`) |
| `GET /api/unassigned` | unassigned spend grouped by attribute value |
| `GET /api/fields` | attributes rules can use, with values and spend |
| `GET /api/projects` | projects as a tree, with lifetime spend, budget and what is left |
| `POST /api/projects` `{name, description?, parent?, budget_usd?, external_ref?}` | create (an existing one is returned unchanged) |
| `PATCH /api/projects/{name}` `{name?, description?, parent?, budget_usd?, closed?, external_ref?}` | change only the fields sent (`parent: null` = top level, `budget_usd: null` = no budget) |
| `DELETE /api/projects/{name}` | delete; its sub-projects move up one level |
| `GET /api/projects/{name}/spend?month_from=&month_to=&vat=` | spend in total and per provider incl. sub-projects, budget, `remaining_usd`, `synced_at` |
| `GET /api/budgets?include_closed=` | every project a budget applies to, with spend and what is left |
| `POST /api/projects/{name}/modal-environment` `{connection?}` or `{token_id, token_secret}` | give the project its own Modal environment (created if missing); a Modal workspace not connected yet is added as a connection |
| `GET /api/projects/{name}/modal-environment` | its Modal environments and the apps running in them now |
| `POST /api/projects/{name}/modal-environment/stop` | stop every app in them (volumes and secrets stay) |
| `DELETE /api/projects/{name}/modal-environment?delete_environment=` | stop the apps and drop the link; `delete_environment=true` deletes the environment at Modal with its volumes (irreversible) |
| `GET /api/rules`, `POST /api/rules` `{field, pattern, splits:[{project, share}], note}` | rules (same field+pattern replaces) |
| `PATCH/DELETE /api/rules/{id}`, `GET /api/rules/preview?field=&pattern=` | edit, delete, preview |
| `GET/POST /api/connections` `{provider, name, secret}` | provider accounts (keys write-only) |
| `PUT /api/connections/{id}/assign` `{splits:[...]}` | assign a whole account ([] = unassign) |
| `GET /api/retention/hf/storage` | HF storage by project / OW org / model, autodelete status |
| `GET/POST/DELETE /api/retention/hf/keep` `{pattern, keep_until?, note?}` | HF keep list |
| `GET /api/retention/tinker/storage` | Tinker checkpoint storage by account / project / user |
| `GET/POST/DELETE /api/retention/tinker/keep`, `GET .../tinker/check?path=` | Tinker keep list |
| `GET /api/retention/tinker/actions?action=would_delete` | latest cleanup run |

Rule attributes: `ow_org`, `ow_key` (OpenWeights API token name), `litellm_team`, `litellm_key`, `litellm_user`, `tinker_project`,
`tinker_user`, `tinker_account_user` (`"<account>: <user>"`), `tinker_meta.<key>`, `modal_env`, `modal_app`, `modal_tag.<key>`,
`runpod_account`, `runpod_pod_name`, `hf_repo`, `project_hint`, `connection`, `provider`.
