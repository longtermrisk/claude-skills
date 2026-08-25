---
name: forecasting
description: Trade the internal forecasting market — read tracked prediction-market questions, build clusters of related questions, and submit scenario-table decompositions that are auto-bet with fractional Kelly. Use when asked to forecast, update forecasts, or work the forecasting AI.
---

# Forecasting on the internal market

You are an externally managed forecasting agent. The forecasting service does not run
agents itself — you read its questions over HTTP, do your own research however you
like, and submit a **scenario table**. The service then fits a coherent probability
distribution, trades it in the internal play-money market (fractional Kelly), and
tracks your portfolio value in Bayes Points (everyone starts at 1000).

## Setup

You need two environment variables (a human provides them; tokens are minted on the
service's `/agents` page or with `fai mint`):

- `FAI_BASE_URL` — e.g. `http://<tailnet-ip>:14160`
- `FAI_TOKEN` — your bearer token; send as `Authorization: Bearer $FAI_TOKEN`

Either call the JSON API directly (`curl`) or use the bundled zero-dependency client
`fai_client.py` (stdlib only — copy it anywhere).

## Workflow

1. **See where you stand**: `GET /api/v1/me` — your cash, positions, P&L.
2. **Read the questions**: `GET /api/v1/questions` — tracked questions with the latest
   outside quotes (Polymarket price/bid/ask).
3. **Pick or build a cluster**: `GET /api/v1/clusters` lists sets of related
   questions. Create one with `POST /api/v1/clusters`
   `{"title": "...", "question_ids": ["...", "..."]}` (min 2 questions).
4. **Fetch the cluster contract**: `GET /api/v1/clusters/{id}` — members with quotes,
   `max_latents`, the exact `scenario_order` rule, and your previous decomposition if
   you have one.
5. **Research** the questions with your own tools (web search, code, whatever your
   harness provides).
6. **Submit a scenario table**: `POST /api/v1/clusters/{id}/submit`.

## The scenario table (the data contract)

Invent 2–4 binary **latents** — hidden drivers that the cluster's questions share.
The scenarios are all combinations of latent outcomes, enumerated **in code, not by
you**: first latent = most significant bit, ordered `[all yes, …, all no]`. With
latents A, B the four scenarios are `A&B, A&!B, !A&B, !A&!B` — your `probs` rows must
line up with that order positionally.

```json
{
  "latents": [
    {"name": "escalation", "question": "Does the conflict escalate by June?"},
    {"name": "sanctions", "question": "Are new sanctions imposed?"}
  ],
  "rows": [
    {"id": "<question_id>", "probs": [0.9, 0.7, 0.4, 0.1],
     "confidence": 0.8, "reasoning": "one or two sentences"},
    {"id": "<other_id>", "probs": [0.8, 0.3, 0.6, 0.2], "confidence": 0.7}
  ],
  "note": "optional summary of your reasoning",
  "run_link": "$GDOCS_RUN_URL if you run inside r-docs"
}
```

- Binary questions: `probs` = P(YES | scenario), one number per scenario.
- Categorical/numeric/date questions: use `"distributions"` instead — one canonical
  distribution per scenario (categorical: `{option_id: prob}` summing to 1; numeric /
  date: `{"grid": [...], "cdf": [...]}` on the question's declared grid).
- A constant row (same value in every scenario) is dropped — it carries no
  information. A submission needs ≥2 usable rows.
- **Always pass `run_link`** (from `$GDOCS_RUN_URL` when set) so humans can open the
  session that produced the forecast from the run page.

The response tells you what was accepted, what was dropped, the fit residual, and the
market result. Your bets are then placed automatically — you never place bets
yourself.

## Single-question estimates

For a standalone probability update without a decomposition:
`POST /api/v1/reports` `{"question_id": "...", "distribution": ..., "confidence": 0-1,
"rationale": "..."}`. For a binary question the distribution is
`{"yes": p, "no": 1-p}`.

## Good practice

- Prefer improving an existing cluster you already traded (the endpoint returns
  `my_decomposition`) over spraying new clusters.
- Confidence scales your Kelly stake; use < 0.5 when your research is thin.
- Your score is portfolio value over time, marked at the daily clearing price —
  overconfident tables lose Bayes Points fast.
