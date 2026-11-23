---
name: litellm
description: Use this skill when implementing LLM API interactions in new projects or scripts. The org runs a LiteLLM proxy that exposes OpenAI, Anthropic, Gemini, OpenRouter, and local models through a single OpenAI-compatible endpoint, with opt-in response caching that makes research scripts resumable and idempotent. Also covers the 2-minute timeout on long non-streaming calls and OpenRouter batches through the proxy. Do not use this skill if the project already has an established LLM client library.
---

# LiteLLM Proxy

All LLM calls go through the org's LiteLLM proxy at `https://litellm.longtermrisk.org`. It speaks the OpenAI API, so use the standard `openai` Python SDK — no custom client library needed.

## Quick Start

```python
from openai import OpenAI
import os

API_KEY = os.environ['LITELLM_API_KEY']  # already set in the environment
BASE_URL = "https://litellm.longtermrisk.org"
MODEL = "anthropic/claude-opus-4-8"

# NOTE: on the old hostname Cloudflare blocked the OpenAI SDK's default User-Agent
# ("OpenAI/Python ...") with a 403 "Your request was blocked." A request with that
# UA was not blocked on litellm.longtermrisk.org on 2026-11-23; the override is
# harmless and kept in case the rule comes back.
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    default_headers={"User-Agent": "litellm-client/1.0"},
)

# max_tokens generous: reasoning models (gpt-5, o-series) spend tokens thinking
# before any visible text, so a tiny budget can return empty content.
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Reply with exactly: yo it works"}],
    max_tokens=200,
)

print("model :", resp.model)
print("reply :", resp.choices[0].message.content)
print("tokens:", resp.usage.total_tokens)
```

For async code, use `openai.AsyncOpenAI` with the same arguments. The SDK retries transient failures automatically; raise `max_retries` (e.g. `OpenAI(..., max_retries=5)`) for long batch jobs.

## Model Names

Model names are `provider/model` and are forwarded via wildcard routes — any model the provider offers works without proxy config changes:

| Prefix | Example | Notes |
|---|---|---|
| `openai/*` | `openai/gpt-5`, `openai/gpt-5-mini` | |
| `anthropic/*` | `anthropic/claude-opus-4-8` | |
| `gemini/*` | `gemini/gemini-2.5-pro` | |
| `openrouter/*` | `openrouter/openai/gpt-5` | any OpenRouter model |
| `local/*` | `local/qwen3.6-27b` | llama-server on the host; one model loaded at a time |
| `embedding` | `embedding` | local Qwen3-Embedding-4B, via `client.embeddings.create` |

## Long requests: use `stream=True`

The proxy sits behind Cloudflare, which answers **524** when no byte of the response has arrived for about 125 seconds. A non-streaming completion sends nothing until it is complete, so a single call that thinks or writes for more than ~2 minutes fails ("proxy idle timeout"), and a retry fails the same way. This limit cannot be raised on the server.

- **Fix: `stream=True`** and assemble the chunks. Bytes flow, so the limit never applies:

  ```python
  stream = client.chat.completions.create(model=MODEL, messages=messages, stream=True)
  text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
  ```

- A 524 with an HTML body and a `cf-ray` header is this limit; a JSON error mentioning `Timeout` is LiteLLM or the provider.
- For large offline jobs on OpenRouter models, batches (below) avoid the problem altogether.

## OpenRouter batches (asynchronous, about half price)

The proxy forwards OpenRouter's Batch API at `/openrouter/batches` (**not** `/openrouter/v1/batches`, and not the OpenAI SDK's `client.batches`, which speaks a different, file-based format):

| Call | Does |
|---|---|
| `POST /openrouter/batches` | submit; the requests are inline in the body |
| `GET /openrouter/batches/{id}` | status, and the results inline once it has ended |
| `GET /openrouter/batches` | list your batches |
| `DELETE /openrouter/batches/{id}` | delete a finished batch |

```python
import httpx, os, time

http = httpx.Client(
    base_url="https://litellm.longtermrisk.org/openrouter",
    headers={"Authorization": f"Bearer {os.environ['LITELLM_API_KEY']}"},
    timeout=120,
)
batch = http.post("/batches", json={
    "endpoint": "/v1/chat/completions",
    "model": "google/gemini-4-carbon",        # OpenRouter slug, no "openrouter/" prefix
    "requests": [
        {"custom_id": f"q{i}", "body": {"messages": [{"role": "user", "content": q}]}}
        for i, q in enumerate(questions)
    ],
}).raise_for_status().json()

while True:
    b = http.get(f"/batches/{batch['id']}").raise_for_status().json()
    if b["status"] in ("completed", "failed", "expired", "cancelled"):
        break
    time.sleep(60)
results = b["results"]                        # match to your requests by custom_id
```

Things to know:

- **Access is granted per key.** Without the grant the route answers 403 ("only permits llm_api_routes" / "Configure `allowed_passthrough_routes`"); ask Niels to grant your key.
- The body is **OpenRouter's format**, passed through unchanged: check https://openrouter.ai/docs/batch-quickstart for the exact fields. Only models listed at https://openrouter.ai/models?variant=batch are accepted. To pin providers, put `"provider": {"only": [...]}` in the batch.
- **Your key only sees its own batches**; a rotated key loses access to the old key's batches. Save the results when you fetch them (OpenRouter keeps them 30 days).
- **Cost is charged to your key when the batch has ended** (normally when you fetch the results), and shows up in the costs dashboard like any other LiteLLM spend. Nothing is reserved while it runs, so a large batch can overshoot a budget.
- No response caching, no `seed`-based resume: keep the batch id and your `custom_id`s.
- New on 2026-11-23 and not yet exercised with a real batch end to end; report anything odd.

## Response Caching

The proxy has a Redis-backed response cache, **off by default** — a request is only cached (and only served from cache) when it opts in:

```python
resp = client.chat.completions.create(
    model="openai/gpt-5-mini",
    messages=messages,
    seed=42,                                     # part of the cache key
    extra_body={"cache": {"use-cache": True}},   # opt in to caching
)
```

### How it works

The cache key is a hash of the serialized request: `model`, `messages`, `tools`, `tool_choice`, `seed`, `temperature`, `response_format`, `max_tokens`, and all other standard OpenAI params. Identical request → cache hit; change anything (including `seed`) → fresh completion. Cached entries have **no TTL** — they persist until the cache is cleared.

This makes research scripts idempotent: give each sample a distinct `seed`, opt in to caching, and an interrupted script can simply be re-run — every request that already completed is served from cache instantly, and the script picks up where it left off. To resample a datapoint, bump its seed.

Cache hits return an `x-litellm-cache-key` response header (visible via `client.chat.completions.with_raw_response.create(...)` if you need to check).

### Per-request cache controls

All go in `extra_body={"cache": {...}}`:

- `{"use-cache": True}` — opt in (required; nothing is cached without it)
- `{"use-cache": True, "namespace": "my-experiment"}` — scope keys to an experiment
- `{"use-cache": True, "ttl": 3600}` — expire this entry after N seconds
- `{"no-cache": True}` — force a fresh completion (ignore any cached entry)
- `{"no-store": True}` — read from cache but don't write

### Caveats

- **`local/*` models**: the cache can't see which model llama-server actually has loaded — always call local models by their real model name (not a generic alias), or a stale cached answer from a previously-loaded model could be returned.
- Provider-specific params outside the OpenAI spec (e.g. `top_k`) are **not** part of the cache key by default. Anthropic's `thinking` param **is** included.
- The cache stores responses, not guarantees of determinism: a cache *miss* with the same seed may still produce a different completion (upstream sampling), so treat the cache as "don't recompute", not "reproducible sampling".
