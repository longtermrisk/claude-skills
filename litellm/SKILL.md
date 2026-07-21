---
name: litellm
description: Use this skill when implementing LLM API interactions in new projects or scripts. The org runs a LiteLLM proxy that exposes OpenAI, Anthropic, Gemini, OpenRouter, and local models through a single OpenAI-compatible endpoint, with opt-in response caching that makes research scripts resumable and idempotent. Do not use this skill if the project already has an established LLM client library.
---

# LiteLLM Proxy

All LLM calls go through the org's LiteLLM proxy at `https://litellm.nielsrolf.com`. It speaks the OpenAI API, so use the standard `openai` Python SDK — no custom client library needed.

## Quick Start

```python
from openai import OpenAI
import os

API_KEY = os.environ['LITELLM_API_KEY']  # already set in the environment
BASE_URL = "https://litellm.nielsrolf.com"
MODEL = "anthropic/claude-opus-4-8"

# NOTE: Cloudflare blocks the OpenAI SDK's default User-Agent ("OpenAI/Python ...")
# with a 403 "Your request was blocked." Overriding the UA avoids it. Remove this
# once Bot Fight Mode is disabled / a WAF skip rule is added for litellm.nielsrolf.com.
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
