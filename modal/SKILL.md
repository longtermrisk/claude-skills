---
name: modal
description: Use this skill when running code on Modal (modal.com) - serverless cloud functions with on-demand GPUs, autoscaling web endpoints, parallel map, volumes, and cron schedules. Trigger for tasks involving Modal apps, serverless GPU jobs, massively parallel fan-out, hosted demos/endpoints, or when deciding between Modal and OpenWeights for a GPU workload.
---

# Modal

Modal is a serverless compute platform: you write Python functions, decorate them, and they run in the cloud on demand — including on GPUs — with per-second billing and scale-to-zero. It complements OpenWeights (see "Modal vs OpenWeights" below).

## Setup

```bash
pip install modal
modal setup   # browser-based auth; creates ~/.modal.toml
```

## Core Programming Model

```python
import modal

app = modal.App("example")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers")   # uv_pip_install is the recommended installer
)

@app.function(image=image, gpu="A100-80GB", timeout=3600)
def train(config: dict) -> str:
    ...
    return "done"

@app.local_entrypoint()
def main():
    print(train.remote({"lr": 1e-4}))       # runs in the cloud
```

- `modal run script.py` — ephemeral run of the local entrypoint (the iteration loop; containers boot in ~1s)
- `modal deploy script.py` — persistent deployment (stable web URLs, cron schedules)
- `modal serve script.py` — ephemeral web app with live-reload during development
- Custom base images: `modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")`, plus `.env({...})`, `.run_commands(...)`

## GPUs and Pricing

Request with `gpu="H100"`, multi-GPU with `gpu="H100:8"` (max 8 per container), fallback list `gpu=["H100", "A100-40GB:2"]` (most-preferred first).

| GPU | ~$/hr | | GPU | ~$/hr |
|---|---|---|---|---|
| T4 | $0.59 | | A100-80GB | $2.50 |
| L4 | $0.80 | | H100 | $3.95 |
| A10 | $1.10 | | H200 | $4.54 |
| L40S | $1.95 | | B200 | $6.25 |
| A100-40GB | $2.10 | | B300 | $7.10 |

Billing is per-second of actual runtime (CPU ~$0.047/core/h and RAM ~$0.008/GiB/h metered separately); nothing is charged when scaled to zero. The free Starter plan includes $30/month of credits (10 concurrent GPUs); academics can apply for up to $10k credits.

Gotchas: bare `"H100"` may be silently upgraded to an H200 at no extra cost — pin with `"H100!"` for reproducible benchmarks. `"A100"` may get 80GB. The name is `A10`, not `A10G`. Pinning a `region=...` adds a 1.5–1.75x price multiplier; leave it unset unless you must.

## Key Features

- **Volumes** (persistent storage): `vol = modal.Volume.from_name("weights", create_if_missing=True)`, mount via `volumes={"/data": vol}`. Write-once/read-many semantics — ideal for HF caches and checkpoints. Call `vol.commit()` after writes (auto-commits also run every few seconds).
- **Secrets**: `modal secret create hf HF_TOKEN=...` then `secrets=[modal.Secret.from_name("hf")]` — injected as env vars.
- **Parallel map**: `results = f.map(inputs, return_exceptions=True)` fans out to ~1,000 concurrent containers (25k inputs per call). `f.spawn(x)` is fire-and-forget with a poll-able handle (up to 1M pending).
- **Class-based GPU servers**: `@app.cls(gpu=...)` with `@modal.enter()` (load the model once per container), `@modal.method()`, and `@modal.concurrent(max_inputs=N)` for request batching. Autoscaling knobs: `min_containers`, `max_containers`, `scaledown_window` (idle timeout, ≤20 min).
- **Web endpoints**: `@modal.fastapi_endpoint()` / `@modal.asgi_app()` / `@app.server(port=...)` → `https://<workspace>--<app>-<function>.modal.run`, with optional proxy auth.
- **Schedules**: `@app.function(schedule=modal.Cron("0 6 * * *"))` for recurring jobs.
- **Sandboxes**: `modal.Sandbox.create(...)` for running untrusted / LLM-generated code.
- **Timeouts & retries**: default timeout is only 300s — set `timeout=` explicitly (max 24h) for training runs; `retries=3` for flaky steps.

## Common Patterns

- **vLLM / OpenAI-compatible server**: official example at https://modal.com/docs/examples/llm_inference — an `@app.server(gpu="H200", port=8000, ...)` class that launches `vllm serve` in `@modal.enter()`, with HF + vLLM caches on Volumes.
- **Fine-tuning**: `@app.function(gpu=..., timeout=4*3600, volumes={"/ckpt": vol})` run via `modal run`; official examples for Unsloth (`/docs/examples/unsloth_finetune`), axolotl, GRPO with verl/trl, and diffusion LoRA.
- **Batch inference / embarrassingly parallel work**: `.map()` over inputs, or a `@modal.concurrent` class server; examples: `batched_whisper`, `amazon_embeddings`, `doc_ocr_jobs`.
- **Cold starts**: container boot is ~1s; model loading dominates. Mitigate with pre-downloaded weights on a Volume, `min_containers=1`, a longer `scaledown_window`, or memory snapshots (`enable_memory_snapshot=True`).

## Modal vs OpenWeights

Both run GPU workloads; they solve different problems. OpenWeights (see the `openweights` skill) is an opinionated job queue for LLM fine-tuning/inference on RunPod; Modal is general-purpose serverless compute.

**Prefer OpenWeights when:**
- Fine-tuning open models with standard recipes — `ow.fine_tuning.create(...)` gives Unsloth SFT/DPO/ORPO/weighted-SFT, logprob tracking, checkpointing, and HF push with zero infra code, plus the org's training defaults.
- Batch inference over open/fine-tuned models or multi-LoRA evals — `ow.inference.create` / `ow.api.deploy` / `ow.api.multi_deploy` handle vLLM setup for you.
- Cost matters more than latency: RunPod GPU-hours are roughly 40–50% cheaper (A100 $1.39 vs $2.50; H200 $3.59 vs $4.54; L40(S) $0.99 vs $1.95), and jobs are content-hash deduplicated so identical reruns are free.
- The workload is "submit job, wait, collect artifacts" — queueing, restarts, and artifact storage are built in.

**Prefer Modal when:**
- The code is custom and you're iterating: `modal run` gives a seconds-long edit-run loop in a real GPU container, vs. OpenWeights custom jobs (docker image + files + entrypoint) which are better suited for validated pipelines than exploration.
- Massive fan-out: thousands of short parallel tasks (`.map()` to ~1,000 concurrent containers) — data processing, sweeps, per-sample GPU work, judge/eval harnesses. OpenWeights has no equivalent.
- You need serving infrastructure: autoscaling scale-to-zero endpoints, public demo URLs, cron schedules, or sandboxes for untrusted code.
- The job isn't an LLM training/inference job at all: CPU pipelines, rendering, scientific compute, RL environments, non-Unsloth training loops (e.g. GRPO via verl/trl using Modal's official examples).
- Latency matters: Modal containers start in ~1s; OpenWeights jobs wait for a RunPod worker to spin up (minutes, sometimes longer when the GPU tier is scarce).

**Rule of thumb:** standard LLM fine-tune/inference job → OpenWeights (cheaper, zero boilerplate); anything interactive, custom, parallel-fan-out, or user-facing → Modal. For big sweeps of *standard* jobs, OpenWeights' queue + dedup usually beats orchestrating the same thing on Modal by hand.
