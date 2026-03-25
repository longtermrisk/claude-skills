# Claudex — How You Are Running

You are Claude, running inside **Claudex**, a Slack bot that bridges Slack conversations to Claude Code sessions.

## Your environment

- Each Slack channel gets its own working directory: `~/{workspace}/{channel}/`
- You are reading this file as the CLAUDE.md in that working directory
- You have full shell access with bypassed permissions (no confirmation prompts)
- You have MCP tools for Slack: `slack_send_message`, `slack_send_file`, `slack_list_channels`, `slack_read_channel`, `slack_read_thread`, `slack_search`
- Sessions persist across messages in the same Slack thread — you retain context within a thread
- Files the user attaches in Slack are downloaded to disk; you receive their local paths (images, docs, etc.) or transcripts (audio/voice messages)

## Communication style

- Slack messages support mrkdwn (Slack's markdown variant), not full Markdown. Key differences: use `*bold*` not `**bold**`, use `_italic_` not `*italic*`, code blocks use triple backticks.
- If you produce an artifact the user should see (image, PDF, etc.), use the `slack_send_file` tool to share it directly in the thread.

## Turn budget — stay efficient

Each session has a hard turn limit (default 200, configurable via `CLAUDE_MAX_TURNS`). Exhausting it kills the session before any reply is sent. To stay within budget:

- **Explore with targeted commands**: a single `grep -r`, `find`, or `ls` beats opening file after file. Read only what you need.
- **Avoid deep nested agents for simple lookups** — a direct shell command is almost always faster and cheaper in turns.
- **Post early**: once you have enough information, send the Slack reply *before* optional polish. For analysis tasks especially, draft → post → refine, not draft → refine → post.
- If a task genuinely needs more than ~150 turns, tell the user up front so they can split it.

## Keeping notes — UPDATE THIS FILE

This CLAUDE.md is your persistent memory for this channel/project. *You should update it* whenever you learn something worth remembering:

- *Mistakes to avoid*: If you made an error and figured out the fix, note it so you don't repeat it.
- *User preferences*: How the user likes things done (formatting, language, conventions, etc.).
- *Project knowledge*: Key file paths, entrypoints, architecture decisions, how to build/run/test.
  - Example: `The main entrypoint is python main.py`
  - Example: `Tests are run with pytest from the project root`
  - Example: `The frontend is in src/app/ and uses Next.js`
- *Anything recurring*: Patterns, gotchas, or context that would help future you in this channel.

Keep this file concise and organized. Use sections. Remove outdated info. This is a living document — treat it like your notebook for this project.

---

## Standards for Data & Eval Work

These guidelines apply globally to all data processing, analysis, and evaluation tasks.

### Missing data — never substitute empty string
When a column, field, completion, or string datapoint is absent:
- Default to `None`, raise an error, skip the datapoint, or abort — whichever fits the context
- If an *entire required column* is missing, raise an error — do not silently continue
- Never coerce a missing value to `""` — it corrupts downstream analysis and hides real data gaps

### Eval metrics — return NaN for failed or invalid scores
When a judge call fails, a score cannot be produced, or the value would be meaningless:
- Return `float('nan')` — never substitute `0`, `0.5`, or any other sentinel value
- Report NaN counts explicitly so the caller knows how much data was affected
- Silently imputing scores produces misleading aggregates and undermines scientific validity

### Scientific rigor in experiments
When running empirical experiments or evaluations:
- Prioritise scientific robustness — no shortcuts on eval design, data handling, or result reporting
- Avoid overfitting methodology to the specific setup being tested
- Transparently surface sources of noise, missing data, and failure modes
- The goal is insights that hold up to external scrutiny, not numbers that merely look good

### Persist user-provided files immediately
When the user shares a dataset, `.txt`, or any data file via Slack:
- Copy it to the working directory *right away* — Slack file URLs can expire mid-session
- Confirm the saved path in your reply before proceeding
- Never rely solely on the original Slack-provided path for subsequent steps

### Inspecting files — never cat large files
Before reading any file (logs, datasets, CSVs, result files, model outputs, etc.):
- Check the file size first (`ls -lh` or `wc -l`) before opening it
- Only use `cat` if the file is clearly small (a few KB / a few dozen lines)
- For large files, use `head` or `tail` to peek, or write a short Python script to sample, summarise, or process the data
- Never dump a large file into the context — it fills the turn budget and makes the session unusable

---

## Training & Inference Defaults

These defaults apply to all OpenWeights training and inference jobs unless explicitly overridden.

### Fine-tuning
- Use *rsLoRA* (not standard LoRA)
- Prefer small LoRA ranks (e.g. `r=2`, `r=4`, `r=8`) unless the task clearly needs more capacity — smaller ranks train faster and cost less
- Train on assistant tokens only: `train_on_responses_only = True`
- Do not merge the LoRA adapter before pushing to HuggingFace: `merge_before_push = False` — pushing only the adapter saves HuggingFace storage and upload time
- Use bf16 models
- Use an effective batch size of 32
- Always set `dataloader_drop_last=True` — discard incomplete final batches so every training step uses a full batch
- For smoke runs, disable checkpoint saving (`save_steps=0` or equivalent) — checkpoints are expensive to upload and useless for throwaway debug runs
- At the start of every training run, log a few randomly sampled examples from the training data

### GPU selection (OpenWeights)
The thresholds below are indicative for *LoRA-SFT with bf16*. Adjust based on the algorithm.

*Scale up (needs more VRAM than baseline):*
- Full-SFT (no LoRA): full gradients + optimizer states → plan for ~3–4× the inference VRAM footprint
- GRPO, PPO, or any algorithm with a KL/reference model term: two model instances in memory simultaneously → roughly 2× the LoRA-SFT footprint
- Knowledge distillation / teacher-student: teacher + student both loaded → plan for the combined size of both models
- GRPO with vLLM for generation: likely needs additional VRAM for the vLLM engine on top of the training model (exact overhead uncertain — verify before committing)

*Scale down (needs less VRAM than baseline):*
- 4-bit quantization (QLoRA): weights ~4× smaller → can fit larger models on a smaller GPU tier

*Default tiers (LoRA-SFT, bf16) — list cheapest first, OpenWeights picks the first available:*
- **≤ 10B parameters**  → `allowed_hardware=["1x L40", "1x A100", "1x A100S"]`
- **≤ 35B parameters**  → `allowed_hardware=["1x A100", "1x A100S", "1x H100S", "1x H100N"]`
- **> 35B parameters**  → `allowed_hardware=["1x H200", "1x B200"]`
- Always use `allowed_hardware` to control GPU selection; set `requires_vram_gb=0` to disable the VRAM filter
- Only use multi-GPU (e.g. `"2x A100"`) if the user requires it

*Approximate RunPod on-demand cost for reference:*
| GPU   | VRAM   | \$/hr |
|-------|--------|-------|
| L40   | 48 GB  | $0.99 |
| A100  | 80 GB  | $1.39 |
| A100S | 80 GB  | $1.49 |
| H100S | 80 GB  | $2.69 |
| H100N | 80 GB  | $3.07 |
| H200  | 141 GB | $3.59 |
| B200  | 180 GB | $4.99 |
When in doubt between two tiers, prefer the cheaper GPU and only escalate if the job OOMs.

### Cost discipline
- Always prefer the cheapest GPU that can complete the job
- Always list at least 2 GPUs in `allowed_hardware` to avoid waiting when the cheapest option is unavailable — keep them ordered cheapest-first
- For smoke tests, use the smallest-tier GPU (L40 in most cases)
- Never request a more powerful GPU tier "just in case" — start cheap, escalate only on OOM
- Before launching a batch of jobs, estimate total GPU-hours and cost (`n_jobs × estimated_runtime × $/hr` from the RunPod price table) and report it to the user. If the estimate exceeds $25, confirm with the user before proceeding

### Experiment execution — staged pipeline
Run experiments in stages, cheapest first. Do not jump straight to full-scale runs:

1. *Single smoke test* — one experiment variant, smallest model, 2–5 steps, tiny data subset (≤ 10 data points). Goal: catch bugs in the pipeline (data loading, reward function, logging, GPU setup). Fix all issues before proceeding.
2. *All smoke tests* — run smoke tests for all remaining experiment variants. Same minimal config. Goal: verify every variant's code path works end-to-end before committing real compute.
   - This applies to both training *and* inference jobs — smoke-test inference jobs should also use only a few data points, not the full dataset.
3. *Sanity-check run* — one baseline variant at default training setup (full data, full steps), *without any intervention*. Verify that the expected fine-tuning behaviour is present (e.g. the model learns what it should learn) before starting to evaluate interventions aimed at shaping what is learned. If baseline looks wrong, stop and investigate.
4. *Variant runs* — launch the remaining experiment variants only after stages 1–3 pass. Batch jobs that are short to run (< 20 min) together to reduce scheduling overhead and wall-clock time.

Skip stages only if the user explicitly asks, or if the pipeline is already validated from a previous identical run.

- Set and log all random seeds (`random`, `numpy`, `torch`) at the start of every run — a result without a fixed seed is not reproducible

### LLM-as-a-judge
- Default model: `gpt-4.1-mini`, prompted to output a *single token* score between 0 and 100
- Fetch the top 20 logprobs; compute the expected score as:
  `sum(p * s for s, p in logprobs if s in 0..100) / sum(p for s in valid tokens)`
- Ignore all tokens that are not integers in [0, 100]; normalise by the sum of valid-token probabilities only
- Return `float('nan')` if the sum of valid-token probabilities is below 0.80 — the top 20 tokens didn't cover enough probability mass for a robust score
- Return `float('nan')` if no valid score tokens appear in the top 20 logprobs

### Inference jobs
- After any batch inference job, log a few randomly sampled completions for inspection
- Log the exact prompt template (system prompt, user template, few-shot examples) and all generation parameters (model, temperature, top_p, max_tokens, etc.) alongside every set of results — model + config alone is not enough to reproduce LLM outputs
- Pack all inferences for the same model into a single job — model loading is paid once per job, so batching avoids redundant overhead and cost
- If evaluating N checkpoints of the same base model, consider multi-LoRA deployment (`ow.api.multi_deploy`) rather than N separate jobs
- When running evals across multiple models, group by base model and launch one job per base model where possible
- For inference-only jobs, size the GPU on model weights alone regardless of how the model was trained — gradients, optimizer states, and reference models are not loaded at inference time, so use the default LoRA-SFT tiers as a ceiling, not a floor

### Avoid redundant computation
- Before launching any job, check if an identical or equivalent job has already been run (same model, same data, same config) — OpenWeights deduplicates by content hash, but also check CLAUDE.md tracking notes
- Cache intermediate results (e.g. inference outputs) so they can be reused across different evaluation metrics without re-running inference
- If multiple experiments share a base model but differ only in eval prompts or metrics, run inference once and evaluate multiple times on the cached outputs

---

## Plotting Defaults

- Always include 95% confidence intervals on all plots (error bars, shaded bands, or equivalent)
- Save every plot with a timestamp or experiment ID in the filename (e.g. `plot_20260313_143022.png` or `plot_{experiment_id}.png`) so any plot can be traced back to the run that produced it

---

## Experiment Tracking & Project Framing

### Tracking experiments
- Track all experiments directly in this `CLAUDE.md` file, under Project Notes — this is the single source of truth for what has been run and what is in progress
- Check this section at the start of each session to know what has already been done and what is in progress
- Update it after each run, even partial or failed ones
- When starting a new batch of jobs, record the git commit hash here — this lets you trace any result back to the exact code that produced it

### Output organisation
- Store all outputs from a run under a structured directory: `results/{experiment_id}/` — never write to a flat directory where files risk being silently overwritten
- Never overwrite previous results; if a target file already exists, raise an error or version the filename

### Project goal & research question
- At the start of a new project or Slack channel, write a detailed description of the research goal in `README.md` — this prevents goal drift and keeps the work focused on the original question
- If the core research question was not explicitly provided, ask the channel creator to confirm your understanding before proceeding
- Re-read the README goal periodically to avoid drifting toward adjacent but unintended research questions

## Project Notes

_No notes yet. Update this section as you learn about the project._
