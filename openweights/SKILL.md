---
name: openweights
description: Use this skill when working with OpenWeights - an OpenAI-like SDK for running LLM fine-tuning, inference, evaluations, and custom GPU workloads on managed RunPod infrastructure. Trigger this skill for tasks involving model training, batch inference, API deployments, custom GPU jobs, or when the user mentions "openweights", "ow", fine-tuning models, or running GPU workloads.
---

# OpenWeights

## Overview

OpenWeights is a Python SDK that provides an OpenAI-like interface for running distributed GPU workloads on managed RunPod infrastructure. Enable users to fine-tune models (SFT, DPO, ORPO), run batch inference, deploy vLLM APIs, execute Inspect AI evaluations, and create custom GPU jobs with a simple Python API.

**Core Philosophy:** Jobs are defined by (1) Docker image, (2) mounted files, and (3) entrypoint command. Built-in jobs provide convenient templates, while custom jobs enable complete flexibility.

## Setup and Validation

### First-Time Setup

When a user is setting up OpenWeights for the first time:

1. **Check if already installed:**
   ```bash
   python scripts/validate_setup.py
   ```

2. **If not installed, run quickstart:**
   ```bash
   bash scripts/quickstart.sh
   ```

3. **Manual setup steps:**
   ```bash
   # Install OpenWeights
   pip install openweights

   # Create API key
   ow signup user@email.com

   # Set environment variable
   export OPENWEIGHTS_API_KEY=your_key_here

   # For persistence, add to shell profile
   echo 'export OPENWEIGHTS_API_KEY=your_key_here' >> ~/.bashrc
   ```

4. **For self-managed clusters (optional):**
   - Set `RUNPOD_API_KEY` for infrastructure provisioning
   - Set `HF_TOKEN`, `HF_USER`, `HF_ORG` for model publishing
   - See `references/env.worker.example` for full list

### Validation

Always validate setup before submitting jobs:
```python
from openweights import OpenWeights

ow = OpenWeights()  # Will raise error if OPENWEIGHTS_API_KEY not set
```

Or use the validation script: `python scripts/validate_setup.py`

## Core Workflows

### 1. Fine-Tuning

OpenWeights supports multiple fine-tuning approaches via Unsloth backend.

#### Standard SFT (Supervised Fine-Tuning)

**When to use:** Training models to follow instructions or replicate specific response patterns.

**Basic workflow:**
```python
from openweights import OpenWeights

ow = OpenWeights()

# Upload training data (conversations format)
training_file = ow.files.upload("data/train.jsonl", purpose="conversations")["id"]

# Create fine-tuning job
job = ow.fine_tuning.create(
    model="unsloth/Qwen3-4B",
    training_file=training_file,
    loss="sft",
    epochs=1,
    learning_rate=1e-4,
    r=32,  # LoRA rank
)

print(f"Job ID: {job.id}")
print(f"Status: {job.status}")
print(f"Model will be pushed to: {job.params['validated_params']['finetuned_model_id']}")
```

**For large models (QLoRA):**
```python
job = ow.fine_tuning.create(
    model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    training_file=training_file,
    test_file=test_file,  # Optional validation set
    load_in_4bit=True,
    max_seq_length=2047,
    loss="sft",
    epochs=1,
    learning_rate=1e-4,
    r=32,
    save_steps=10,  # Checkpoint frequency
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    allowed_hardware=["1x H200"],  # Specify GPU requirements
    merge_before_push=False,  # Push only LoRA adapter
)
```

**See example:** `assets/examples/sft/lora_qwen3_4b.py` or `assets/examples/sft/qlora_llama3_70b.py`

#### DPO/ORPO (Preference Learning)

**When to use:** Training models to prefer certain responses over others using preference pairs.

```python
# Upload preference dataset
training_file = ow.files.upload("data/preferences.jsonl", purpose="preference")["id"]

# DPO
job = ow.fine_tuning.create(
    model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    training_file=training_file,
    loss="dpo",  # or "orpo"
    epochs=1,
    learning_rate=5e-6,
    r=32,
)
```

**Data format:** See `assets/templates/preferences.jsonl` for structure.

#### Weighted SFT (Token-Level Loss Weighting)

**When to use:** Fine-grained control over which tokens to emphasize, minimize, or ignore during training.

```python
training_file = ow.files.upload("data/weighted_data.jsonl", purpose="conversations")["id"]

job = ow.weighted_sft.create(
    model="unsloth/Qwen3-4B",
    training_file=training_file,
    loss="sft",
    epochs=20,
    learning_rate=1e-4,
    r=32,
)
```

**Data format:** Use block-formatted conversations with `weight` fields. See `references/02_workflows.md` → "Conversations, block-formatted".

#### Log Probability Tracking

**When to use:** Monitor model's confidence on specific examples during training.

```python
logp_file = ow.files.upload("data/logp_tracking.jsonl", purpose="conversations")["id"]

job = ow.fine_tuning.create(
    model="unsloth/Qwen3-4B",
    training_file=training_file,
    loss="sft",
    epochs=4,
    eval_every_n_steps=1,
    logp_callback_datasets={"in-distribution": logp_file},
)

# After completion, fetch events
events = ow.events.list(run_id=job.runs[-1].id)
# Events contain log probabilities per tag
```

**See example:** `assets/examples/sft/logprob_tracking.py`

### 2. Inference

#### Batch Inference

**When to use:** Running inference on many prompts efficiently.

```python
from openweights import OpenWeights

ow = OpenWeights()

# Upload prompts (conversations format)
input_file = ow.files.upload("prompts.jsonl", purpose="conversations")["id"]

# Create inference job
job = ow.inference.create(
    model="unsloth/Qwen3-4B",  # Can be HF model, LoRA adapter, or checkpoint
    input_file_id=input_file,
    max_tokens=1000,
    temperature=0.8,
    max_model_len=2048,
)

# Wait for completion
import time
while job.refresh().status != "completed":
    time.sleep(5)

# Get results
outputs_str = ow.files.content(job.outputs["file"]).decode("utf-8")
import json
outputs = [json.loads(line) for line in outputs_str.split("\n") if line]

for output in outputs:
    print(output["completion"])
```

**Model formats supported:**
- HuggingFace models: `"unsloth/Qwen3-4B"`
- LoRA adapters: `"hf-org/repo-with-lora-adapter"`
- Checkpoints: `"hf-org/repo/path/to/checkpoint.ckpt"`

**See example:** `assets/examples/inference/run_inference.py`

#### vLLM API Deployment

**When to use:** Need an OpenAI-compatible API endpoint for interactive use.

**Basic deployment:**
```python
from openweights import OpenWeights

ow = OpenWeights()

model = "unsloth/Qwen3-4B"

# Deploy as context manager (auto-cleanup)
with ow.api.deploy(model):
    completion = ow.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "is 9.11 > 9.9?"}]
    )
    print(completion.choices[0].message)
# API automatically shuts down when exiting context
```

**Manual control:**
```python
api = ow.api.deploy(model)
api.up()  # Start the API

# Use the API
completion = ow.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello!"}]
)

api.down()  # Shut down when done
```

**Multi-model deployment (LoRA adapters):**
```python
# Deploy multiple LoRA adapters of same base model
apis = ow.api.multi_deploy([
    "org/model-lora-v1",
    "org/model-lora-v2",
    "org/model-lora-v3",
])
# All adapters share one API, reducing costs
```

**Gradio UI:**
```bash
python assets/examples/api-deployment/gradio_ui.py unsloth/Qwen3-4B
```

**See examples:** `assets/examples/api-deployment/`

### 3. Evaluations (Inspect AI)

**When to use:** Running standardized benchmarks on models.

```python
from openweights import OpenWeights

ow = OpenWeights()

job = ow.inspect_ai.create(
    model='meta-llama/Llama-3.3-70B-Instruct',
    eval_name='inspect_evals/gpqa_diamond',
    options='--top-p 0.9',  # Any options that `inspect eval` accepts
)

# Wait for completion
import time
while job.refresh().status in ["pending", "in_progress"]:
    time.sleep(10)

if job.status == 'completed':
    job.download('output')  # Download results
```

**See example:** `assets/examples/inspect_eval.py`

### 4. Custom Jobs

**When to use:** Need to run arbitrary Python scripts on GPU infrastructure (custom training loops, data processing, model merging, etc.).

#### Creating a Custom Job

1. **Define the job class:**

```python
from openweights import OpenWeights, register, Jobs
from pydantic import BaseModel, Field
import json
import os

@register('my_custom_job')
class MyCustomJob(Jobs):
    mount = {
        'local/path/to/script.py': 'script.py',
        'local/path/to/dir/': 'dirname/'
    }
    params = MyParams  # Pydantic model for validation
    requires_vram_gb = 24
    base_image = 'nielsrolf/ow-default'  # Optional

    def get_entrypoint(self, validated_params: BaseModel) -> str:
        return f'python script.py {json.dumps(validated_params.model_dump())}'
```

2. **Create the worker script** (runs on GPU):

```python
# worker_script.py
import json
import sys
from openweights import OpenWeights

# Parse parameters
params = json.loads(sys.argv[1])

# Do work
result = do_computation(params)

# Log results
ow = OpenWeights()
ow.run.log({"result": result})
ow.run.log({"metric": 0.95})

# Upload files (place in /uploads directory)
# Files in /uploads are automatically uploaded as job outputs
```

3. **Submit the job:**

```python
ow = OpenWeights()

job = ow.my_custom_job.create(
    param1="value",
    param2=42
)
```

**Template:** Use `assets/templates/custom_job_template.py` as starting point.

**Complete example:** See `assets/examples/custom_job/` for full implementation with client and worker sides.

## Job Management

### Monitoring Jobs

```python
# Check job status
job = ow.fine_tuning.get(job_id)
print(f"Status: {job.status}")

# List all jobs
jobs = ow.jobs.list()

# Stream logs
logs = ow.files.content(job.runs[-1].log_file).decode("utf-8")
print(logs)

# Get events (metrics, checkpoints)
events = ow.events.list(run_id=job.runs[-1].id)
for event in events:
    print(event["data"])
```

### Job Lifecycle

Jobs progress through states: `pending` → `in_progress` → `completed`/`failed`/`canceled`

**Key concepts:**
- **Jobs** are reusable templates (identified by content hash)
- **Runs** are individual executions of a job
- **Events** are structured logs/outputs during a run

### Waiting for Completion

```python
import time

while job.refresh().status in ["pending", "in_progress"]:
    print(f"Status: {job.status}")
    time.sleep(5)

if job.status == "failed":
    logs = ow.files.content(job.runs[-1].log_file).decode("utf-8")
    print("Error logs:", logs)
elif job.status == "completed":
    print("Success!")
    print(f"Outputs: {job.outputs}")
```

### Downloading Artifacts

```python
# Download all job outputs
job.download("local/output/dir", only_last_run=False)

# Download specific file
file_content = ow.files.content(file_id).decode("utf-8")
```

## Data Formats

### Conversations Format

For SFT training and inference prompts:

```json
{
    "messages": [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant response"}
    ]
}
```

**Template:** `assets/templates/conversations.jsonl`

**Note:** Inference files ending with an assistant message will continue that message (prefix generation).

### Block-Formatted Conversations

For weighted SFT and log-probability tracking:

```json
{
    "messages": [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Minimize log-likelihood on this",
                    "weight": -1,
                    "tag": "minimize"
                },
                {
                    "type": "text",
                    "text": "Maximize log-likelihood on this",
                    "weight": 1,
                    "tag": "maximize"
                }
            ]
        }
    ]
}
```

**See:** `references/02_workflows.md` → "Conversations, block-formatted"

### Preferences Format

For DPO/ORPO training:

```json
{
    "prompt": [
        {"role": "user", "content": "Question"}
    ],
    "chosen": [
        {"role": "assistant", "content": "Preferred response"}
    ],
    "rejected": [
        {"role": "assistant", "content": "Dispreferred response"}
    ]
}
```

**Template:** `assets/templates/preferences.jsonl`

## CLI Tools

OpenWeights provides CLI commands for advanced workflows:

```bash
# Sign up and get API key
ow signup

# Manage tokens
ow token create --name "my-token"
ow token list
ow token revoke <token-id>

# List jobs
ow ls

# View logs
ow logs <job-id>

# Cancel job
ow cancel <job-id>

# Fetch file content
ow fetch <file-id>

# SSH into a GPU worker (with live file sync)
ow ssh

# Execute command on remote GPU
ow exec "python train.py"

# Cluster management (self-hosted)
ow cluster --env-file .env.worker
ow deploy --env-file .env.worker

# Manage environment variables
ow env import .env.worker
ow env list
ow env delete KEY_NAME
```

**Development workflow with live sync:**
```bash
ow ssh  # Opens SSH session with live file sync from CWD
# Edit files locally, test immediately on GPU
```

## Advanced Features

### Hardware Selection

**Automatic (based on VRAM):**
```python
job = ow.fine_tuning.create(
    model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    requires_vram_gb=80,  # Will select appropriate GPU
    # ...
)
```

**Manual specification:**
```python
job = ow.fine_tuning.create(
    model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    allowed_hardware=["1x H200", "2x A100"],  # Whitelist specific configs
    # ...
)
```

### Content-Addressable IDs

**Key insight:** Job and file IDs are content hashes.

- Submitting identical job returns existing job
- Resubmitting failed job resets it to `pending`
- Uploading identical file returns existing file ID

**Implication:** No duplicate work, automatic deduplication.

### Checkpoint Management

```python
job = ow.fine_tuning.create(
    model="unsloth/Qwen3-4B",
    training_file=training_file,
    save_steps=10,  # Save checkpoint every 10 steps
    # ...
)

# After training, checkpoints are uploaded as events
events = ow.events.list(run_id=job.runs[-1].id)
checkpoint_events = [e for e in events if "checkpoint" in e.get("data", {})]

# Use checkpoint for inference
checkpoint_path = f"{job.params['validated_params']['finetuned_model_id']}/checkpoint-100"
inference_job = ow.inference.create(model=checkpoint_path, input_file_id=prompts)
```

### Cluster Management (Self-Hosted)

For users managing their own infrastructure:

**Local cluster manager:**
```bash
ow cluster --env-file .env.worker
```

**Deploy to RunPod CPU instance:**
```bash
ow deploy --env-file .env.worker
```

**Managed (trust openweights with keys):**
```bash
ow env import .env.worker
ow manage start
```

**Required environment variables:** See `references/env.worker.example`

## Troubleshooting

### Common Issues

**"Job stays in pending":**
- Check cluster manager is running
- Verify RUNPOD_API_KEY is set (for self-hosted)
- Check hardware requirements aren't too restrictive

**"Job failed immediately":**
- View logs: `ow logs <job-id>` or `ow.files.content(job.runs[-1].log_file)`
- Common causes: data format errors, model not found, insufficient VRAM

**"File validation error":**
- Ensure JSONL format (one JSON object per line)
- Validate against templates in `assets/templates/`
- Check purpose matches data type ("conversations" vs "preference")

**"Import error in custom job":**
- Verify all dependencies in base image
- Consider building custom Docker image with required packages

### Getting Help

- **Documentation:** Read `references/01_overview.md`, `references/02_workflows.md`, `references/03_architecture.md`
- **Examples:** Browse `assets/examples/` for complete working examples
- **CLI help:** `ow <command> --help`

## Workflow Decision Trees

### "Should I use a built-in job or custom job?"

**Use built-in job when:**
- Standard fine-tuning (SFT, DPO, ORPO, weighted SFT)
- Batch inference with vLLM
- API deployment
- Inspect AI evaluations

**Use custom job when:**
- Custom training loop not supported by Unsloth
- Data preprocessing on GPU
- Model merging or conversion
- Arbitrary Python script needs GPU

**Modifying built-in jobs:**
Built-in jobs are just Python code in `openweights/jobs/`. To modify:
1. Copy relevant job class code
2. Create custom job with modifications
3. Register with `@register('my_modified_job')`

### "How do I structure my training pipeline?"

**Linear pipeline:**
```python
# 1. Upload data
train_file = ow.files.upload("train.jsonl", purpose="conversations")["id"]

# 2. Fine-tune
job = ow.fine_tuning.create(model="unsloth/Qwen3-4B", training_file=train_file)
wait_for_completion(job)

# 3. Evaluate
eval_job = ow.inspect_ai.create(
    model=job.params['validated_params']['finetuned_model_id'],
    eval_name='inspect_evals/gpqa_diamond'
)
wait_for_completion(eval_job)

# 4. Deploy
with ow.api.deploy(job.params['validated_params']['finetuned_model_id']):
    # Use the model
    pass
```

**Iterative tuning:**
```python
# Try different hyperparameters
for lr in [1e-4, 5e-5, 1e-5]:
    job = ow.fine_tuning.create(
        model="unsloth/Qwen3-4B",
        training_file=train_file,
        learning_rate=lr,
    )
    # Jobs run in parallel (if workers available)
```

## References

This skill includes comprehensive reference documentation:

- **references/01_overview.md**: Complete feature overview and quickstart
- **references/02_workflows.md**: Detailed examples and data formats
- **references/03_architecture.md**: System architecture and advanced concepts
- **references/env.worker.example**: Required environment variables

Load these when deeper understanding is needed. For most workflows, this SKILL.md provides sufficient guidance.

## Resources

- **scripts/validate_setup.py**: Check OpenWeights installation and configuration
- **scripts/quickstart.sh**: Automated setup script
- **assets/examples/**: Complete working examples for all features
- **assets/templates/**: Data format templates and custom job boilerplate
