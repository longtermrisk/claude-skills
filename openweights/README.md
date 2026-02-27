# OpenWeights Skill

This skill enables Claude to work with OpenWeights - an OpenAI-like SDK for running LLM fine-tuning, inference, evaluations, and custom GPU workloads on managed RunPod infrastructure.

## Features

This skill provides comprehensive guidance for:

- **Fine-tuning**: SFT, DPO, ORPO, weighted SFT, log-probability tracking
- **Inference**: Batch inference and vLLM API deployments
- **Evaluations**: Inspect AI integration
- **Custom Jobs**: Running arbitrary GPU workloads
- **Job Management**: Monitoring, logging, and artifact management
- **Setup & Validation**: Installation, configuration, and troubleshooting

## Installation

1. **Install the skill** by copying the `openweights/` directory to your `~/.claude/skills/` folder, or by unzipping the packaged `openweights.zip` file.

2. **Restart Claude Code** to load the skill.

3. **The skill will automatically trigger** when you mention openweights, fine-tuning, model training, or GPU workloads.

## Contents

### SKILL.md
The main skill file with comprehensive workflow guidance covering all OpenWeights features.

### scripts/
- `validate_setup.py` - Check OpenWeights installation and configuration
- `quickstart.sh` - Automated setup script

### references/
- `01_overview.md` - Complete feature overview from README
- `02_workflows.md` - Detailed cookbook examples and data formats
- `03_architecture.md` - System architecture documentation
- `env.worker.example` - Environment variable template

### assets/
- `examples/` - Complete working examples copied from cookbook:
  - `sft/` - Fine-tuning examples (LoRA, QLoRA, weighted SFT, etc.)
  - `inference/` - Batch inference examples
  - `api-deployment/` - vLLM API deployment examples
  - `custom_job/` - Custom job implementation examples
  - `preference_learning/` - DPO/ORPO examples
  - `inspect_eval.py` - Evaluation example
- `templates/` - Data format templates and boilerplate:
  - `conversations.jsonl` - Standard conversation format
  - `preferences.jsonl` - Preference dataset format
  - `custom_job_template.py` - Custom job template

## Regenerating the Skill

When the OpenWeights repository is updated, regenerate the skill content:

```bash
# From the skill directory
python build_skill.py /path/to/openweights/repo
```

This will:
1. Update reference documentation from the latest README and CLAUDE.md
2. Copy latest cookbook examples to assets/
3. Preserve the SKILL.md and scripts/

After regenerating, review changes and repackage:

```bash
python /path/to/skill-creator/scripts/package_skill.py /path/to/openweights/skill
```

## Build Script

The `build_skill.py` script automatically generates skill content from the openweights repository:

- **references/**: Extracts and organizes documentation
- **assets/examples/**: Copies entire cookbook directory
- **assets/templates/**: Creates data format templates
- **scripts/**: Generates validation and setup scripts

This ensures the skill stays up-to-date with the latest OpenWeights features and examples.

## Usage

Once installed, the skill will automatically activate when working with OpenWeights tasks. Claude will:

1. Guide you through setup and installation if needed
2. Validate your configuration before running jobs
3. Provide workflow-specific guidance for fine-tuning, inference, etc.
4. Reference examples and templates from assets/
5. Help troubleshoot common issues

## Examples

**Fine-tune a model:**
```python
from openweights import OpenWeights

ow = OpenWeights()
training_file = ow.files.upload("data/train.jsonl", purpose="conversations")["id"]
job = ow.fine_tuning.create(
    model="unsloth/Qwen3-4B",
    training_file=training_file,
    loss="sft",
    epochs=1,
    learning_rate=1e-4,
    r=32,
)
```

**Run batch inference:**
```python
input_file = ow.files.upload("prompts.jsonl", purpose="conversations")["id"]
job = ow.inference.create(
    model="unsloth/Qwen3-4B",
    input_file_id=input_file,
    max_tokens=1000,
)
```

**Deploy vLLM API:**
```python
with ow.api.deploy("unsloth/Qwen3-4B"):
    completion = ow.chat.completions.create(
        model="unsloth/Qwen3-4B",
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

See `assets/examples/` for complete working examples.

## Support

For issues with OpenWeights itself, see:
- GitHub: https://github.com/longtermrisk/openweights
- Documentation in `references/`

For skill-related issues, refer to the skill-creator documentation.
