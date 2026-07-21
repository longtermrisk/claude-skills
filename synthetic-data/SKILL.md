---
name: synthetic-data
description: Use this skill when generating synthetic datasets using LLMs. Covers implementation with the org's LiteLLM proxy, prompting strategies for diversity, model selection, parallelization patterns, and dataset testing workflows. Ideal for creating training data, test datasets, or augmenting existing data.
---

# Synthetic Data Generation

## Overview

Generate high-quality synthetic datasets using LLMs through the org's LiteLLM proxy (see the `litellm` skill for full access details). This skill covers the complete workflow from implementation to validation, with emphasis on diversity, efficiency, and best practices.

## When to Use This Skill

Use this skill when:
- Generating synthetic training data for ML models
- Creating test datasets for software validation
- Augmenting existing datasets with additional examples
- Building evaluation benchmarks
- Producing diverse examples for research or analysis

## Prerequisites

The standard `openai` SDK pointed at the LiteLLM proxy:

```bash
pip install openai
```

```python
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ['LITELLM_API_KEY'],  # already set in the environment
    base_url="https://litellm.nielsrolf.com",
    # Cloudflare blocks the SDK's default User-Agent — see the litellm skill
    default_headers={"User-Agent": "litellm-client/1.0"},
    max_retries=5,  # built-in exponential backoff for batch jobs
)
```

## Quick Start

Always use caching for synthetic data generation. The proxy caches responses keyed on the full request (model, messages, seed, temperature, ...), so interrupted processes can resume without regenerating completed samples — just re-run the script:

```python
import asyncio

async def generate_sample(seed: int):
    response = await client.chat.completions.create(
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": "Generate a creative story about a robot learning to paint."}],
        temperature=0.8,                             # higher temperature for creativity
        seed=seed,                                   # distinct seed per sample; part of the cache key
        extra_body={"cache": {"use-cache": True}},   # opt in to proxy caching
    )
    return response.choices[0].message.content

result = asyncio.run(generate_sample(seed=12345))
print(result)
```

## Model Selection

Model names are `provider/model` (see the `litellm` skill). Choose based on task requirements:

**Most capable:**
- `anthropic/claude-sonnet-4-5` - Best for complex reasoning tasks
- `openai/gpt-5` - Balanced capability and speed
- `gemini/gemini-2.5-pro` - Google's flagship model

**Fast and cost-effective:**
- `openai/gpt-5-mini` - Recommended for most synthetic data generation
- `gemini/gemini-2.5-flash` - Fast alternative for high-volume generation

**Local (free, but slow and one model loaded at a time):**
- `local/<model-name>` - llama-server on the host

### Model Selection Guidelines

- **Most synthetic data generation**: Use `openai/gpt-5-mini` or `anthropic/claude-sonnet-4-5`
- **Complex reasoning tasks**: Use a flagship model with reasoning enabled (`reasoning_effort`), and set `max_tokens` generously — reasoning models spend tokens thinking before any visible text
- **High-volume generation**: Use `openai/gpt-5-mini` or `gemini/gemini-2.5-flash`
- **Multiple models**: Combine different models for additional diversity

**Important**: Claude models may refuse some AI safety-related tasks that appear dual-use (e.g., jailbreak datasets).

## Structured Output Generation

Use Pydantic models for type-safe structured data via the SDK's `parse` helper:

```python
from pydantic import BaseModel, Field
from typing import List

class CalendarEvent(BaseModel):
    name: str = Field(description="Event name")
    date: str = Field(description="Event date in YYYY-MM-DD format")
    participants: List[str] = Field(description="List of participant names")
    description: str = Field(description="Event description")

async def generate_structured_sample(seed: int):
    response = await client.beta.chat.completions.parse(
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": "Generate a fictional team meeting event."}],
        response_format=CalendarEvent,
        seed=seed,
        extra_body={"cache": {"use-cache": True}},
    )
    return response.choices[0].message.parsed  # validated CalendarEvent instance

event = asyncio.run(generate_structured_sample(seed=42))
print(f"Event: {event.name} on {event.date}")
```

The `response_format` schema is part of the cache key, so structured requests cache just like plain ones.

## Large-Scale Dataset Generation

For generating large datasets, use parallelization with appropriate concurrency limits. The provided script in `scripts/generate_dataset.py` implements this pattern.

### Parallelization Guidelines

- **OpenAI/Google models**: Use 50-100 concurrent requests
- **Anthropic models**: Use up to 20 concurrent requests
- **Always test prompts on 10-20 examples first**
- **Write outputs incrementally** for progress monitoring
- **Use terminal tool (not Jupyter)** for long-running processes

### Example: Parallel Generation

```python
import asyncio
import json

async def generate_single_sample(prompt: str, seed: int) -> str:
    response = await client.chat.completions.create(
        model="openai/gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        seed=seed,
        extra_body={"cache": {"use-cache": True}},
    )
    return response.choices[0].message.content

async def generate_dataset(prompts, output_file="dataset.jsonl"):
    """Generate a large dataset with parallelization"""
    semaphore = asyncio.Semaphore(50)  # Adjust based on provider

    async def generate_with_semaphore(prompt, seed):
        async with semaphore:
            return await generate_single_sample(prompt, seed)

    # Generate tasks
    tasks = [
        generate_with_semaphore(prompt, seed)
        for seed, prompt in enumerate(prompts)
    ]

    # Process in batches and write incrementally
    batch_size = 100
    with open(output_file, 'w') as f:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            results = await asyncio.gather(*batch)

            # Write batch results
            for j, result in enumerate(results):
                data = {
                    "id": i + j,
                    "prompt": prompts[i + j],
                    "response": result
                }
                f.write(json.dumps(data) + '\n')
                f.flush()  # Ensure data is written immediately

            print(f"Completed batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")

# Usage
prompts = ["Generate a story about...", "Create a dialogue between...", ...]
asyncio.run(generate_dataset(prompts))
```

Because every request is cached with its seed, re-running this script after a crash skips all completed samples — cached requests return instantly and only the remaining ones hit the upstream provider.

## Ensuring Diversity

Diversity is critical for most datasets. LLMs may produce only 10-50 diverse outputs for identical prompts, even with different seeds and high temperature. Use these strategies:

### 1. Combinatory Prompts

Combine multiple dimensions to create diverse prompt variations:

```python
topics = ["technology", "healthcare", "education", "finance"]
styles = ["formal", "casual", "technical", "creative"]
perspectives = ["optimistic", "critical", "neutral", "innovative"]

prompts = []
seed = 0
for topic in topics:
    for style in styles:
        for perspective in perspectives:
            prompt = f"Write a {style} {perspective} analysis about {topic}"
            prompts.append((prompt, seed))
            seed += 1

# This produces 4×4×4 = 64 diverse prompts
```

### 2. Data Augmentation

Add variation through augmentation layers:

```python
base_prompts = ["Write a story about friendship", "Describe a future city"]
augmentations = {
    "greetings": ["Hello!", "Hi there!", "Greetings!"],
    "contexts": ["In a fantasy world,", "In the year 2050,", "During a storm,"],
    "styles": ["Write creatively:", "Be descriptive:", "Keep it concise:"]
}

augmented_prompts = []
for base in base_prompts:
    for greeting in augmentations["greetings"]:
        for context in augmentations["contexts"]:
            for style in augmentations["styles"]:
                augmented = f"{greeting} {style} {context} {base}"
                augmented_prompts.append(augmented)
```

### 3. Using Existing Datasets as Seeds

Leverage diverse existing data as generation seeds:

```python
import pandas as pd

# Load existing diverse dataset
seed_data = pd.read_csv("diverse_topics.csv")

# Use each row as input for generation
async def generate_from_seeds(seed_df):
    results = []
    for idx, row in seed_df.iterrows():
        prompt = f"Based on this topic: {row['topic']}, generate a {row['format']} about {row['subject']}"
        result = await generate_single_sample(prompt, idx)
        results.append(result)
    return results
```

### 4. Multiple Models

Use different models as another source of diversity. Combine outputs from GPT-5, Claude, and Gemini for varied perspectives and styles.

## Tool-Based Generation

Use tools for complex generation workflows requiring structured interaction (standard OpenAI tools format works across providers through the proxy):

```python
data_gen_tool = {
    "type": "function",
    "function": {
        "name": "generate_sample",
        "description": "Generate a data sample with specific parameters",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Data category"},
                "format": {"type": "string", "description": "Output format"},
                "count": {"type": "integer", "description": "Number of samples"}
            },
            "required": ["category", "format"]
        }
    }
}

async def generate_with_tools(seed: int):
    response = await client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Generate customer feedback data in JSON format"}],
        tools=[data_gen_tool],
        seed=seed,
        extra_body={"cache": {"use-cache": True}},
    )

    for tool_call in response.choices[0].message.tool_calls or []:
        print(f"Tool: {tool_call.function.name}, Args: {tool_call.function.arguments}")

    return response
```

Tool definitions are part of the cache key, so changing a tool schema correctly invalidates the cache.

## Dataset Exploration and Validation

After generation, explore and validate datasets using the provided exploration script in `scripts/explore_dataset.py`. The script provides:

- Basic statistics and shape information
- Sample examples for manual review
- Categorical distribution analysis
- Duplicate detection
- Text length analysis for content fields

### Quick Exploration Example

```python
import pandas as pd

# Load dataset
df = pd.read_json("dataset.jsonl", lines=True)

# Basic exploration
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nRandom samples:")
print(df.sample(5))

# Check for duplicates
print(f"\nDuplicates: {df.duplicated().sum()}")

# Analyze text lengths
text_cols = [col for col in df.columns if 'text' in col.lower()]
for col in text_cols:
    df[f'{col}_length'] = df[col].str.len()
    print(f"\n{col} length stats:")
    print(df[f'{col}_length'].describe())
```

## Best Practices Workflow

1. **Start small**: Test prompts on 10-20 examples before large runs
2. **Use caching**: Always pass a distinct `seed` per sample and `extra_body={"cache": {"use-cache": True}}` for resumable generation
3. **Write incrementally**: Save results to files in batches for progress monitoring
4. **Monitor quality**: Spot-check outputs during generation
5. **Test diversity**: Analyze sample batches for variety before scaling up
6. **Choose appropriate concurrency**: Follow provider-specific limits
7. **Use terminal for long runs**: Avoid Jupyter for processes taking >10 minutes
8. **Validate outputs**: Run exploration scripts on generated data
9. **Iterate on prompts**: Refine based on initial results before full generation
10. **Document your pipeline**: Save prompt templates and generation configs

## Common Patterns

### Pattern 1: Simple Text Generation at Scale

```python
# Define diverse prompts
prompts = [f"Write a story about {topic}" for topic in topics]

# Generate with caching and parallelization
asyncio.run(generate_dataset(prompts, output_file="stories.jsonl"))
```

### Pattern 2: Structured Data with Validation

```python
# Define Pydantic schema
class DataSchema(BaseModel):
    field1: str
    field2: int

# Generate structured samples
async def generate_structured_batch(count):
    results = []
    for i in range(count):
        response = await client.beta.chat.completions.parse(
            model="openai/gpt-5-mini",
            messages=[...],
            response_format=DataSchema,
            seed=i,
            extra_body={"cache": {"use-cache": True}},
        )
        results.append(response.choices[0].message.parsed)
    return results
```

### Pattern 3: Multi-Turn Conversation Data

```python
# Generate conversational data
async def generate_conversation(seed):
    messages = [{"role": "user", "content": "Start a conversation about AI ethics"}]

    # Generate multiple turns
    for turn in range(3):
        response = await client.chat.completions.create(
            model="anthropic/claude-sonnet-4-5",
            messages=messages,
            seed=seed * 100 + turn,
            extra_body={"cache": {"use-cache": True}},
        )
        messages.append({"role": "assistant", "content": response.choices[0].message.content})

        # Add next user message based on context
        messages.append({"role": "user", "content": "Continue the discussion"})

    return messages
```

## Resources

This skill includes helper scripts:

- `scripts/generate_dataset.py` - Production-ready parallel generation with progress tracking
- `scripts/explore_dataset.py` - Comprehensive dataset exploration and validation

## Summary Checklist

When generating synthetic data:

- ✓ Route all requests through the LiteLLM proxy with a distinct `seed` per sample and `extra_body={"cache": {"use-cache": True}}`
- ✓ Choose appropriate model for task (usually `openai/gpt-5-mini` or `anthropic/claude-sonnet-4-5`)
- ✓ Test prompts on 10-20 examples before large runs
- ✓ Implement parallelization with provider-appropriate concurrency limits
- ✓ Ensure diversity through combinatory prompts, augmentation, or seed datasets
- ✓ Write outputs incrementally to files
- ✓ Use terminal tool for long-running processes
- ✓ Explore and validate datasets after generation
- ✓ Use Pydantic models for structured outputs when appropriate
