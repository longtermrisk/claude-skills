---
name: synthetic-data
description: Use this skill when generating synthetic datasets using LLMs. Covers implementation with localrouter, prompting strategies for diversity, model selection, parallelization patterns, and dataset testing workflows. Ideal for creating training data, test datasets, or augmenting existing data.
---

# Synthetic Data Generation

## Overview

Generate high-quality synthetic datasets using LLMs through localrouter's unified interface. This skill covers the complete workflow from implementation to validation, with emphasis on diversity, efficiency, and best practices.

## When to Use This Skill

Use this skill when:
- Generating synthetic training data for ML models
- Creating test datasets for software validation
- Augmenting existing datasets with additional examples
- Building evaluation benchmarks
- Producing diverse examples for research or analysis

## Prerequisites

Ensure localrouter is installed (refer to the localrouter skill for details):

```bash
pip install localrouter
```

Localrouter will detect API keys to register providers that it can access. Always start by viewing available models:
```bash
python -c "from localrouter import print_available_models; print_available_models()"
```

If no models are available, you need to set at least one of the following environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
export OPENROUTER_API_KEY="your-openrouter-key"
```

## Quick Start

Always use caching and backoff for synthetic data generation. Caching ensures interrupted processes can resume without regenerating completed samples:

```python
import asyncio
from localrouter import get_response_cached_with_backoff as get_response, ChatMessage, MessageRole, TextBlock

async def generate_sample():
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Generate a creative story about a robot learning to paint.")]
        )
    ]

    response = await get_response(
        model="gpt-5-mini",
        messages=messages,
        temperature=0.8,  # Higher temperature for creativity
        cache_seed=12345  # Required for caching
    )

    return response.content[0].text

result = asyncio.run(generate_sample())
print(result)
```

## Model Selection

Choose models based on task requirements:

### Current Generation Models (Recommended)

**Most capable:**
- `claude-sonnet-4-5-20250929` - Best for complex reasoning tasks
- `gpt-5` - Balanced capability and speed
- `gemini-2.5-pro` - Google's flagship model

**Fast and cost-effective:**
- `gpt-5-mini` - Recommended for most synthetic data generation
- `gemini-2.5-flash` - Fast alternative for high-volume generation

**Specialized:**
- `o3` - For complex reasoning and mathematical tasks
- `o4-mini` - Fast reasoning for structured outputs

### Model Selection Guidelines

- **Most synthetic data generation**: Use `gpt-5-mini` or `claude-sonnet-4-5-20250929`
- **Complex reasoning tasks**: Use one of the flagship models (`gpt-5`, `claude-sonnet-4-5-20250929`, `gemini-2.5-pro`) with reasoning enables
- **High-volume generation**: Use `gpt-5-mini` or `gemini-2.5-flash`
- **Multiple models**: Combine different models for additional diversity

**Important**: Claude models may refuse some AI safety-related tasks that appear dual-use (e.g., jailbreak datasets).

## Structured Output Generation

Use Pydantic models for type-safe structured data:

```python
from pydantic import BaseModel, Field
from typing import List

class CalendarEvent(BaseModel):
    name: str = Field(description="Event name")
    date: str = Field(description="Event date in YYYY-MM-DD format")
    participants: List[str] = Field(description="List of participant names")
    description: str = Field(description="Event description")

async def generate_structured_sample():
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Generate a fictional team meeting event.")]
        )
    ]

    response = await get_response(
        model="gpt-5-mini",
        messages=messages,
        response_format=CalendarEvent,
        cache_seed=42
    )

    return response.parsed  # Returns validated CalendarEvent instance

event = asyncio.run(generate_structured_sample())
print(f"Event: {event.name} on {event.date}")
```

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
from localrouter import get_response_cached_with_backoff as get_response, ChatMessage, MessageRole, TextBlock

async def generate_single_sample(prompt, seed):
    """Generate a single data sample"""
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model="gpt-5-mini",
        messages=messages,
        temperature=0.8,
        cache_seed=seed
    )

    return response.content[0].text

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

Use tools for complex generation workflows requiring structured interaction:

```python
from localrouter import ToolDefinition, ToolUseBlock, ToolResultBlock

# Define a data generation tool
data_gen_tool = ToolDefinition(
    name="generate_sample",
    description="Generate a data sample with specific parameters",
    input_schema={
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Data category"},
            "format": {"type": "string", "description": "Output format"},
            "count": {"type": "integer", "description": "Number of samples"}
        },
        "required": ["category", "format"]
    }
)

async def generate_with_tools():
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Generate customer feedback data in JSON format")]
        )
    ]

    response = await get_response(
        model="claude-sonnet-4-5-20250929",
        messages=messages,
        tools=[data_gen_tool],
        cache_seed=123
    )

    # Check for tool calls
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            print(f"Tool: {block.name}, Args: {block.input}")

    return response
```

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
2. **Use caching**: Always include `cache_seed` parameter for resumable generation
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
        response = await get_response(
            model="gpt-5-mini",
            messages=[...],
            response_format=DataSchema,
            cache_seed=i
        )
        results.append(response.parsed)
    return results
```

### Pattern 3: Multi-Turn Conversation Data

```python
# Generate conversational data
async def generate_conversation(seed):
    messages = []

    # Initial message
    messages.append(ChatMessage(
        role=MessageRole.user,
        content=[TextBlock(text="Start a conversation about AI ethics")]
    ))

    # Generate multiple turns
    for turn in range(3):
        response = await get_response(
            model="claude-sonnet-4-5-20250929",
            messages=messages,
            cache_seed=seed * 100 + turn
        )
        messages.append(response)

        # Add next user message based on context
        messages.append(ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Continue the discussion")]
        ))

    return messages
```

## Resources

This skill includes helper scripts:

- `scripts/generate_dataset.py` - Production-ready parallel generation with progress tracking
- `scripts/explore_dataset.py` - Comprehensive dataset exploration and validation

## Summary Checklist

When generating synthetic data:

- ✓ Use `get_response_cached_with_backoff` with `cache_seed`
- ✓ Choose appropriate model for task (usually `gpt-5-mini` or `claude-sonnet-4-5-20250929`)
- ✓ Test prompts on 10-20 examples before large runs
- ✓ Implement parallelization with provider-appropriate concurrency limits
- ✓ Ensure diversity through combinatory prompts, augmentation, or seed datasets
- ✓ Write outputs incrementally to files
- ✓ Use terminal tool for long-running processes
- ✓ Explore and validate datasets after generation
- ✓ Use Pydantic models for structured outputs when appropriate
