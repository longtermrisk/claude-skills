#!/usr/bin/env python3
"""
Production-ready script for parallel synthetic data generation via the org's
LiteLLM proxy (https://litellm.nielsrolf.com).

This script demonstrates best practices for large-scale dataset generation:
- Parallel execution with appropriate concurrency limits
- Incremental writing to output file
- Progress tracking and error handling
- Proxy-side response caching for resumable generation: each request opts in
  with {"cache": {"use-cache": True}} and carries a distinct seed, so
  re-running the script after an interruption serves completed samples from
  cache instantly.
"""

import asyncio
import json
import os
import argparse
from typing import List

from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.environ['LITELLM_API_KEY'],
    base_url="https://litellm.nielsrolf.com",
    # Cloudflare blocks the SDK's default User-Agent — see the litellm skill
    default_headers={"User-Agent": "litellm-client/1.0"},
    max_retries=5,
)


async def generate_single_sample(
    prompt: str,
    seed: int,
    model: str = "openai/gpt-5-mini",
    temperature: float = 0.8,
    **kwargs
) -> str:
    """Generate a single data sample via the LiteLLM proxy.

    Args:
        prompt: The generation prompt
        seed: Per-sample seed; part of the proxy cache key
        model: Model identifier (default: openai/gpt-5-mini)
        temperature: Sampling temperature (default: 0.8)
        **kwargs: Additional arguments to pass to chat.completions.create

    Returns:
        Generated text content
    """
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        seed=seed,
        extra_body={"cache": {"use-cache": True}},
        **kwargs
    )

    return response.choices[0].message.content


async def generate_dataset(
    prompts: List[str],
    output_file: str = "dataset.jsonl",
    model: str = "openai/gpt-5-mini",
    temperature: float = 0.8,
    max_concurrent: int = 50,
    batch_size: int = 100,
    start_seed: int = 0,
    **kwargs
) -> None:
    """Generate a large dataset with parallelization and incremental writing.

    Args:
        prompts: List of prompts to generate from
        output_file: Output file path (JSONL format)
        model: Model identifier (default: openai/gpt-5-mini)
        temperature: Sampling temperature (default: 0.8)
        max_concurrent: Maximum concurrent requests (default: 50)
        batch_size: Number of samples to process before writing (default: 100)
        start_seed: Starting seed value (default: 0)
        **kwargs: Additional arguments to pass to chat.completions.create
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_with_semaphore(prompt: str, seed: int) -> str:
        async with semaphore:
            try:
                return await generate_single_sample(
                    prompt=prompt,
                    seed=seed,
                    model=model,
                    temperature=temperature,
                    **kwargs
                )
            except Exception as e:
                print(f"Error generating sample {seed}: {e}")
                return f"ERROR: {str(e)}"

    # Generate tasks
    tasks = [
        generate_with_semaphore(prompt, start_seed + seed)
        for seed, prompt in enumerate(prompts)
    ]

    total_batches = (len(tasks) - 1) // batch_size + 1

    # Process in batches and write incrementally
    with open(output_file, 'w') as f:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            results = await asyncio.gather(*batch)

            # Write batch results
            for j, result in enumerate(results):
                data = {
                    "id": i + j,
                    "prompt": prompts[i + j],
                    "response": result,
                    "seed": start_seed + i + j
                }
                f.write(json.dumps(data) + '\n')
                f.flush()  # Ensure data is written immediately

            current_batch = i // batch_size + 1
            print(f"Completed batch {current_batch}/{total_batches} "
                  f"({len(results)} samples)")

    print(f"\nDataset generation complete! Saved to {output_file}")
    print(f"Total samples: {len(prompts)}")


def main():
    """Command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets via the LiteLLM proxy"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="Path to file containing prompts (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.jsonl",
        help="Output file path (default: dataset.jsonl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5-mini",
        help="Model identifier (default: openai/gpt-5-mini)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent requests (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for incremental writing (default: 100)"
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Starting seed value (default: 0)"
    )

    args = parser.parse_args()

    # Load prompts from file
    with open(args.prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    print(f"Model: {args.model}")
    print(f"Max concurrent requests: {args.max_concurrent}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output}\n")

    # Run generation
    asyncio.run(generate_dataset(
        prompts=prompts,
        output_file=args.output,
        model=args.model,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
        start_seed=args.start_seed
    ))


if __name__ == "__main__":
    main()
