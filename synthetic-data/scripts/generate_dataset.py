#!/usr/bin/env python3
"""
Production-ready script for parallel synthetic data generation using localrouter.

This script demonstrates best practices for large-scale dataset generation:
- Parallel execution with appropriate concurrency limits
- Incremental writing to output file
- Progress tracking and error handling
- Caching support for resumable generation
"""

import asyncio
import json
import argparse
from typing import List, Callable, Any
from localrouter import (
    get_response_cached_with_backoff as get_response,
    ChatMessage,
    MessageRole,
    TextBlock
)


async def generate_single_sample(
    prompt: str,
    seed: int,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.8,
    **kwargs
) -> str:
    """Generate a single data sample using localrouter.

    Args:
        prompt: The generation prompt
        seed: Cache seed for reproducible generation
        model: Model identifier (default: gpt-4.1-mini)
        temperature: Sampling temperature (default: 0.8)
        **kwargs: Additional arguments to pass to get_response

    Returns:
        Generated text content
    """
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        temperature=temperature,
        cache_seed=seed,
        **kwargs
    )

    return response.content[0].text


async def generate_dataset(
    prompts: List[str],
    output_file: str = "dataset.jsonl",
    model: str = "gpt-4.1-mini",
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
        model: Model identifier (default: gpt-4.1-mini)
        temperature: Sampling temperature (default: 0.8)
        max_concurrent: Maximum concurrent requests (default: 50)
        batch_size: Number of samples to process before writing (default: 100)
        start_seed: Starting cache seed value (default: 0)
        **kwargs: Additional arguments to pass to get_response
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
        description="Generate synthetic datasets using localrouter"
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
        default="gpt-4.1-mini",
        help="Model identifier (default: gpt-4.1-mini)"
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
        help="Starting cache seed value (default: 0)"
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
