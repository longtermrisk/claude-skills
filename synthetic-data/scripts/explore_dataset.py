#!/usr/bin/env python3
"""
Dataset exploration and validation script.

This script provides comprehensive analysis of generated datasets:
- Basic statistics and shape information
- Sample examples for manual review
- Categorical distribution analysis
- Duplicate detection
- Text length analysis for content fields
- Missing value detection
"""

import argparse
import json
import pandas as pd
from typing import Optional


def explore_dataset(
    file_path: str,
    n_samples: int = 5,
    format: str = "auto"
) -> None:
    """Explore and validate a dataset with comprehensive statistics.

    Args:
        file_path: Path to dataset file
        n_samples: Number of random samples to display (default: 5)
        format: File format - 'csv', 'json', 'jsonl', or 'auto' (default: auto)
    """
    print(f"Exploring dataset: {file_path}\n")
    print("=" * 80)

    # Load dataset
    if format == "auto":
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.jsonl'):
            df = pd.read_json(file_path, lines=True)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(
                f"Cannot auto-detect format for {file_path}. "
                "Please specify --format"
            )
    elif format == "csv":
        df = pd.read_csv(file_path)
    elif format == "jsonl":
        df = pd.read_json(file_path, lines=True)
    elif format == "json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Basic information
    print("\n📊 BASIC INFORMATION")
    print("-" * 80)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nColumn types:")
    print(df.dtypes)

    # Missing values
    print("\n\n❓ MISSING VALUES")
    print("-" * 80)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found ✓")
    else:
        print(missing[missing > 0])

    # Duplicates
    print("\n\n🔄 DUPLICATES")
    print("-" * 80)
    n_duplicates = df.duplicated().sum()
    if n_duplicates == 0:
        print("No duplicate rows found ✓")
    else:
        print(f"Found {n_duplicates} duplicate rows ({n_duplicates/len(df)*100:.2f}%)")
        pct_unique = df.drop_duplicates().shape[0] / len(df) * 100
        print(f"Unique rows: {df.drop_duplicates().shape[0]} ({pct_unique:.2f}%)")

    # Numerical columns analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print("\n\n📈 NUMERICAL COLUMNS")
        print("-" * 80)
        print(df[numeric_cols].describe())

    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0 and len(categorical_cols) <= 10:
        print("\n\n📋 CATEGORICAL COLUMNS")
        print("-" * 80)
        for col in categorical_cols:
            unique_count = df[col].nunique()
            # Only show value counts for columns with reasonable cardinality
            if unique_count <= 20:
                print(f"\n{col} (unique values: {unique_count}):")
                print(df[col].value_counts().head(10))
            else:
                print(f"\n{col}: {unique_count} unique values (too many to display)")

    # Text length analysis
    text_cols = [col for col in df.columns
                 if 'text' in col.lower() or 'content' in col.lower()
                 or 'response' in col.lower() or 'prompt' in col.lower()]

    if text_cols:
        print("\n\n📝 TEXT LENGTH ANALYSIS")
        print("-" * 80)
        for col in text_cols:
            if col in df.columns:
                df[f'{col}_length'] = df[col].astype(str).str.len()
                stats = df[f'{col}_length'].describe()
                print(f"\n{col}:")
                print(f"  Mean length: {stats['mean']:.1f} characters")
                print(f"  Median length: {stats['50%']:.1f} characters")
                print(f"  Min length: {stats['min']:.0f} characters")
                print(f"  Max length: {stats['max']:.0f} characters")
                print(f"  Std dev: {stats['std']:.1f} characters")

    # Random samples
    print(f"\n\n🎲 RANDOM SAMPLES (n={n_samples})")
    print("-" * 80)
    samples = df.sample(min(n_samples, len(df)))

    for idx, row in samples.iterrows():
        print(f"\nSample {idx}:")
        for col in df.columns:
            value = row[col]
            # Truncate long text for display
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            print(f"  {col}: {value}")
        print()

    # Summary
    print("\n" + "=" * 80)
    print("✓ Exploration complete!")


def main():
    """Command-line interface for dataset exploration."""
    parser = argparse.ArgumentParser(
        description="Explore and validate synthetic datasets"
    )
    parser.add_argument(
        "file",
        type=str,
        help="Path to dataset file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of random samples to display (default: 5)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["auto", "csv", "json", "jsonl"],
        default="auto",
        help="File format (default: auto-detect)"
    )

    args = parser.parse_args()

    try:
        explore_dataset(
            file_path=args.file,
            n_samples=args.samples,
            format=args.format
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
