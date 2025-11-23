"""
Generate estimate summary files from collected KGGen data.

This script:
1. Reads all JSON files from a target directory in kggen_estimates/
2. Groups articles by their length bucket
3. Calculates average prompt and completion tokens per bucket
4. Saves summary to {directory-name}-estimates.json

Usage:
    uv run python src/wiki_kg/scripts/_2_kggen_samples_2.py gpt-5-nano-medium
    uv run python src/wiki_kg/scripts/_2_kggen_samples_2.py gpt-5-nano-minimal

The script expects the target directory to exist in:
    analysis/kggen_estimates/{target_directory}/

And will output to:
    analysis/kggen_estimates/{target_directory}-estimates.json
"""

import json
import typer
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


def round_to_nearest_bucket(length: int) -> int:
    """Round article length to nearest bucket."""
    # Define length buckets: every 500 chars up to 15k, then every 5k chars
    if length <= 15000:
        bucket = round(length / 500) * 500
    elif length <= 100000:
        bucket = round(length / 5000) * 5000
    else:
        bucket = 100000

    # Ensure minimum bucket is 500
    return max(500, bucket)


def process_directory(directory_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Process all JSON files in a directory and compute statistics.

    Args:
        directory_path: Path to directory containing individual article JSON files

    Returns:
        Dictionary mapping bucket sizes to statistics
    """
    # Group articles by bucket
    bucket_data = defaultdict(list)

    # Read all JSON files
    json_files = list(directory_path.glob("*.json"))

    if not json_files:
        print(f"⚠️  No JSON files found in {directory_path}")
        return {}

    print(f"Processing {len(json_files)} files from {directory_path.name}...")

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Extract relevant data
            article_length = data.get("article_length")
            if not article_length:
                print(f"  ⚠️  Skipping {json_file.name}: missing article_length")
                continue

            # Get bucket (either from data or calculate from length)
            bucket = data.get("bucket") or round_to_nearest_bucket(article_length)

            # Extract token counts from extraction field
            extraction = data.get("extraction", {})
            tokens = extraction.get("tokens", {})

            prompt_tokens = tokens.get("prompt_tokens", 0)
            completion_tokens = tokens.get("completion_tokens", 0)

            if prompt_tokens == 0 and completion_tokens == 0:
                print(f"  ⚠️  Skipping {json_file.name}: no token data")
                continue

            bucket_data[bucket].append(
                {
                    "article_id": data.get("article_id"),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )

        except json.JSONDecodeError:
            print(f"  ❌ Error reading {json_file.name}")
            continue
        except Exception as e:
            print(f"  ❌ Error processing {json_file.name}: {e}")
            continue

    # Calculate averages per bucket
    bucket_estimates = {}

    for bucket in sorted(bucket_data.keys()):
        articles = bucket_data[bucket]

        avg_prompt = sum(a["prompt_tokens"] for a in articles) / len(articles)
        avg_completion = sum(a["completion_tokens"] for a in articles) / len(articles)

        bucket_estimates[str(bucket)] = {
            "avg_prompt_tokens": round(avg_prompt, 2),
            "avg_completion_tokens": round(avg_completion, 2),
            "article_count": len(articles),
        }

        print(
            f"  [{bucket:6d}] {len(articles):3d} articles - "
            f"Prompt: {avg_prompt:7.1f} tokens, "
            f"Completion: {avg_completion:7.1f} tokens"
        )

    return bucket_estimates


def main(
    target_directory: str = typer.Argument(
        ...,
        help="Name of the target directory within analysis/kggen_estimates/ (e.g., 'gpt-5-nano-medium')",
    ),
):
    """Generate estimate summary from collected KGGen data."""

    # Construct paths
    base_dir = (
        Path(__file__).parent.parent.parent.parent / "analysis" / "kggen_estimates"
    )
    target_path = base_dir / target_directory

    if not target_path.exists():
        print(f"❌ Error: Directory not found: {target_path}")
        print(f"\nAvailable directories in {base_dir}:")
        for d in base_dir.iterdir():
            if d.is_dir():
                print(f"  - {d.name}")
        raise typer.Exit(1)

    if not target_path.is_dir():
        print(f"❌ Error: {target_path} is not a directory")
        raise typer.Exit(1)

    print("=" * 80)
    print("KGGen Estimates Summary Generator")
    print("=" * 80)
    print(f"Source Directory: {target_path}")
    print("=" * 80)
    print()

    # Process the directory
    estimates = process_directory(target_path)

    if not estimates:
        print("\n❌ No valid data found to process")
        raise typer.Exit(1)

    # Save results
    output_file = base_dir / f"{target_directory}-estimates.json"
    with open(output_file, "w") as f:
        json.dump(estimates, f, indent=2)

    print()
    print("=" * 80)
    print("SUMMARY COMPLETE")
    print("=" * 80)
    print(f"✅ Processed {len(estimates)} buckets")
    print(f"✅ Saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    typer.run(main)
