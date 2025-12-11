"""Shared utilities for the 5_generate pipeline."""

from typing import Optional

# Constants
GCP_KG_PREFIX = "gs://wikipedia-graph/graph"
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_REASONING_EFFORT = "minimal"


def build_subdir(
    model: str,
    reasoning_effort: str,
    limit: Optional[int] = None,
) -> str:
    """
    Build a subdirectory path for model/reasoning_effort organization.

    Examples:
        >>> build_subdir("gpt-5-nano", "minimal")
        'gpt-5-nano/minimal'
        >>> build_subdir("gpt-5-nano", "high", limit=100)
        'gpt-5-nano/high_l100'
    """
    suffix = f"_l{limit}" if limit else ""
    return f"{model}/{reasoning_effort}{suffix}"


def build_filename(
    base: str,
    ext: str = ".jsonl",
) -> str:
    """
    Build a simple filename without model/reasoning parameters.

    Examples:
        >>> build_filename("batch")
        'batch.jsonl'
        >>> build_filename("batch_info", ext=".json")
        'batch_info.json'
    """
    return f"{base}{ext}"


def get_batch_paths(
    batch_type: str,
    wiki: str,
    model: str,
    reasoning_effort: str,
    limit: Optional[int] = None,
) -> dict:
    """
    Get the standard GCS file paths for a batch type.

    Args:
        batch_type: Type of batch (e.g., 'entities', 'relations')
        wiki: Wiki identifier (default: 'enwiki')
        model: Model name used for generation
        reasoning_effort: Reasoning effort level
        limit: Optional limit used during generation

    Returns:
        Dictionary with GCS paths for batch files
    """
    subdir = build_subdir(model, reasoning_effort, limit)
    batch_dir = f"{GCP_KG_PREFIX}/{wiki}/{batch_type}/{subdir}"

    return {
        "dir": batch_dir,
        "batch_file": f"{batch_dir}/batch.jsonl",
        "info_file": f"{batch_dir}/batch_info.json",
        "results_file": f"{batch_dir}/batch_results.jsonl",
    }
