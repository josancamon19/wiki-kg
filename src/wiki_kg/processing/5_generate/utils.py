"""Shared utilities for the 5_generate pipeline."""

from typing import Optional

# Constants
GCP_KG_PREFIX = "gs://wikipedia-graph/graph"
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_REASONING_EFFORT = "minimal"


def build_filename(
    base: str,
    model: str,
    reasoning_effort: str,
    limit: Optional[int] = None,
    ext: str = ".jsonl",
) -> str:
    """
    Build filename with model, reasoning effort, and optional limit suffix.

    Examples:
        >>> build_filename("batch", "gpt-5-nano", "minimal")
        'batch_gpt-5-nano_minimal.jsonl'
        >>> build_filename("batch", "gpt-5-nano", "high", limit=100)
        'batch_gpt-5-nano_high_l100.jsonl'
    """
    suffix = f"_l{limit}" if limit else ""
    return f"{base}_{model}_{reasoning_effort}{suffix}{ext}"


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
    batch_dir = f"{GCP_KG_PREFIX}/{wiki}/{batch_type}"
    batch_filename = build_filename("batch", model, reasoning_effort, limit)
    info_filename = build_filename("batch_info", model, reasoning_effort, limit, ext=".json")
    results_filename = build_filename("batch_results", model, reasoning_effort, limit)

    return {
        "dir": batch_dir,
        "batch_file": f"{batch_dir}/{batch_filename}",
        "info_file": f"{batch_dir}/{info_filename}",
        "results_file": f"{batch_dir}/{results_filename}",
    }
