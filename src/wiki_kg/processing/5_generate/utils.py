"""Shared utilities for the 5_generate pipeline."""

from typing import Optional, Dict, List, Any

import json

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


def load_entities_map(
    entities_file: str,
    fs: Any,
) -> Dict[str, List[str]]:
    """
    Load entities from entities `parsed.jsonl` into a dict: custom_id -> entities list.

    This is shared by downstream steps like relation batch generation and relation parsing.
    """
    entities_map: Dict[str, List[str]] = {}
    with fs.open(entities_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except Exception:
                # Skip malformed lines; upstream failures are tracked elsewhere.
                continue

            custom_id = data.get("custom_id")
            entities = data.get("entities")

            if not isinstance(custom_id, str) or not isinstance(entities, list):
                continue

            entities_map[custom_id] = [e for e in entities if isinstance(e, str)]

    return entities_map
