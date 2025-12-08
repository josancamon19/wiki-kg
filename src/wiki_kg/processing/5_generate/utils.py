"""Shared utilities for the 5_generate pipeline."""

import json
import re
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


def extract_entities_from_text(text: str) -> list[str] | None:
    """
    Extract the entities list from the response text.

    The text format is:
    [[ ## entities ## ]]
    ["entity1", "entity2", ...]
    [[ ## completed ## ]]
    """
    # Find the content between markers
    pattern = r"\[\[ ## entities ## \]\]\s*\n?(.*?)\s*\n?\[\[ ## completed ## \]\]"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    entities_text = match.group(1).strip()

    # Remove any trailing comments (like "# note: the value you produce...")
    entities_text = re.sub(r"\s*#.*$", "", entities_text, flags=re.MULTILINE)
    entities_text = entities_text.strip()

    # Remove any trailing whitespace or newlines within the JSON
    entities_text = " ".join(entities_text.split())

    # Fix question marks outside quotes (e.g., "text"? -> "text")
    entities_text = re.sub(r'"(\?+)', r'"', entities_text)

    # Fix common malformations at the end
    # Remove trailing characters like ], ), }, etc. and rebuild proper ending
    entities_text = entities_text.rstrip("])}\t ")

    # Sometimes the model outputs wrong closing brackets
    # Make sure it ends with ]
    if not entities_text.endswith("]"):
        entities_text += "]"

    try:
        # Parse the JSON array
        entities = json.loads(entities_text)
        return entities if isinstance(entities, list) else None
    except json.JSONDecodeError as e:
        print(f"Failed to parse entities JSON: {e}")
        print(f"Text: {entities_text[:200]}")
        return None


def extract_relations_from_text(text: str) -> list[dict] | None:
    """
    Extract the relations list from the response text.

    The text format is:
    [[ ## relations ## ]]
    [{"subject": "...", "predicate": "...", "object": "..."}, ...]
    [[ ## completed ## ]]
    """
    pattern = r"\[\[\s*##\s*relations\s*##\s*\]\]\s*\n?(.*?)\s*\n?\[\[\s*##\s*completed\s*##\s*\]\]"
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return None

    relations_text = match.group(1).strip()
    relations_text = re.sub(r"\s*#.*$", "", relations_text, flags=re.MULTILINE)
    relations_text = relations_text.strip()

    try:
        relations = json.loads(relations_text)
        return relations if isinstance(relations, list) else None
    except json.JSONDecodeError as e:
        print(f"Failed to parse relations JSON: {e}")
        print(f"Text preview: {relations_text[:300]}")
        return None
