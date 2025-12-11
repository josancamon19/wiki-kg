"""
Generate batch API requests for relation extraction.

Loads parsed entities, streams articles, and generates batch requests
for articles that have entities.
"""

import json
import logging
from typing import Dict, Any, Optional, Annotated, List
from pathlib import Path

import gcsfs
import typer
from datasets import load_dataset
from dotenv import load_dotenv
from kg_gen.steps._2_get_relations import _load_relations_prompt, _create_relations_model

try:
    from .utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        build_subdir,
        load_entities_map,
    )
except ImportError:
    from utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        build_subdir,
        load_entities_map,
    )

load_dotenv()

# Logging
HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / "generation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Constants
MAX_TOKENS = 100000
TEMPERATURE = 1.0

# Load prompt template from kg_gen package
_RELATION_SYSTEM_PROMPT = _load_relations_prompt()


def _create_relations_schema(entities: List[str]) -> Dict[str, Any]:
    """Create a JSON schema for relations with entity constraints using kg_gen's model."""
    _, RelationsResponse = _create_relations_model(entities)
    schema = RelationsResponse.model_json_schema()
    schema["additionalProperties"] = False
    # Also set additionalProperties on nested objects
    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            if def_schema.get("type") == "object":
                def_schema["additionalProperties"] = False
    return schema

app = typer.Typer()


def make_batch_request(
    article_id: str,
    text: str,
    entities: list,
    model: str,
    reasoning_effort: str,
) -> Dict[str, Any]:
    """Create a single batch API request with system/user messages and structured output."""
    # Build user prompt with entities and text tags (matching kg_gen format)
    entities_str = "\n".join(f"- {e}" for e in entities)
    user_prompt = f"""
Here is the list of entities that were previously extracted from the source text:

<entities>
{entities_str}
</entities>

Here is the source text to analyze:

<text>
{text}
</text>
"""
    
    # Create schema with entity constraints
    schema = _create_relations_schema(entities)
    
    return {
        "custom_id": article_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {"role": "system", "content": _RELATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "reasoning": {"effort": reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "relations_response",
                    "schema": schema,
                    "strict": True,
                }
            },
        },
    }


def generate_relations_batch_file(
    entities_file: str,
    output_path: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    limit: Optional[int] = None,
    force: bool = False,
):
    """Generate batch JSONL file for relation extraction."""
    fs = gcsfs.GCSFileSystem()

    if fs.exists(output_path) and not force:
        logger.info(f"Batch file already exists: {output_path}. Use --force to regenerate.")
        return

    # Load all entities into memory
    logger.info(f"Loading entities from: {entities_file}")
    entities_dict = load_entities_map(entities_file, fs)
    entity_ids = set(entities_dict.keys())
    logger.info(f"Loaded {len(entities_dict)} entities")
    
    # Debug: show sample entity IDs
    sample_ids = list(entity_ids)[:3]
    logger.info(f"Sample entity IDs: {sample_ids}")

    logger.info("=" * 80)
    logger.info("Starting Relation Batch File Generation")
    logger.info(f"Model: {model}, Reasoning: {reasoning_effort}")
    logger.info(f"Looking for {len(entity_ids)} articles with entities")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 80)

    # Stream dataset
    logger.info("Loading dataset josancamon/finewiki...")
    dataset = load_dataset("josancamon/finewiki", name="default", split="en", streaming=True)

    requests = []
    found = 0
    scanned = 0
    
    # Max articles to scan - if limit is set, we expect entities to be in first N articles
    # Add buffer for safety (10x limit or 10k minimum)
    max_scan = max(10000, (limit or 100) * 10) if limit else None
    if max_scan:
        logger.info(f"Max scan limit: {max_scan} articles")

    for article in dataset:
        scanned += 1
        article_id = str(article["id"])
        
        # Debug: show first few dataset article IDs
        if scanned <= 3:
            logger.info(f"Dataset article ID sample: '{article_id}'")

        if article_id in entity_ids:
            text = article.get("text", "")
            if text:
                req = make_batch_request(
                    article_id, text, entities_dict[article_id], model, reasoning_effort
                )
                requests.append(req)
                found += 1

        # Progress logging - more frequent at start
        if scanned <= 100 and scanned % 10 == 0:
            logger.info(f"Scanned {scanned} articles, found {found}/{len(entity_ids)}")
        elif scanned % 10000 == 0:
            logger.info(f"Scanned {scanned} articles, found {found}/{len(entity_ids)}")

        # Stop conditions
        if found >= len(entity_ids):
            logger.info(f"Found all {len(entity_ids)} articles")
            break
        if limit and found >= limit:
            logger.info(f"Reached limit of {limit}")
            break
        if max_scan and scanned >= max_scan:
            logger.warning(f"Reached max scan limit ({max_scan}), stopping. Found {found}/{len(entity_ids)}")
            break

    # Write output
    output_dir = "/".join(output_path.replace("gs://", "").split("/")[:-1])
    fs.makedirs(output_dir, exist_ok=True)

    with fs.open(output_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Wrote {len(requests)} requests to {output_path}")
    logger.info("=" * 80)


@app.command()
def main(
    wiki: Annotated[str, typer.Option(help="Wiki identifier")] = "enwiki",
    model: Annotated[str, typer.Option(help="Model name")] = DEFAULT_MODEL,
    reasoning_effort: Annotated[str, typer.Option(help="Reasoning effort")] = DEFAULT_REASONING_EFFORT,
    limit: Annotated[Optional[int], typer.Option(help="Limit articles")] = None,
    force: Annotated[bool, typer.Option(help="Force regeneration")] = False,
):
    """Generate Batch API files for relation extraction."""
    subdir = build_subdir(model, reasoning_effort, limit)
    entities_file = f"{GCP_KG_PREFIX}/{wiki}/entities/{subdir}/parsed.jsonl"
    output_path = f"{GCP_KG_PREFIX}/{wiki}/relations/{subdir}/batch.jsonl"

    logger.info(f"Script: {__file__}")
    logger.info(f"Args: wiki={wiki}, model={model}, reasoning_effort={reasoning_effort}, limit={limit}, force={force}")

    generate_relations_batch_file(
        entities_file=entities_file,
        output_path=output_path,
        model=model,
        reasoning_effort=reasoning_effort,
        limit=limit,
        force=force,
    )

    # Force exit to clean up streaming dataset background threads
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
