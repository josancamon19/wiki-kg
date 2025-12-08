"""
Generate batch API requests for relation extraction.

Loads parsed entities, streams articles, and generates batch requests
for articles that have entities.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, Annotated
from pathlib import Path

import gcsfs
import typer
from datasets import load_dataset
from dotenv import load_dotenv

try:
    from .utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        build_filename,
    )
except ImportError:
    from utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        build_filename,
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
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts")
MAX_TOKENS = 100000
TEMPERATURE = 1.0

with open(os.path.join(PROMPTS_PATH, "relations.txt"), "r") as f:
    RELATION_PROMPT_TEMPLATE = f.read()

app = typer.Typer()


def load_entities(entities_file: str, fs: gcsfs.GCSFileSystem) -> Dict[str, list]:
    """Load entities from JSONL into a dict: article_id -> entities list."""
    logger.info(f"Loading entities from: {entities_file}")
    entities = {}
    with fs.open(entities_file, "r") as f:
        for line in f:
            data = json.loads(line)
            entities[data["custom_id"]] = data["entities"]
    logger.info(f"Loaded {len(entities)} entities")
    return entities


def make_batch_request(
    article_id: str,
    text: str,
    entities: list,
    model: str,
    reasoning_effort: str,
) -> Dict[str, Any]:
    """Create a single batch API request."""
    prompt = RELATION_PROMPT_TEMPLATE.replace("{_source_text_}", text)
    prompt = prompt.replace("{_entities_}", json.dumps(entities))
    return {
        "custom_id": article_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": prompt,
            "max_output_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "reasoning": {"effort": reasoning_effort},
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
    entities_dict = load_entities(entities_file, fs)
    entity_ids = set(entities_dict.keys())

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

    for article in dataset:
        scanned += 1
        article_id = str(article["id"])

        if article_id in entity_ids:
            text = article.get("text", "")
            if text:
                req = make_batch_request(
                    article_id, text, entities_dict[article_id], model, reasoning_effort
                )
                requests.append(req)
                found += 1

        # Progress logging
        if scanned % 10000 == 0:
            logger.info(f"Scanned {scanned} articles, found {found}/{len(entity_ids)}")

        # Stop conditions
        if found >= len(entity_ids):
            logger.info(f"Found all {len(entity_ids)} articles")
            break
        if limit and found >= limit:
            logger.info(f"Reached limit of {limit}")
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
    entities_filename = build_filename("entities_parsed", model, reasoning_effort, limit)
    entities_file = f"{GCP_KG_PREFIX}/{wiki}/entities/{entities_filename}"

    output_filename = build_filename("relations_batch", model, reasoning_effort, limit)
    output_path = f"{GCP_KG_PREFIX}/{wiki}/relations/{output_filename}"

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


if __name__ == "__main__":
    app()
