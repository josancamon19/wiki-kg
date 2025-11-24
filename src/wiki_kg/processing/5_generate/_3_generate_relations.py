"""
Generate batch API requests for relation extraction.

This script:
1. Loads the parsed entities from _2_parse_entities.py output
2. Loads articles from the dataset
3. Matches articles with their extracted entities
4. Generates batch API requests for relation extraction
5. Uses parallel processing for efficiency
"""

import json
import os
import logging
import argparse
from typing import Dict, Any, Optional, List
from multiprocessing import Pool, cpu_count

from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
RESULTS_BASE_PATH = os.path.join(os.path.dirname(__file__), "results")
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts")
MODEL_NAME = "gpt-5-nano"
REASONING_EFFORT = "minimal"
MAX_TOKENS = 100000
CHUNK_THRESHOLD = int(2e5)  # Characters
CHUNK_SIZE = int(1e5)
TEMPERATURE = 1.0

# Cache for prompt template
with open(os.path.join(PROMPTS_PATH, "relations.txt"), "r") as f:
    _RELATION_PROMPT_TEMPLATE = f.read()

_ENTITIES_LOOKUP = {}
with open(
    os.path.join(RESULTS_BASE_PATH, "entities", "parsed_entities.jsonl"), "r"
) as f:
    for line in f:
        data = json.loads(line)
        _ENTITIES_LOOKUP[data["custom_id"]] = data["entities"]


def article_to_batch_requests(article: Dict[str, Any]) -> Dict[str, Any] | None:
    global _ENTITIES_LOOKUP
    text = article.get("text", "")
    if not text:
        return None

    article_id = str(article["id"])
    entities = _ENTITIES_LOOKUP.get(article_id)
    if not entities:
        logger.warning(
            f"No entities found for article {article_id}, skipping, len(entities_lookup): {len(_ENTITIES_LOOKUP)}"
        )
        return None

    entities_str = json.dumps(entities)
    custom_id = str(article_id)
    # TODO: handle chunk depending of custom_id, custom_id_chunk_0...
    prompt = _RELATION_PROMPT_TEMPLATE.replace("{_source_text_}", text)
    prompt = prompt.replace("{_entities_}", entities_str)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL_NAME,
            "input": prompt,
            "max_output_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "reasoning": {"effort": REASONING_EFFORT},
        },
    }


def process_articles_batch(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_requests = [article_to_batch_requests(a) for a in articles]
    all_requests = [r for r in all_requests if r is not None]
    return all_requests


def write_batch_file(requests: List[Dict[str, Any]], output_path: str):
    """Write batch requests to a JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")
    logger.info(f"Wrote {len(requests)} requests to {output_path}")


def generate_relations_batch_file(
    output_path: str,
    limit: Optional[int] = None,
    force: bool = False,
    num_workers: Optional[int] = None,
):
    """
    Generate a single batch JSONL file containing all relation extraction requests.
    Uses multiprocessing for parallel article processing.

    Args:
        entities_file: Path to parsed_entities.jsonl from step 2
        output_path: Path to output JSONL file
        limit: Maximum number of articles to process
        force: If True, regenerate even if file exists
        num_workers: Number of parallel workers (defaults to CPU count)
    """

    # Check if file exists
    if os.path.exists(output_path) and not force:
        logger.info(f"Batch file already exists: {output_path}")
        logger.info("Use --force to regenerate")
        return

    if num_workers is None:
        num_workers = cpu_count()

    logger.info("Starting Relation Batch File Generation...")
    logger.info(f"Model: {MODEL_NAME}, Reasoning: {REASONING_EFFORT}")
    logger.info(f"Workers: {num_workers} parallel processes")
    logger.info(f"Output: {output_path}")

    # Load dataset
    logger.info("Loading dataset josancamon/finewiki...")
    fw = load_dataset(
        "josancamon/finewiki",
        name="default",
        split="en",
        streaming=True,
    )

    all_requests = []
    count = 0
    skipped = 0
    batch_size = num_workers * 10  # Process articles in batches

    # Accumulate articles in batches for parallel processing
    article_batch = []

    with Pool(processes=num_workers) as pool:
        for article in fw:
            if limit and count >= limit:
                break

            # Only process articles that have entities
            if str(article["id"]) in _ENTITIES_LOOKUP:
                article_batch.append(article)
                count += 1
            else:
                skipped += 1

            # When batch is full, process it in parallel
            if len(article_batch) >= batch_size:
                # Split articles across workers
                chunk_size = len(article_batch) // num_workers
                if chunk_size == 0:
                    chunk_size = 1

                article_chunks = [
                    article_batch[i : i + chunk_size]
                    for i in range(0, len(article_batch), chunk_size)
                ]

                # Process chunks in parallel
                results = pool.map(process_articles_batch, article_chunks)

                # Collect results
                for requests in results:
                    all_requests.extend(requests)

                logger.info(
                    f"Processed {count} articles, {len(all_requests)} total requests, {skipped} skipped..."
                )

                # Clear batch
                article_batch = []

        # Process remaining articles
        if article_batch:
            chunk_size = len(article_batch) // num_workers
            if chunk_size == 0:
                chunk_size = 1

            article_chunks = [
                article_batch[i : i + chunk_size]
                for i in range(0, len(article_batch), chunk_size)
            ]

            results = pool.map(process_articles_batch, article_chunks)

            for requests in results:
                all_requests.extend(requests)

    # Write all requests to file
    write_batch_file(all_requests, output_path)
    logger.info(f"Done. Generated {len(all_requests)} requests from {count} articles.")
    logger.info(f"Skipped {skipped} articles without entities.")
    return all_requests


def main():
    parser = argparse.ArgumentParser(
        description="Generate Batch API files for relation extraction from entities"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of articles to process"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if batch file exists",
    )
    parser.add_argument(
        "--entities-file",
        type=str,
        default=None,
        help="Path to parsed_entities.jsonl (defaults to results/entities/parsed_entities.jsonl)",
    )
    args = parser.parse_args()

    relations_dir = os.path.join(RESULTS_BASE_PATH, "relations")
    output_path = os.path.join(relations_dir, "batch.jsonl")

    generate_relations_batch_file(
        output_path=output_path,
        limit=args.limit,
        force=args.force,
        num_workers=os.cpu_count(),
    )


if __name__ == "__main__":
    main()
