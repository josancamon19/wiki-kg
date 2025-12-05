import json
import os
import logging
import argparse
from typing import Dict, Any, Optional, List
from multiprocessing import Pool, cpu_count
from pathlib import Path

import gcsfs
from datasets import load_dataset
from dotenv import load_dotenv
from kg_gen.utils.chunk_text import chunk_text

# Load environment variables
load_dotenv()

# Configure logging
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
GCP_KG_PREFIX = "gs://wikipedia-graph/graph"
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts")
MODEL_NAME = "gpt-5-nano"
REASONING_EFFORT = "minimal"
MAX_TOKENS = 100000
CHUNK_THRESHOLD = int(2e5)  # Characters
CHUNK_SIZE = int(1e5)
TEMPERATURE = 1.0

# Cache for prompt template
with open(os.path.join(PROMPTS_PATH, "entities.txt"), "r") as f:
    _ENTITY_PROMPT_TEMPLATE = f.read()


def article_to_batch_requests(article: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert an article to batch API request(s) for entity extraction.
    Returns a list of batch requests (one per chunk if article is large).
    """
    text = article.get("text", "")
    if not text:
        return []

    article_id = str(article["id"])

    # Determine if chunking is needed
    if len(text) > CHUNK_THRESHOLD:
        chunks = chunk_text(text, CHUNK_SIZE)
        logger.info(f"Article {article_id}: chunked into {len(chunks)} pieces")
    else:
        chunks = [text]

    batch_requests = []
    for chunk_idx, chunk in enumerate(chunks):
        # Create custom_id with article_id and chunk index
        if len(chunks) > 1:
            custom_id = f"{article_id}_chunk_{chunk_idx}"
        else:
            custom_id = str(article_id)

        prompt = _ENTITY_PROMPT_TEMPLATE.replace("{_source_text_}", chunk)

        # OpenAI Batch API format
        batch_request = {
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
        batch_requests.append(batch_request)

    return batch_requests


def process_articles_batch(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a batch of articles in parallel.
    This function is called by each worker process.
    """
    all_requests = []
    for article in articles:
        requests = article_to_batch_requests(article)
        all_requests.extend(requests)
    return all_requests


def write_batch_file(
    requests: List[Dict[str, Any]], output_path: str, fs: gcsfs.GCSFileSystem
):
    """Write batch requests to a JSONL file on GCS."""
    # Ensure output directory exists
    output_dir = "/".join(output_path.replace("gs://", "").split("/")[:-1])
    fs.makedirs(output_dir, exist_ok=True)

    # Write to GCS
    with fs.open(output_path, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")
    logger.info(f"Wrote {len(requests)} requests to {output_path}")


def generate_entities_batch_file(
    output_path: str,
    wiki: str = "enwiki",
    limit: Optional[int] = None,
    force: bool = False,
    num_workers: Optional[int] = None,
):
    """
    Generate a single batch JSONL file containing all entity extraction requests.
    Uses multiprocessing for parallel article processing.
    Saves results to Google Cloud Storage.

    Args:
        output_path: GCS path to output JSONL file (e.g., gs://bucket/path/batch.jsonl)
        wiki: Wiki identifier (e.g., 'enwiki')
        limit: Maximum number of articles to process
        force: If True, regenerate even if file exists
        num_workers: Number of parallel workers (defaults to CPU count)
    """
    # Initialize GCS filesystem
    fs = gcsfs.GCSFileSystem()

    # Check if file exists on GCS
    if fs.exists(output_path) and not force:
        logger.info(f"Batch file already exists: {output_path}")
        logger.info("Use --force to regenerate")
        return

    if num_workers is None:
        num_workers = cpu_count()

    logger.info("=" * 80)
    logger.info("Starting Entity Batch File Generation")
    logger.info("=" * 80)
    logger.info(f"Wiki: {wiki}")
    logger.info(f"Model: {MODEL_NAME}, Reasoning: {REASONING_EFFORT}")
    logger.info(f"Workers: {num_workers} parallel processes")
    logger.info(f"Output: {output_path}")
    logger.info(f"Limit: {limit if limit else 'None (all articles)'}")

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
    batch_size = num_workers * 10  # Process articles in batches

    # Accumulate articles in batches for parallel processing
    article_batch = []

    with Pool(processes=num_workers) as pool:
        for article in fw:
            if limit and count >= limit:
                break

            article_batch.append(article)
            count += 1

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
                    f"Processed {count} articles, {len(all_requests)} total requests..."
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

    # Write all requests to GCS
    write_batch_file(all_requests, output_path, fs)
    logger.info(f"Done. Generated {len(all_requests)} requests from {count} articles.")
    logger.info("=" * 80)
    return all_requests


def main():
    parser = argparse.ArgumentParser(
        description="Generate Batch API files for Knowledge Graph extraction from FineWiki"
    )
    parser.add_argument(
        "--wiki",
        type=str,
        default="enwiki",
        help="Wiki identifier (default: enwiki)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of articles to process"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if batch file exists",
    )
    args = parser.parse_args()

    # Generate GCS path for output
    output_path = f"{GCP_KG_PREFIX}/{args.wiki}/entities/batch.jsonl"

    logger.info(f"Script: {__file__}")
    logger.info(f"Arguments: {vars(args)}")

    generate_entities_batch_file(
        output_path=output_path, limit=args.limit, force=args.force, wiki=args.wiki
    )


if __name__ == "__main__":
    main()
