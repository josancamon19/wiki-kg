"""
Generate batch API requests for relation extraction.

This script:
1. Loads the parsed entities from _2_parse_entities.py output
2. Loads articles from the dataset
3. Matches articles with their extracted entities
4. Generates batch API requests for relation extraction
5. Uses parallel processing for efficiency
6. Uses Google Cloud Storage for file operations
"""

import json
import os
import logging
import argparse
import sqlite3
import tempfile
from typing import Dict, Any, Optional, List
from multiprocessing import Pool, cpu_count
from pathlib import Path

import gcsfs
from datasets import load_dataset
from dotenv import load_dotenv

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
with open(os.path.join(PROMPTS_PATH, "relations.txt"), "r") as f:
    _RELATION_PROMPT_TEMPLATE = f.read()

# Global path to SQLite database - will be set at runtime
_ENTITIES_DB_PATH = None

## ========= DB UTILS ===========


def get_entities_from_db(article_id: str) -> Optional[List[str]]:
    """Lookup entities from SQLite database."""
    global _ENTITIES_DB_PATH
    if _ENTITIES_DB_PATH is None:
        return None

    conn = sqlite3.connect(_ENTITIES_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT entities FROM entities WHERE custom_id = ?", (article_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return json.loads(row[0])
    return None


def build_entities_db(entities_file: str, fs: gcsfs.GCSFileSystem) -> str:
    """
    Build an SQLite database from entities JSONL file.
    Returns path to the database file.

    This is much more memory-efficient than loading everything into a dict -
    SQLite uses disk-backed storage with B-tree indexes for O(log n) lookups.
    """
    # Create temp database file
    db_path = tempfile.mktemp(suffix=".db")
    logger.info(f"Building entities SQLite database at: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table with index for fast lookups
    cursor.execute("""
        CREATE TABLE entities (
            custom_id TEXT PRIMARY KEY,
            entities TEXT
        )
    """)

    # Insert entities in batches for better performance
    batch_size = 10000
    batch = []
    count = 0

    logger.info(f"Loading entities from: {entities_file}")
    with fs.open(entities_file, "r") as f:
        for line in f:
            data = json.loads(line)
            batch.append((data["custom_id"], json.dumps(data["entities"])))
            count += 1

            if len(batch) >= batch_size:
                cursor.executemany(
                    "INSERT INTO entities (custom_id, entities) VALUES (?, ?)", batch
                )
                conn.commit()
                batch = []
                if count % 100000 == 0:
                    logger.info(f"  Loaded {count} entities...")

    # Insert remaining
    if batch:
        cursor.executemany(
            "INSERT INTO entities (custom_id, entities) VALUES (?, ?)", batch
        )
        conn.commit()

    conn.close()
    logger.info(f"Built entities database with {count} entries")
    return db_path


def get_entity_ids_from_db(db_path: str) -> set:
    """Get all entity IDs from the database for filtering."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT custom_id FROM entities")
    ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    return ids


## =========


def article_to_batch_requests(article: Dict[str, Any]) -> Dict[str, Any] | None:
    text = article.get("text", "")
    if not text:
        return None

    article_id = str(article["id"])
    entities = get_entities_from_db(article_id)
    if not entities:
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


def generate_relations_batch_file(
    entities_file: str,
    output_path: str,
    wiki: str = "enwiki",
    limit: Optional[int] = None,
    force: bool = False,
    num_workers: Optional[int] = None,
):
    """
    Generate a single batch JSONL file containing all relation extraction requests.
    Uses multiprocessing for parallel article processing.
    Saves results to Google Cloud Storage.

    Args:
        entities_file: GCS path to parsed_entities.jsonl from step 2
        output_path: GCS path to output JSONL file
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

    # Build SQLite database from entities file (memory-efficient)
    global _ENTITIES_DB_PATH
    _ENTITIES_DB_PATH = build_entities_db(entities_file, fs)

    # Get entity IDs set for fast filtering (just IDs, not the full entities)
    entity_ids = get_entity_ids_from_db(_ENTITIES_DB_PATH)
    logger.info(f"Loaded {len(entity_ids)} entity IDs for filtering")

    logger.info("=" * 80)
    logger.info("Starting Relation Batch File Generation")
    logger.info("=" * 80)
    logger.info(f"Wiki: {wiki}")
    logger.info(f"Model: {MODEL_NAME}, Reasoning: {REASONING_EFFORT}")
    logger.info(f"Workers: {num_workers} parallel processes")
    logger.info(f"Entities file: {entities_file}")
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
    skipped = 0
    batch_size = num_workers * 10  # Process articles in batches

    # Accumulate articles in batches for parallel processing
    article_batch = []

    try:
        with Pool(processes=num_workers) as pool:
            for article in fw:
                if limit and count >= limit:
                    break

                # Only process articles that have entities
                if str(article["id"]) in entity_ids:
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

        # Write all requests to GCS
        write_batch_file(all_requests, output_path, fs)
        logger.info(
            f"Done. Generated {len(all_requests)} requests from {count} articles."
        )
        logger.info(f"Skipped {skipped} articles without entities.")
        logger.info("=" * 80)
    finally:
        # Clean up temp database
        if _ENTITIES_DB_PATH and os.path.exists(_ENTITIES_DB_PATH):
            os.remove(_ENTITIES_DB_PATH)
            logger.info("Cleaned up temporary entities database")

    return all_requests


def main():
    parser = argparse.ArgumentParser(
        description="Generate Batch API files for relation extraction from entities"
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
    parser.add_argument(
        "--entities-file",
        type=str,
        default=None,
        help="GCS path to parsed_entities.jsonl (defaults to GCP_KG_PREFIX/{wiki}/entities/parsed_entities.jsonl)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (defaults to CPU count)",
    )
    args = parser.parse_args()

    # Generate GCS paths
    entities_file = (
        args.entities_file
        or f"{GCP_KG_PREFIX}/{args.wiki}/entities/parsed_entities.jsonl"
    )
    output_path = f"{GCP_KG_PREFIX}/{args.wiki}/relations/batch.jsonl"

    logger.info(f"Script: {__file__}")
    logger.info(f"Arguments: {vars(args)}")

    generate_relations_batch_file(
        entities_file=entities_file,
        output_path=output_path,
        wiki=args.wiki,
        limit=args.limit,
        force=args.force,
        num_workers=args.workers or os.cpu_count(),
    )


if __name__ == "__main__":
    main()
