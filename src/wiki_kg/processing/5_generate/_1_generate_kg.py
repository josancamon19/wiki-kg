import asyncio
import json
import os
import logging
import argparse
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gcsfs
from datasets import load_dataset
from dotenv import load_dotenv
from kg_gen import KGGen
from kg_gen.steps._3_deduplicate import DeduplicateMethod

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
GCP_OUTPUT_PATH = "gs://wikipedia-graph/knowledge_graphs"
MODEL_NAME = "openai/gpt-5-nano"
REASONING_EFFORT = "minimal"
MAX_TOKENS = 100000
CHUNK_THRESHOLD = int(2e5)  # Characters
CHUNK_SIZE = int(1e5)  #
MAX_CONCURRENT = 1  # Adjust based on rate limits and capacity

# TODO: Log token usage for each one?


def get_gcs_fs():
    """Initialize GCS filesystem."""
    return gcsfs.GCSFileSystem()


def article_to_graph_dict(graph) -> Dict[str, Any]:
    """Convert KGGen graph to dictionary format for export."""
    return {
        "entities": list(graph.entities),
        "relations": list(graph.relations),
        "edges": list(graph.edges),
        "entity_clusters": {k: list(v) for k, v in graph.entity_clusters.items()}
        if graph.entity_clusters
        else None,
        "edge_clusters": {k: list(v) for k, v in graph.edge_clusters.items()}
        if graph.edge_clusters
        else None,
    }


def process_article_sync(article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Synchronous function to process a single article with KGGen.
    To be run in a ThreadPoolExecutor.
    """
    try:
        text = article.get("text", "")
        if not text:
            return None

        # Initialize KGGen
        kg = KGGen(
            model=MODEL_NAME,
            reasoning_effort=REASONING_EFFORT,
            max_tokens=MAX_TOKENS,
            temperature=1.0,
        )

        # Determine chunk size
        chunk_size = None
        if len(text) > CHUNK_THRESHOLD:
            chunk_size = CHUNK_SIZE
            logger.info(f"Chunking article {article.get('id')} (length {len(text)})")

        # Generate KG
        graph = kg.generate(
            input_data=text,
            chunk_size=chunk_size,
            deduplication_method=DeduplicateMethod.SEMHASH,
        )

        return article_to_graph_dict(graph)

    except Exception as e:
        logger.error(f"Error processing article {article.get('id')}: {e}")
        return None


async def process_article_task(
    article: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    fs: gcsfs.GCSFileSystem,
):
    """Async task to process a single article."""
    article_id_raw = str(article["id"])
    # Sanitize ID for filename (take last part if it's a path/url)
    file_id = article_id_raw.split("/")[-1]
    output_file = f"{GCP_OUTPUT_PATH}/{file_id}.json"

    async with semaphore:
        # Check if exists (blocking call wrapped or simple enough to be quick)
        # fs.exists is synchronous but usually fast. For high concurrency, could offload.
        try:
            if fs.exists(output_file):
                logger.info(f"Skipping existing: {file_id}")
                return
        except Exception as e:
            logger.warning(f"Error checking existence for {file_id}: {e}")

        logger.info(f"Processing: {file_id} ({len(article.get('text', ''))} chars)")

        # Run CPU/Network bound KGGen in thread pool
        loop = asyncio.get_event_loop()
        result_dict = await loop.run_in_executor(
            executor, process_article_sync, article
        )

        if result_dict:
            # Write to GCS
            try:
                # GCS write is synchronous in gcsfs, can be blocking.
                # For optimal async, could run in executor, but writing small JSONs is fast.
                # Let's run in executor to be safe.
                await loop.run_in_executor(
                    executor, lambda: _write_to_gcs(fs, output_file, result_dict)
                )
                logger.info(f"Saved: {file_id}")
            except Exception as e:
                logger.error(f"Error writing to GCS for {file_id}: {e}")


def _write_to_gcs(fs, path, data):
    with fs.open(path, "w") as f:
        json.dump(data, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(
        description="Generate Knowledge Graphs from FineWiki"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of articles to process"
    )
    parser.add_argument(
        "--concurrency", type=int, default=MAX_CONCURRENT, help="Max concurrent tasks"
    )
    args = parser.parse_args()

    logger.info("Starting KG Generation...")
    logger.info(f"Model: {MODEL_NAME}, Reasoning: {REASONING_EFFORT}")
    logger.info(f"Output: {GCP_OUTPUT_PATH}")

    # Initialize GCS FS
    fs = get_gcs_fs()
    try:
        fs.makedirs(GCP_OUTPUT_PATH, exist_ok=True)
    except Exception:
        pass  # Ignore if already exists

    # Load dataset
    logger.info("Loading dataset josancamon/finewiki...")
    fw = load_dataset(
        "josancamon/finewiki",
        name="default",
        split="en",
        streaming=True,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    executor = ThreadPoolExecutor(max_workers=args.concurrency)

    running_tasks = set()
    MAX_PENDING_TASKS = args.concurrency * 3  # Buffer for pending tasks

    # Create tasks
    count = 0
    for article in fw:
        if args.limit and count >= args.limit:
            break

        task = asyncio.create_task(
            process_article_task(article, semaphore, executor, fs)
        )
        running_tasks.add(task)
        # We use add_done_callback to remove from set, but we need to be careful with async sets.
        # simpler to just manually manage the set in the loop with asyncio.wait

        count += 1

        if len(running_tasks) >= MAX_PENDING_TASKS:
            # Wait for at least one task to complete
            done, pending = await asyncio.wait(
                running_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            running_tasks = pending

            if count % 100 == 0:
                logger.info(f"Dispatched {count} articles...")

    if running_tasks:
        logger.info(f"Waiting for remaining {len(running_tasks)} tasks...")
        await asyncio.wait(running_tasks)

    executor.shutdown(wait=True)
    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
