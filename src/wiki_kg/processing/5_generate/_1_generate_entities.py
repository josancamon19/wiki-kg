import json
import os
import logging
import argparse
from typing import Dict, Any, Optional, List, Iterator
from multiprocessing import Pool, cpu_count
from pathlib import Path

import gcsfs
from datasets import load_dataset, DownloadConfig
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
WRITE_BUFFER_SIZE = 512  # Number of requests to hold before flushing to GCS

# Cache for prompt template
with open(os.path.join(PROMPTS_PATH, "entities.txt"), "r") as f:
    _ENTITY_PROMPT_TEMPLATE = f.read()


def _article_iterator(dataset, limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    """Yield at most ``limit`` articles from the dataset."""
    if limit is None:
        for article in dataset:
            yield article
    else:
        for idx, article in enumerate(dataset):
            if idx >= limit:
                break
            yield article


def _load_finewiki_dataset(
    *, local_dataset: bool, hf_cache_dir: Optional[str]
):
    """Load the FineWiki dataset either via streaming or from the local cache."""
    dataset_kwargs = dict(name="default", split="en")
    if hf_cache_dir:
        dataset_kwargs["cache_dir"] = hf_cache_dir

    if local_dataset:
        logger.info(
            "Loading dataset josancamon/finewiki from local cache%s...",
            f" ({hf_cache_dir})" if hf_cache_dir else "",
        )
        download_config = DownloadConfig(local_files_only=True)
        try:
            return load_dataset(
                "josancamon/finewiki",
                streaming=False,
                download_config=download_config,
                **dataset_kwargs,
            )
        except FileNotFoundError as exc:
            logger.error(
                "Dataset not found in the local Hugging Face cache. "
                "Download it first (e.g. with `datasets-cli download`) or "
                "omit --local-dataset."
            )
            raise FileNotFoundError(
                "FineWiki dataset missing from the local Hugging Face cache"
            ) from exc
    else:
        logger.info("Loading dataset josancamon/finewiki via streaming...")
        return load_dataset(
            "josancamon/finewiki",
            streaming=True,
            **dataset_kwargs,
        )


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

def generate_entities_batch_file(
    output_path: str,
    wiki: str = "enwiki",
    limit: Optional[int] = None,
    force: bool = False,
    num_workers: Optional[int] = None,
    local_dataset: bool = False,
    hf_cache_dir: Optional[str] = None,
):
    """
    Generate a batch JSONL file containing entity extraction requests and push it to GCS.

    Args:
        output_path: Destination path on GCS.
        wiki: Wiki identifier used only for logging/output path generation.
        limit: Maximum number of articles to process.
        force: Regenerate even if the file already exists.
        num_workers: Number of worker processes (defaults to cpu_count).
        local_dataset: If True, read FineWiki from the local Hugging Face cache
            (dataset must be downloaded beforehand).
        hf_cache_dir: Optional Hugging Face cache directory to use when
            ``local_dataset`` is True.
    Returns:
        Dict[str, int]: Counts of processed articles and generated requests.
    """
    fs = gcsfs.GCSFileSystem()

    if fs.exists(output_path) and not force:
        logger.info(f"Batch file already exists: {output_path}")
        logger.info("Use --force to regenerate")
        return {"articles": 0, "requests": 0}

    if num_workers is None:
        num_workers = cpu_count()
    num_workers = max(1, num_workers)

    logger.info("=" * 80)
    logger.info("Starting Entity Batch File Generation")
    logger.info("=" * 80)
    logger.info(f"Wiki: {wiki}")
    logger.info(f"Model: {MODEL_NAME}, Reasoning: {REASONING_EFFORT}")
    logger.info(f"Workers: {num_workers} parallel processes")
    logger.info(f"Dataset mode: {'local-cache' if local_dataset else 'streaming'}")
    if local_dataset and hf_cache_dir:
        logger.info(f"HF cache dir: {hf_cache_dir}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Limit: {limit if limit else 'None (all articles)'}")

    dataset = _load_finewiki_dataset(
        local_dataset=local_dataset, hf_cache_dir=hf_cache_dir
    )
    article_iter = _article_iterator(dataset, limit)

    # Prepare output file on GCS
    output_dir = "/".join(output_path.replace("gs://", "").split("/")[:-1])
    fs.makedirs(output_dir, exist_ok=True)

    # Heuristic chunk/log settings tuned for streaming vs local data
    worker_chunk_size = 1 if not local_dataset else min(64, num_workers * 4)
    log_every_articles = max(50, num_workers * 20)

    total_articles = 0
    total_requests = 0
    write_buffer: List[str] = []

    def flush_buffer(file_obj):
        if write_buffer:
            file_obj.write("\n".join(write_buffer) + "\n")
            write_buffer.clear()

    def consume_batches(request_batches, file_obj):
        nonlocal total_articles, total_requests
        for requests in request_batches:
            total_articles += 1
            if not requests:
                continue
            for request in requests:
                write_buffer.append(json.dumps(request))
                total_requests += 1
                if len(write_buffer) >= WRITE_BUFFER_SIZE:
                    flush_buffer(file_obj)
            if total_articles % log_every_articles == 0:
                logger.info(
                    "Processed %d articles, %d total requests...",
                    total_articles,
                    total_requests,
                )

    with fs.open(output_path, "w") as output_file:
        if num_workers == 1:
            request_iter = (
                article_to_batch_requests(article) for article in article_iter
            )
            consume_batches(request_iter, output_file)
        else:
            with Pool(processes=num_workers) as pool:
                request_iter = pool.imap_unordered(
                    article_to_batch_requests,
                    article_iter,
                    chunksize=worker_chunk_size,
                )
                consume_batches(request_iter, output_file)
        flush_buffer(output_file)

    logger.info(
        f"Done. Generated {total_requests} requests from {total_articles} articles."
    )
    logger.info("=" * 80)
    return {"articles": total_articles, "requests": total_requests}


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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--local-dataset",
        action="store_true",
        help=(
            "Load FineWiki from the local Hugging Face cache instead of streaming. "
            "Requires downloading the dataset beforehand."
        ),
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory to use with --local-dataset",
    )
    args = parser.parse_args()

    # Generate GCS path for output
    output_path = f"{GCP_KG_PREFIX}/{args.wiki}/entities/batch.jsonl"

    logger.info(f"Script: {__file__}")
    logger.info(f"Arguments: {vars(args)}")

    generate_entities_batch_file(
        output_path=output_path,
        limit=args.limit,
        force=args.force,
        wiki=args.wiki,
        num_workers=args.workers,
        local_dataset=args.local_dataset,
        hf_cache_dir=args.hf_cache_dir,
    )


if __name__ == "__main__":
    main()
