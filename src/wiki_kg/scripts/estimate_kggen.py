"""
Estimate KGGen performance on FineWiki articles.

This script:
1. Selects 10 articles from FineWiki with text length around 6200*4 characters (±200)
2. Processes each article with KGGen in parallel (no chunking)
3. Compares both deduplication methods: SEMHASH (no tokens) vs FULL (uses tokens)
4. Tracks detailed timing for extraction, semhash dedup, and full dedup
5. Tracks token usage for extraction and full dedup
6. Saves results to analysis/kggen_estimates/articles/{id}.json
"""

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from dotenv import load_dotenv
from kg_gen import KGGen
from kg_gen.steps._3_deduplicate import DeduplicateMethod

# Load environment variables
load_dotenv()

# Configuration
TARGET_LENGTH = 6200 * 4  # ~24,800 characters, mean tokens
LENGTH_TOLERANCE = 200  # +- 50 tokens from mean
NUM_ARTICLES = 50  # estimate with n articles
MAX_CONCURRENT = 32  # Maximum number of articles to process in parallel
OUTPUT_DIR = Path("analysis/kggen_estimates/articles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_suitable_articles(fw) -> List[Dict[str, Any]]:
    """Find articles with text length close to target_length."""
    suitable_articles = []
    min_length = TARGET_LENGTH - LENGTH_TOLERANCE
    max_length = TARGET_LENGTH + LENGTH_TOLERANCE

    print(
        f"Looking for {NUM_ARTICLES} articles with length between {min_length} and {max_length} characters..."
    )

    for article in fw:
        text_length = len(article["text"])

        if min_length <= text_length <= max_length:
            suitable_articles.append(
                {
                    "id": article["id"],
                    "title": article.get("title", "Unknown"),
                    "text": article["text"],
                    "length": text_length,
                }
            )
            print(
                f"  Found: {article.get('title', 'Unknown')} (ID: {article['id']}, Length: {text_length})"
            )

            if len(suitable_articles) >= NUM_ARTICLES:
                break

    print(f"\nFound {len(suitable_articles)} suitable articles")
    return suitable_articles


def convert_sets_to_lists(obj: Any) -> Any:
    """Recursively convert all sets to lists for JSON serialization."""
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_sets_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    else:
        return obj


def extract_token_usage_from_history(lm, start_idx: int = 0) -> Dict[str, int]:
    """Extract token usage from dspy LM history starting from a specific index."""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for entry in lm.history[start_idx:]:
        if isinstance(entry, dict):
            # Check for usage information in various possible locations
            usage = entry.get("usage") or entry.get("response", {}).get("usage")

            if usage:
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)

    return {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }


def process_single_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single article without chunking, comparing both dedup methods."""
    # Create fresh KGGen instance for this article
    kg = KGGen(
        model="openai/gpt-5-nano",
        temperature=1.0,
        reasoning_effort="medium",
        api_key=os.getenv("OPENAI_API_KEY"),
        # retrieval_model="all-MiniLM-L6-v2",
        disable_cache=True,
    )

    article_id = article["id"]
    text = article["text"]  # TODO: consider using title, and some metadata

    # === STAGE 1: EXTRACTION ===
    start_extraction = time.time()
    kg.lm.history = []
    graph_no_cluster = kg.generate(input_data=text, chunk_size=None, cluster=False)

    graph_no_cluster.to_file("graph.json")
    extraction_time = time.time() - start_extraction
    extraction_tokens = extract_token_usage_from_history(kg.lm, 0)

    entities_before = len(graph_no_cluster.entities)
    relations_before = len(graph_no_cluster.relations)

    # === STAGE 2: SEMHASH DEDUPLICATION (no tokens) ===
    start_semhash = time.time()
    graph_semhash = kg.deduplicate(
        graph=graph_no_cluster, method=DeduplicateMethod.SEMHASH
    )
    semhash_time = time.time() - start_semhash

    total_time = extraction_time + semhash_time

    # Calculate cleanup percentages
    semhash_entity_cleanup = (
        (1 - len(graph_semhash.entities) / entities_before) * 100
        if entities_before > 0
        else 0
    )
    semhash_relation_cleanup = (
        (1 - len(graph_semhash.relations) / relations_before) * 100
        if relations_before > 0
        else 0
    )

    # Prepare result
    result = {
        "article_id": article_id,
        "article_title": article["title"],
        "article_length": article["length"],
        "timing": {
            "extraction_seconds": extraction_time,
            "semhash_dedup_seconds": semhash_time,
            "total_seconds": total_time,
        },
        "extraction": {
            "entities": entities_before,
            "relations": relations_before,
            "tokens": extraction_tokens,
        },
        "semhash_dedup": {
            "entities": len(graph_semhash.entities),
            "relations": len(graph_semhash.relations),
            "entity_cleanup_percent": semhash_entity_cleanup,
            "relation_cleanup_percent": semhash_relation_cleanup,
            "entity_clusters": len(graph_semhash.entity_clusters)
            if graph_semhash.entity_clusters
            else 0,
            "edge_clusters": len(graph_semhash.edge_clusters)
            if graph_semhash.edge_clusters
            else 0,
            "tokens": 0,
        },
        # TODO: full dedup is so fucking slow, that this doesn't make sense even for 1 article,
        # "full_dedup": {
        #     "entities": len(graph_full.entities),
        #     "relations": len(graph_full.relations),
        #     "entity_cleanup_percent": full_entity_cleanup,
        #     "relation_cleanup_percent": full_relation_cleanup,
        #     "entity_clusters": len(graph_full.entity_clusters)
        #     if graph_full.entity_clusters
        #     else 0,
        #     "edge_clusters": len(graph_full.edge_clusters)
        #     if graph_full.edge_clusters
        #     else 0,
        #     "tokens": full_tokens,
        # },
    }

    return result


async def process_article_async(
    article: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    article_num: int,
    total: int,
) -> Dict[str, Any]:
    """Process a single article with semaphore control."""
    article_id = str(article["id"]).split("/")[-1]
    output_file = OUTPUT_DIR / f"{article_id}.json"

    async with semaphore:
        # Check if already processed
        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                output_file.unlink()

        # Process in thread pool (since KGGen is synchronous)
        print(f"[{article_num}/{total}] Processing: {article['title'][:50]}...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, process_single_article, article)

        # Save result
        result = convert_sets_to_lists(result)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        print(
            f"[{article_num}/{total}] ✓ {article['title'][:50]} - {result['timing']['total_seconds']:.1f}s"
        )
        return result


async def main_async():
    """Main execution function with parallel processing."""
    print("=" * 80)
    print("KGGen Estimation Script - Parallel Processing")
    print("=" * 80)
    fw = load_dataset(
        "josancamon/finewiki",
        name="default",
        split="en",
        streaming=True,
    )

    articles = find_suitable_articles(fw)
    if len(articles) < NUM_ARTICLES:
        print(f"\n⚠️  Warning: Only found {len(articles)} articles matching criteria")

    # TODO: sentence transformer fails when using multiple threads
    # process_single_article(articles[0])
    # return

    # Process articles in parallel with semaphore
    print(
        f"\n🚀 Processing {len(articles)} articles in parallel (max {MAX_CONCURRENT} concurrent)...\n"
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)

    start_time = time.time()
    tasks = [
        process_article_async(article, semaphore, executor, i, len(articles))
        for i, article in enumerate(articles, 1)
    ]
    all_results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    executor.shutdown(wait=True)

    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT - DEDUPLICATION COMPARISON")
    print("=" * 80)

    n = len(all_results)

    # Extraction stats
    avg_extraction_time = (
        sum(r["timing"]["extraction_seconds"] for r in all_results) / n
    )
    avg_entities_extracted = sum(r["extraction"]["entities"] for r in all_results) / n
    avg_relations_extracted = sum(r["extraction"]["relations"] for r in all_results) / n
    avg_extraction_tokens = (
        sum(r["extraction"]["tokens"]["total_tokens"] for r in all_results) / n
    )

    # SEMHASH stats
    avg_semhash_time = (
        sum(r["timing"]["semhash_dedup_seconds"] for r in all_results) / n
    )
    avg_semhash_entities = sum(r["semhash_dedup"]["entities"] for r in all_results) / n
    avg_semhash_relations = (
        sum(r["semhash_dedup"]["relations"] for r in all_results) / n
    )
    avg_semhash_entity_cleanup = (
        sum(r["semhash_dedup"]["entity_cleanup_percent"] for r in all_results) / n
    )
    avg_semhash_relation_cleanup = (
        sum(r["semhash_dedup"]["relation_cleanup_percent"] for r in all_results) / n
    )
    avg_semhash_tokens = (
        sum(r["semhash_dedup"]["tokens"]["total_tokens"] for r in all_results) / n
    )

    # FULL stats
    avg_full_time = sum(r["timing"]["full_dedup_seconds"] for r in all_results) / n
    avg_full_entities = sum(r["full_dedup"]["entities"] for r in all_results) / n
    avg_full_relations = sum(r["full_dedup"]["relations"] for r in all_results) / n
    avg_full_entity_cleanup = (
        sum(r["full_dedup"]["entity_cleanup_percent"] for r in all_results) / n
    )
    avg_full_relation_cleanup = (
        sum(r["full_dedup"]["relation_cleanup_percent"] for r in all_results) / n
    )
    avg_full_tokens = (
        sum(r["full_dedup"]["tokens"]["total_tokens"] for r in all_results) / n
    )

    stats = {
        "total_wall_time_seconds": total_time,
        "extraction": {
            "avg_time_seconds": avg_extraction_time,
            "avg_entities": avg_entities_extracted,
            "avg_relations": avg_relations_extracted,
            "avg_tokens": avg_extraction_tokens,
        },
        "semhash_dedup": {
            "avg_time_seconds": avg_semhash_time,
            "avg_entities": avg_semhash_entities,
            "avg_relations": avg_semhash_relations,
            "avg_entity_cleanup_percent": avg_semhash_entity_cleanup,
            "avg_relation_cleanup_percent": avg_semhash_relation_cleanup,
            "avg_tokens": avg_semhash_tokens,
        },
        "full_dedup": {
            "avg_time_seconds": avg_full_time,
            "avg_entities": avg_full_entities,
            "avg_relations": avg_full_relations,
            "avg_entity_cleanup_percent": avg_full_entity_cleanup,
            "avg_relation_cleanup_percent": avg_full_relation_cleanup,
            "avg_tokens": avg_full_tokens,
        },
        "comparison": {
            "time_difference_seconds": avg_full_time - avg_semhash_time,
            "time_difference_percent": (
                (avg_full_time - avg_semhash_time) / avg_semhash_time * 100
            )
            if avg_semhash_time > 0
            else 0,
            "entity_cleanup_difference_percent": avg_full_entity_cleanup
            - avg_semhash_entity_cleanup,
            "relation_cleanup_difference_percent": avg_full_relation_cleanup
            - avg_semhash_relation_cleanup,
            "token_difference": avg_full_tokens - avg_semhash_tokens,
        },
    }

    summary = {
        "config": {
            "num_articles": NUM_ARTICLES,
            "max_concurrent": MAX_CONCURRENT,
            "target_length": TARGET_LENGTH,
            "length_tolerance": LENGTH_TOLERANCE,
        },
        "total_articles_processed": len(articles),
        "statistics": stats,
        "results": all_results,
    }

    # TODO: re run extraction for snapshot 357, 86

    # Save summary
    summary_file = OUTPUT_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved to {summary_file}")


def main():
    """Entry point that runs the async main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
