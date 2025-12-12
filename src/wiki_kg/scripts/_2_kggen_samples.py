"""
Estimate KGGen performance on FineWiki articles across different lengths.

This script:
1. Samples 2 articles every 500 chars up to 15k, then every 5k chars up to 100k
2. Processes each article with KGGen in parallel (no chunking)
3. Tracks detailed timing for extraction and deduplication
4. Tracks token usage (total, prompt, completion)
5. Saves results to analysis/kggen_estimates/{model}/{reasoning}_{id}.json
6. Generates summary statistics grouped by length buckets
"""

import os
import json
import time
import asyncio
import contextvars
import traceback
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import argparse
from datasets import load_dataset
from dotenv import load_dotenv
from kg_gen import KGGen
from kg_gen.steps._3_deduplicate import DeduplicateMethod

# Load environment variables
load_dotenv()


# Model configuration
class ModelName(str, Enum):
    """Supported models for knowledge graph generation."""

    GPT_5_NANO = "gpt-5-nano"
    GPT_OSS_20B = "gpt-oss-20b-together"
    GPT_OSS_20B_DEEPINFRA = "gpt-oss-20b-deepinfra"


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for OpenAI models."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Configuration
ARTICLES_PER_BUCKET = 2  # Number of articles to sample per length bucket
LENGTH_TOLERANCE = 50  # +- tolerance in characters for bucket matching
MAX_CONCURRENT = 50  # Maximum number of articles to process in parallel

# Define length buckets: every 500 chars up to 15k, then every 5k chars
LENGTH_BUCKETS = list(range(500, 15001, 500)) + list(range(20000, 100001, 5000))
# LENGTH_BUCKETS = list(range(500, 2001, 500))


# Model-specific configurations
MODEL_CONFIGS = {
    ModelName.GPT_5_NANO: {
        "model_name": "openai/gpt-5-nano",
        "base_url": None,  # Uses default OpenAI API
        "api_key_env": "OPENAI_API_KEY",
        "supports_reasoning": True,
    },
    ModelName.GPT_OSS_20B: {
        "model_name": "together_ai/openai/gpt-oss-20b",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "supports_reasoning": True,
    },
    ModelName.GPT_OSS_20B_DEEPINFRA: {
        "model_name": "deepinfra/openai/gpt-oss-20b",
        "base_url": "https://api.deepinfra.com/v1",
        "api_key_env": "DEEPINFRA_API_KEY",
        "supports_reasoning": True,
    },
}


def find_suitable_articles(fw) -> List[Dict[str, Any]]:
    """Find articles for each length bucket."""
    # Initialize buckets dictionary
    buckets = {bucket: [] for bucket in LENGTH_BUCKETS}
    total_needed = len(LENGTH_BUCKETS) * ARTICLES_PER_BUCKET

    print(
        f"Looking for {ARTICLES_PER_BUCKET} articles in each of {len(LENGTH_BUCKETS)} length buckets..."
    )
    print(f"Total target: {total_needed} articles\n")

    for article in fw:
        text_length = len(article["text"])

        # Find matching bucket
        for bucket in LENGTH_BUCKETS:
            if abs(text_length - bucket) <= LENGTH_TOLERANCE:
                if len(buckets[bucket]) < ARTICLES_PER_BUCKET:
                    article_data = {
                        "id": article["id"],
                        "title": article.get("title", "Unknown"),
                        "text": article["text"],
                        "length": text_length,
                        "bucket": bucket,
                    }
                    buckets[bucket].append(article_data)
                    print(
                        f"  [{bucket:6d} chars] Found: {article.get('title', 'Unknown')[:50]} (ID: {article['id']}, Length: {text_length})"
                    )
                    break

        # Check if all buckets are filled
        if all(len(articles) >= ARTICLES_PER_BUCKET for articles in buckets.values()):
            break

    # Flatten buckets into single list
    suitable_articles = []
    for bucket in LENGTH_BUCKETS:
        suitable_articles.extend(buckets[bucket])

    # Print summary
    filled_buckets = sum(
        1 for articles in buckets.values() if len(articles) >= ARTICLES_PER_BUCKET
    )
    print(f"\n✓ Filled {filled_buckets}/{len(LENGTH_BUCKETS)} buckets completely")
    print(f"✓ Found {len(suitable_articles)} total articles")

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


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: Any) -> None:
        """Add tokens from a LiteLLM/OpenAI usage payload."""
        if not usage:
            return

        def _get_any(obj: Any, keys: List[str]) -> Any:
            if isinstance(obj, dict):
                for k in keys:
                    if k in obj and obj[k] is not None:
                        return obj[k]
                return None
            for k in keys:
                val = getattr(obj, k, None)
                if val is not None:
                    return val
            return None

        # LiteLLM normalizes to OpenAI-ish names, but Responses API uses input/output.
        prompt = _get_any(usage, ["prompt_tokens", "input_tokens"])
        completion = _get_any(usage, ["completion_tokens", "output_tokens"])
        total = _get_any(usage, ["total_tokens"])

        prompt_i = int(prompt or 0)
        completion_i = int(completion or 0)
        total_i = int(total or (prompt_i + completion_i) or 0)

        self.prompt_tokens += prompt_i
        self.completion_tokens += completion_i
        self.total_tokens += total_i

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": int(self.total_tokens),
        }


_LITELLM_USAGE_ACCUM: contextvars.ContextVar[Optional[TokenUsage]] = (
    contextvars.ContextVar("litellm_usage_accum", default=None)
)
_LITELLM_PATCHED = False
_ORIG_LITELLM_RESPONSES = None
_ORIG_LITELLM_COMPLETION = None


def _install_litellm_token_tracker() -> None:
    """
    Monkeypatch LiteLLM call sites used by KGGen `no_dspy=True` so we can
    aggregate `response.usage` the same way we do for DSPy history.
    """
    global _LITELLM_PATCHED, _ORIG_LITELLM_RESPONSES, _ORIG_LITELLM_COMPLETION
    if _LITELLM_PATCHED:
        return

    import litellm  # imported lazily to keep script import time low

    _ORIG_LITELLM_RESPONSES = litellm.responses
    _ORIG_LITELLM_COMPLETION = getattr(litellm, "completion", None)

    def _record_usage(resp: Any) -> None:
        accum = _LITELLM_USAGE_ACCUM.get()
        if accum is None:
            return

        usage = getattr(resp, "usage", None)
        if usage is None and isinstance(resp, dict):
            usage = resp.get("usage")
        accum.add(usage)

    def wrapped_responses(*args, **kwargs):
        resp = _ORIG_LITELLM_RESPONSES(*args, **kwargs)
        _record_usage(resp)
        return resp

    litellm.responses = wrapped_responses

    # Some versions/paths may call `litellm.completion()` instead of `responses()`
    if _ORIG_LITELLM_COMPLETION is not None:

        def wrapped_completion(*args, **kwargs):
            resp = _ORIG_LITELLM_COMPLETION(*args, **kwargs)
            _record_usage(resp)
            return resp

        litellm.completion = wrapped_completion

    _LITELLM_PATCHED = True


@contextmanager
def track_litellm_tokens() -> TokenUsage:
    """
    Context manager to collect LiteLLM token usage for the current call.

    Note: this tracks calls made in the *current thread*. If KGGen internally
    spawns threads (e.g., due to automatic chunking), those calls may not be
    attributed to this accumulator.
    """
    _install_litellm_token_tracker()
    accum = TokenUsage()
    token = _LITELLM_USAGE_ACCUM.set(accum)
    try:
        yield accum
    finally:
        _LITELLM_USAGE_ACCUM.reset(token)


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


def process_single_article(
    article: Dict[str, Any],
    model_config: Dict[str, Any],
    reasoning_effort: Optional[str] = None,
    no_dspy: bool = False,
) -> Dict[str, Any]:
    """Process a single article without chunking, comparing both dedup methods."""
    # Get API key
    api_key = model_config.get("api_key_override") or os.getenv(
        model_config["api_key_env"]
    )

    # Build KGGen kwargs
    kg_kwargs = {
        "model": model_config["model_name"],
        "temperature": 1.0,
        "api_key": api_key,
        "disable_cache": False,
        "max_tokens": 64000,
    }

    # Add base_url if specified
    if model_config.get("base_url"):
        kg_kwargs["api_base"] = model_config["base_url"]

    # Add reasoning_effort if supported and specified
    if model_config.get("supports_reasoning") and reasoning_effort:
        kg_kwargs["reasoning_effort"] = reasoning_effort

    # Create fresh KGGen instance for this article
    kg = KGGen(**kg_kwargs)

    article_id = article["id"]
    text = article["text"]

    # === STAGE 1: EXTRACTION ===
    start_extraction = time.time()
    kg.lm.history = []
    if no_dspy:
        with track_litellm_tokens() as litellm_usage:
            graph_no_cluster = kg.generate(input_data=text, no_dspy=True)
        extraction_tokens = litellm_usage.to_dict()
    else:
        graph_no_cluster = kg.generate(input_data=text, no_dspy=False)
        extraction_tokens = extract_token_usage_from_history(kg.lm, 0)

    extraction_time = time.time() - start_extraction
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
        "status": "ok",
        "article_id": article_id,
        "article_title": article["title"],
        "article_length": article["length"],
        "bucket": article.get("bucket"),
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
    }

    return result


async def process_article_async(
    article: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    article_num: int,
    total: int,
    output_dir: Path,
    model_config: Dict[str, Any],
    reasoning_effort: Optional[str] = None,
    no_dspy: bool = False,
) -> Dict[str, Any]:
    """Process a single article with semaphore control."""
    article_id = str(article["id"]).split("/")[-1]
    output_file = output_dir / f"{article_id}.json"

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
        try:
            result = await loop.run_in_executor(
                executor,
                process_single_article,
                article,
                model_config,
                reasoning_effort,
                no_dspy,
            )
        except Exception as e:
            # Never crash the whole run due to a single bad article / provider hiccup.
            err_tb = "".join(
                traceback.format_exception(type(e), e, getattr(e, "__traceback__", None))
            )
            result = {
                "status": "failed",
                "article_id": article.get("id"),
                "article_title": article.get("title", "Unknown"),
                "article_length": article.get("length"),
                "bucket": article.get("bucket"),
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": err_tb,
                },
            }

        # Save result
        result = convert_sets_to_lists(result)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        if result.get("status") == "ok":
            print(
                f"[{article_num}/{total}] ✓ {article['title'][:50]} - {result['timing']['total_seconds']:.1f}s"
            )
        else:
            err = result.get("error", {})
            print(
                f"[{article_num}/{total}] ✗ {article['title'][:50]} - {err.get('type', 'Error')}: {err.get('message', '')}"
            )
        return result


async def main_async(
    model: ModelName,
    reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
    no_dspy: bool = False,
):
    """Main execution function with parallel processing."""
    # Get model configuration
    model_config = MODEL_CONFIGS[model]

    # Determine output directory based on model, reasoning, and no_dspy flag
    output_dir = Path(
        f"analysis/kggen_estimates/{model.value}-{reasoning_effort.value}"
        + ("-no-dspy" if no_dspy else "")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("KGGen Estimation Script - Parallel Processing")
    print("=" * 80)
    print(f"Model: {model_config['model_name']}")
    print(f"Reasoning Effort: {reasoning_effort.value}")
    print(f"No DSPy Mode: {no_dspy}")
    if model_config.get("base_url"):
        print(f"Base URL: {model_config['base_url']}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    fw = load_dataset(
        "josancamon/finewiki",
        name="default",
        split="en",
        streaming=True,
    )

    articles = find_suitable_articles(fw)
    expected_articles = len(LENGTH_BUCKETS) * ARTICLES_PER_BUCKET
    if len(articles) < expected_articles:
        print(
            f"\n⚠️  Warning: Only found {len(articles)}/{expected_articles} articles matching criteria"
        )

    # Process articles in parallel with semaphore
    print(
        f"\n🚀 Processing {len(articles)} articles in parallel (max {MAX_CONCURRENT} concurrent)...\n"
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)

    start_time = time.time()
    tasks = [
        process_article_async(
            article,
            semaphore,
            executor,
            i,
            len(articles),
            output_dir,
            model_config,
            reasoning_effort,
            no_dspy,
        )
        for i, article in enumerate(articles, 1)
    ]
    all_results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    executor.shutdown(wait=True)

    # Print completion summary
    print("\n" + "=" * 80)
    print("DATA COLLECTION COMPLETE")
    print("=" * 80)
    ok_count = sum(1 for r in all_results if isinstance(r, dict) and r.get("status") == "ok")
    fail_count = sum(
        1 for r in all_results if isinstance(r, dict) and r.get("status") == "failed"
    )
    print(
        f"✅ Processed {len(all_results)} articles in {total_time:.1f}s "
        f"({ok_count} ok, {fail_count} failed)"
    )
    print(f"✅ Results saved to {output_dir}")
    print("\n💡 Run compute_summary.py to generate statistics from collected data")

    return all_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate knowledge graphs from FineWiki articles for different models."
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=[e.value for e in ModelName],
        default=ModelName.GPT_5_NANO.value,
        help="Model to use for knowledge graph generation.",
    )

    parser.add_argument(
        "--reasoning-effort",
        "-r",
        type=str,
        choices=[e.value for e in ReasoningEffort],
        default=ReasoningEffort.MEDIUM.value,
        help="Reasoning effort level (only for models that support it, like gpt-5-nano).",
    )

    parser.add_argument(
        "--no-dspy",
        action="store_true",
        help="Use direct LiteLLM prompts instead of DSPy (saves results in a separate directory).",
    )

    args = parser.parse_args()

    # Convert string values back to enums
    model = ModelName(args.model)
    reasoning_effort = ReasoningEffort(args.reasoning_effort)
    asyncio.run(main_async(model, reasoning_effort, args.no_dspy))


if __name__ == "__main__":
    main()
