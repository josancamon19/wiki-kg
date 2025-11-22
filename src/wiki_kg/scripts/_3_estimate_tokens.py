from datasets import load_dataset
from multiprocessing import Pool, cpu_count
import numpy as np
from pathlib import Path
import json


def load_token_estimates():
    """Load the token estimates from estimates.json."""
    estimates_path = (
        Path(__file__).parent.parent.parent.parent
        / "analysis"
        / "kggen_estimates"
        / "estimates.json"
    )
    with open(estimates_path, "r") as f:
        estimates = json.load(f)

    # Convert bucket keys to integers and sort
    bucket_map = {int(k): v for k, v in estimates.items()}
    sorted_buckets = sorted(bucket_map.keys())

    return bucket_map, sorted_buckets


def find_closest_bucket(char_count, sorted_buckets):
    """Find the closest bucket for a given character count."""
    if char_count <= sorted_buckets[0]:
        return sorted_buckets[0]
    if char_count >= sorted_buckets[-1]:
        return sorted_buckets[-1]

    # Find closest bucket
    closest = min(sorted_buckets, key=lambda x: abs(x - char_count))
    return closest


def process_batch(args):
    """Process a batch of texts and return bucket assignments."""
    texts, sorted_buckets = args
    buckets = np.array(
        [find_closest_bucket(len(text), sorted_buckets) for text in texts],
        dtype=np.int32,
    )
    return buckets


def batch_iterator(dataset, batch_size, max_items, sorted_buckets):
    """Efficiently iterate over dataset in batches, extracting only text."""
    batch = []
    for i, row in enumerate(dataset):
        if i >= max_items:
            break
        batch.append(row.get("text", ""))
        if len(batch) >= batch_size:
            yield (batch, sorted_buckets)
            batch = []
    if batch:
        yield (batch, sorted_buckets)


def main():
    print("Loading token estimates...")
    bucket_map, sorted_buckets = load_token_estimates()

    print("Loading dataset...")
    fw = load_dataset(
        "HuggingFaceFW/finewiki", name="en", split="train", streaming=True
    )

    # Number of articles to process
    num_articles = int(1e10)
    batch_size = 10000  # Larger batches for better efficiency

    print("Processing articles...")

    # Process batches
    all_buckets = []
    processed = 0

    with Pool(cpu_count()) as pool:
        batch_gen = batch_iterator(fw, batch_size, num_articles, sorted_buckets)

        for buckets in pool.imap(process_batch, batch_gen, chunksize=1):
            all_buckets.append(buckets)
            processed += len(buckets)

            if processed % 100000 == 0:
                print(f"  {processed:,} articles processed")

    # Concatenate results
    all_buckets = np.concatenate(all_buckets)

    # Count articles per bucket
    bucket_counts = {}
    for bucket in sorted_buckets:
        bucket_counts[bucket] = int(np.sum(all_buckets == bucket))

    total_articles = len(all_buckets)

    # Save bucket distribution
    bucket_distribution = {
        str(bucket): {
            "article_count": count,
            "avg_prompt_tokens": bucket_map[bucket]["avg_prompt_tokens"],
            "avg_completion_tokens": bucket_map[bucket]["avg_completion_tokens"],
        }
        for bucket, count in sorted(bucket_counts.items())
    }

    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "fineweb_bucket_distribution.json"
    with open(output_file, "w") as f:
        json.dump(bucket_distribution, f, indent=2)

    print(f"\nDone. Processed {total_articles:,} articles")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
