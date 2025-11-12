from datasets import load_dataset
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns


def process_batch(texts):
    """Process a batch of texts and return character counts as numpy array."""
    return np.array([len(text) for text in texts], dtype=np.int32)


def batch_iterator(dataset, batch_size, max_items):
    """Efficiently iterate over dataset in batches, extracting only text."""
    batch = []
    for i, row in enumerate(dataset):
        if i >= max_items:
            break
        batch.append(row.get("text", ""))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    print("Loading dataset...")
    fw = load_dataset(
        "HuggingFaceFW/finewiki", name="en", split="train", streaming=True
    )

    # Number of articles to process (adjust as needed)
    num_articles = int(1e10)  # 1e4
    batch_size = 1000  # Larger batches = less overhead

    print(f"Processing up to {num_articles:,} articles in parallel...")

    # Process batches on-the-fly with multiprocessing
    char_counts = []
    processed = 0

    with Pool(cpu_count()) as pool:
        batch_gen = batch_iterator(fw, batch_size, num_articles)

        # Process batches as they come with chunksize for better performance
        for batch_results in pool.imap(process_batch, batch_gen, chunksize=10):
            char_counts.append(batch_results)
            processed += len(batch_results)

            if processed % 10000 == 0:
                print(f"Processed {processed:,} articles...")

    # Concatenate all results into single numpy array
    char_counts = np.concatenate(char_counts)

    # Calculate statistics
    mean_chars = np.mean(char_counts)
    median_chars = np.median(char_counts)
    std_chars = np.std(char_counts)
    p25 = np.percentile(char_counts, 25)
    p75 = np.percentile(char_counts, 75)
    p95 = np.percentile(char_counts, 95)
    p99 = np.percentile(char_counts, 99)
    total_chars = np.sum(char_counts)
    total_tokens = total_chars / 4  # Approximate tokens

    # Create output directory
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 11

    # Create simplified visualization
    plt.figure(figsize=(14, 6))

    # 1. Zoomed view (up to 95th percentile)
    ax1 = plt.subplot(1, 2, 1)
    filtered_counts = [c for c in char_counts if c <= p95]
    ax1.hist(filtered_counts, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.axvline(
        median_chars,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_chars:,.0f}",
    )
    ax1.axvline(
        mean_chars,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_chars:,.0f}",
    )
    ax1.set_xlabel("Number of Characters", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Character Distribution (up to 95th percentile: {p95:,.0f})",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Statistics text box
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis("off")
    stats_text = f"""
    STATISTICAL SUMMARY
    {"=" * 35}
    
    Total Articles:     {len(char_counts):>15,}
    Total Characters:   {total_chars:>15,}
    Total Tokens (≈):   {total_tokens:>15,.0f}
    
    Per Article Stats:
      • Mean chars:     {mean_chars:>15,.0f}
      • Median chars:   {median_chars:>15,.0f}
      • Std Dev:        {std_chars:>15,.0f}
      • Min chars:      {np.min(char_counts):>15,}
      • Max chars:      {np.max(char_counts):>15,}
    
    Percentiles:
      • 25th:           {p25:>15,.0f}
      • 50th:           {median_chars:>15,.0f}
      • 75th:           {p75:>15,.0f}
      • 95th:           {p95:>15,.0f}
      • 99th:           {p99:>15,.0f}
    
    IQR:                {p75 - p25:>15,.0f}
    """
    ax2.text(
        0.05,
        0.5,
        stats_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.suptitle(
        "FineWeb Article Character Count Analysis",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    # Save to analysis directory
    output_file = output_dir / "finewiki.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to '{output_file}'")


if __name__ == "__main__":
    main()
