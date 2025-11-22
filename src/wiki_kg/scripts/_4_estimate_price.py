import json
from pathlib import Path


def main():
    # Pricing for gpt-5-nano
    input_price_per_1m = 0.025  # $0.025 per 1M input tokens
    output_price_per_1m = 0.20  # $0.20 per 1M output tokens

    # Load bucket distribution
    bucket_file = Path("analysis/fineweb_bucket_distribution.json")
    with open(bucket_file, "r") as f:
        bucket_data = json.load(f)

    print("=" * 80)
    print("GPT-5-NANO COST ESTIMATION")
    print("=" * 80)
    print(f"Input:  ${input_price_per_1m} per 1M tokens")
    print(f"Output: ${output_price_per_1m} per 1M tokens")
    print("=" * 80)

    total_articles = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for bucket, data in bucket_data.items():
        article_count = data["article_count"]
        avg_input = data["avg_prompt_tokens"]
        avg_output = data["avg_completion_tokens"]

        bucket_input_tokens = article_count * avg_input
        bucket_output_tokens = article_count * avg_output

        total_articles += article_count
        total_input_tokens += bucket_input_tokens
        total_output_tokens += bucket_output_tokens

    # Calculate costs
    input_cost = (total_input_tokens / 1_000_000) * input_price_per_1m
    output_cost = (total_output_tokens / 1_000_000) * output_price_per_1m
    total_cost = input_cost + output_cost

    print(f"\nTotal Articles: {total_articles:,}")
    print(f"\nTotal Input Tokens:  {total_input_tokens:>20,}")
    print(f"Total Output Tokens: {total_output_tokens:>20,}")
    print(f"Total Tokens:        {total_input_tokens + total_output_tokens:>20,}")
    print(f"\nInput Cost:  ${input_cost:>15,.2f}")
    print(f"Output Cost: ${output_cost:>15,.2f}")
    print("-" * 80)
    print(f"TOTAL COST:  ${total_cost:>15,.2f}")
    print("=" * 80)

    # Per article cost
    cost_per_article = total_cost / total_articles
    print(f"\nCost per article: ${cost_per_article:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

