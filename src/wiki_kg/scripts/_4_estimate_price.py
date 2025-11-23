import json
from pathlib import Path


def main():
    # Pricing for GPT-5 models (per 1M tokens)
    models = {
        "gpt-5-nano": {
            "input": 0.025,  # $0.025 per 1M input tokens
            "output": 0.20,  # $0.20 per 1M output tokens
        },
        "gpt-5-mini": {
            "input": 0.125,  # $0.125 per 1M input tokens
            "output": 1.0,  # $1.0 per 1M output tokens
        },
        "gpt-5": {
            "input": 0.625,  # $0.625 per 1M input tokens
            "output": 5.0,  # $5.0 per 1M output tokens
        },
        "gpt-oss-20b-together": {
            "input": 0.05, 
            "output": 0.2, 
        },
        "gpt-oss-20b-deepinfra": {
            "input": 0.03, 
            "output": 0.14, 
        },
    }

    # Load bucket distribution
    bucket_file = Path("analysis/fineweb_bucket_distribution.json")
    with open(bucket_file, "r") as f:
        bucket_data = json.load(f)

    # Calculate total tokens
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

    # Print header
    print("=" * 80)
    print("GPT-5 MODELS COST ESTIMATION")
    print("=" * 80)
    print(f"Total Articles:      {total_articles:>20,}")
    print(f"Total Input Tokens:  {total_input_tokens:>20,}")
    print(f"Total Output Tokens: {total_output_tokens:>20,}")
    print(f"Total Tokens:        {total_input_tokens + total_output_tokens:>20,}")
    print("=" * 80)

    # Calculate and display costs for each model
    for model_name, pricing in models.items():
        input_price_per_1m = pricing["input"]
        output_price_per_1m = pricing["output"]

        input_cost = (total_input_tokens / 1_000_000) * input_price_per_1m
        output_cost = (total_output_tokens / 1_000_000) * output_price_per_1m
        total_cost = input_cost + output_cost
        cost_per_article = total_cost / total_articles

        print(f"\n{model_name.upper()}")
        print(
            f"  Pricing: ${input_price_per_1m} / ${output_price_per_1m} per 1M tokens"
        )
        print(f"  Input Cost:  ${input_cost:>15,.2f}")
        print(f"  Output Cost: ${output_cost:>15,.2f}")
        print(f"  Total Cost:  ${total_cost:>15,.2f}")
        print(f"  Per Article: ${cost_per_article:.6f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
