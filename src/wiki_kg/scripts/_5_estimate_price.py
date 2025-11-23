import json
from pathlib import Path


def main():
    # Pricing for models (per 1M tokens)
    # Note: Only models with estimate files will be processed
    model_pricing = {
        "gpt-5-nano-medium": {"input": 0.025, "output": 0.20},
        "gpt-5-nano-minimal": {"input": 0.025, "output": 0.20},
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

    # Load bucket distribution (article counts per bucket)
    bucket_file = Path("analysis/fineweb_bucket_distribution.json")
    with open(bucket_file, "r") as f:
        article_counts = json.load(f)

    estimates_dir = Path("analysis/kggen_estimates")

    # Find available estimate files
    available_models = []
    for model_name in model_pricing.keys():
        estimate_file = estimates_dir / f"{model_name}-estimates.json"
        if estimate_file.exists():
            available_models.append(model_name)

    if not available_models:
        print("❌ No estimate files found in analysis/kggen_estimates/")
        print("   Run _2_kggen_samples.py first to generate estimates.")
        return

    print("=" * 80)
    print("MODEL COST ESTIMATION")
    print("=" * 80)
    print(f"Available models with estimates: {', '.join(available_models)}")
    print("=" * 80)

    # Process each available model
    for model_name in available_models:
        estimate_file = estimates_dir / f"{model_name}-estimates.json"

        # Load token estimates for this model
        with open(estimate_file, "r") as f:
            token_estimates = json.load(f)

        # Calculate total tokens for this model
        total_articles = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for bucket, article_count in article_counts.items():
            # Get token estimates for this bucket
            if bucket in token_estimates:
                avg_input = token_estimates[bucket]["avg_prompt_tokens"]
                avg_output = token_estimates[bucket]["avg_completion_tokens"]

                bucket_input_tokens = article_count * avg_input
                bucket_output_tokens = article_count * avg_output

                total_articles += article_count
                total_input_tokens += bucket_input_tokens
                total_output_tokens += bucket_output_tokens

        # Get pricing for this model
        pricing = model_pricing[model_name]
        input_price_per_1m = pricing["input"]
        output_price_per_1m = pricing["output"]

        # Calculate costs
        input_cost = (total_input_tokens / 1_000_000) * input_price_per_1m
        output_cost = (total_output_tokens / 1_000_000) * output_price_per_1m
        total_cost = input_cost + output_cost
        cost_per_article = total_cost / total_articles if total_articles > 0 else 0

        # Display results
        print(f"\n{model_name.upper()}")
        print(f"  Estimate file: {estimate_file.name}")
        print(
            f"  Pricing: ${input_price_per_1m} / ${output_price_per_1m} per 1M tokens (input/output)"
        )
        print(f"  Total Articles:      {total_articles:>20,}")
        print(f"  Total Input Tokens:  {total_input_tokens:>20,.0f}")
        print(f"  Total Output Tokens: {total_output_tokens:>20,.0f}")
        print(
            f"  Total Tokens:        {total_input_tokens + total_output_tokens:>20,.0f}"
        )
        print(f"  Input Cost:          ${input_cost:>15,.2f}")
        print(f"  Output Cost:         ${output_cost:>15,.2f}")
        print(f"  Total Cost:          ${total_cost:>15,.2f}")
        print(f"  Cost Per Article:    ${cost_per_article:.6f}")
        print("-" * 80)


if __name__ == "__main__":
    main()
