"""
Prompt optimization using DSPy GEPA.

Optimizes entity extraction prompts for a cheaper model
to match the output quality of a more expensive model (ground truth).

Usage:
    python entities.py gpt-5.1 high gpt-5-nano minimal 100
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Annotated, Optional

import dspy
import gcsfs
import typer
from dotenv import load_dotenv
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from kg_gen.steps._1_get_entities import TextEntities
import mlflow




try:
    from ..utils import GCP_KG_PREFIX, build_subdir
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils import GCP_KG_PREFIX, build_subdir

load_dotenv()

# Logging
HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / "generation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = typer.Typer()


# --- DSPy Module ---


class EntityExtractor(dspy.Module):
    """DSPy module for entity extraction."""

    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(TextEntities)

    def forward(self, source_text: str) -> list[str]:
        result = self.extract(source_text=source_text)
        return result


# --- Metric ---


def entities_f1(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Compute F1 score for entity extraction with feedback for GEPA optimization.

    Returns a ScoreWithFeedback object as expected by GEPA.
    """
    gold = example.entities

    # Handle parsing errors
    if not hasattr(pred, "entities"):
        return ScoreWithFeedback(
            score=0.0,
            feedback="PARSING ERROR: Output missing 'entities' field. Must return a list of entity strings.",
        )

    predicted = pred.entities

    if predicted is None:
        return ScoreWithFeedback(
            score=0.0,
            feedback="PARSING ERROR: 'entities' is null. Must return a list of entity strings.",
        )

    if not isinstance(predicted, list):
        return ScoreWithFeedback(
            score=0.0,
            feedback=f"PARSING ERROR: 'entities' is {type(predicted).__name__}, expected list.",
        )

    # Check for non-string items
    invalid_items = [e for e in predicted if not isinstance(e, str)]
    if invalid_items:
        return ScoreWithFeedback(
            score=0.0,
            feedback=f"PARSING ERROR: 'entities' contains non-string items: {invalid_items[:3]}",
        )

    # Normalize for comparison (lowercase, stripped)
    gold_normalized = {e.lower().strip(): e for e in gold}
    pred_normalized = {e.lower().strip(): e for e in predicted}

    gold_set = set(gold_normalized.keys())
    pred_set = set(pred_normalized.keys())

    # Calculate sets
    true_positives = gold_set & pred_set
    false_negatives = gold_set - pred_set
    false_positives = pred_set - gold_set

    # Edge cases
    if not gold_set and not pred_set:
        return ScoreWithFeedback(
            score=1.0, feedback="Correct: no entities expected, none extracted."
        )

    if not pred_set:
        missing = [gold_normalized[e] for e in false_negatives]
        return ScoreWithFeedback(
            score=0.0, feedback=f"No entities extracted. MISSING: {missing}"
        )

    if not gold_set:
        extra = [pred_normalized[e] for e in false_positives]
        return ScoreWithFeedback(
            score=0.0,
            feedback=f"Extracted {len(extra)} entities when none expected. EXTRA: {extra}",
        )

    # Calculate F1
    tp = len(true_positives)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Build concise feedback
    parts = [f"F1={f1:.2f} P={precision:.2f} R={recall:.2f}"]

    if true_positives:
        correct = [gold_normalized[e] for e in true_positives]
        parts.append(f"CORRECT: {correct}")

    if false_negatives:
        missing = [gold_normalized[e] for e in false_negatives]
        parts.append(f"MISSING: {missing}")

    if false_positives:
        extra = [pred_normalized[e] for e in false_positives]
        parts.append(f"EXTRA: {extra}")

    return ScoreWithFeedback(score=f1, feedback=" | ".join(parts))


# --- Data Loading ---


def load_ground_truth(
    wiki: str, model: str, reasoning_effort: str, limit: Optional[int]
) -> list[dict]:
    fs = gcsfs.GCSFileSystem()

    """Load parsed ground truth data from GCS."""
    subdir = build_subdir(model, reasoning_effort, limit)
    path = f"{GCP_KG_PREFIX}/{wiki}/entities/{subdir}/parsed.jsonl"

    logger.info(f"Loading ground truth from: {path}")

    if not fs.exists(path):
        logger.error(f"Ground truth file not found: {path}")
        return []

    data = []
    with fs.open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} samples from {model}/{reasoning_effort}")
    return data


def load_articles(article_ids: set[str], max_scan: int) -> dict[str, str]:
    """Load article texts from HuggingFace dataset."""
    from datasets import load_dataset

    logger.info(f"Loading articles for {len(article_ids)} IDs...")
    dataset = load_dataset(
        "josancamon/finewiki", name="default", split="en", streaming=True
    )

    articles = {}
    for i, article in enumerate(dataset):
        if i >= max_scan:
            break
        aid = str(article["id"])
        if aid in article_ids:
            articles[aid] = article.get("text", "")
        if len(articles) >= len(article_ids):
            break

    logger.info(f"Loaded {len(articles)} articles")
    return articles


def build_dataset(gt_data: list[dict], articles: dict[str, str]) -> list[dspy.Example]:
    """Build DSPy dataset from ground truth and articles."""
    dataset = []

    for item in gt_data:
        aid = item["custom_id"]
        if aid not in articles or not articles[aid]:
            continue

        example = dspy.Example(
            source_text=articles[aid], entities=item["entities"]
        ).with_inputs("source_text")
        dataset.append(example)

    logger.info(f"Built dataset with {len(dataset)} examples")
    return dataset


# --- Main ---


@app.command()
def main(
    gt_model: Annotated[str, typer.Argument(help="Ground truth model")],
    gt_reasoning: Annotated[str, typer.Argument(help="Ground truth reasoning effort")],
    opt_model: Annotated[str, typer.Argument(help="Model to optimize")],
    opt_reasoning: Annotated[str, typer.Argument(help="Target reasoning effort")],
    limit: Annotated[int, typer.Argument(help="Number of samples")],
    wiki: Annotated[str, typer.Option(help="Wiki identifier")] = "enwiki",
    train_ratio: Annotated[float, typer.Option(help="Train/val split")] = 0.75,
    auto: Annotated[
        str, typer.Option(help="GEPA auto budget: light, medium, heavy")
    ] = "light",
):
    """
    Optimize entity extraction prompts using DSPy GEPA.

    Example:
        python entities.py gpt-5.1 high gpt-5-nano minimal 100
    """
    logger.info("=" * 80)
    logger.info("DSPy GEPA Optimization - Entity Extraction")
    logger.info(f"Ground truth: {gt_model} ({gt_reasoning})")
    logger.info(f"Optimize for: {opt_model} ({opt_reasoning})")
    logger.info(f"Auto budget: {auto}")
    logger.info("=" * 80)

    threads = 64
    gt_data = load_ground_truth(wiki, gt_model, gt_reasoning, limit)
    assert len(gt_data) > 0, "gt_data empty"

    article_ids = set(item["custom_id"] for item in gt_data)
    articles = load_articles(article_ids, limit * 10)

    all_data = build_dataset(gt_data, articles)
    assert len(all_data) >= 5, "Insufficient data"

    all_data = all_data[:50]
    # Split data
    random.seed(42)
    random.shuffle(all_data)
    split = int(len(all_data) * train_ratio)
    trainset, valset = all_data[:split], all_data[split:]
    logger.info(f"Train: {len(trainset)}, Val: {len(valset)}")

    # Configure LMs
    api_key = os.getenv("OPENAI_API_KEY")
    student_lm = dspy.LM(
        f"openai/{opt_model}",
        temperature=1.0,
        api_key=api_key,
        model_type="responses",
        reasoning={"effort": "minimal"},
    )
    reflection_lm = dspy.LM(
        f"openai/{gt_model}",
        temperature=1.0,
        api_key=api_key,
        model_type="responses",
        # reasoning={"effort": "medium"},
    )

    dspy.configure(lm=student_lm)

    mlflow.dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces_from_compile=True,
        log_traces_from_eval=True,
    )
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("gepa-optimize-4")

    # Create program
    program = EntityExtractor()

    logger.info("Evaluating baseline...")
    evaluator = dspy.Evaluate(
        devset=valset, metric=entities_f1, display_progress=True, num_threads=threads
    )
    baseline = evaluator(program)
    logger.info(f"Baseline F1: {baseline.score:.4f}")
    # return
    # return

    # Optimize with GEPA
    logger.info(f"Running GEPA optimization (auto={auto})...")
    optimizer = dspy.GEPA(
        metric=entities_f1,
        auto=auto,
        reflection_lm=reflection_lm,
        num_threads=threads,
        track_stats=True,
        log_dir=HERE / "gepa_logs" / f"{opt_model}_{opt_reasoning}",
        # kwargs gepa
        # track_best_outputs=True,
        # display_progress_bar=True,
        # use_mlflow=True,
        # gepa_kwargs={
        #     "mlflow_tracking_uri": "http://127.0.0.1:5000",
        #     "mlflow_experiment_name": "gepa-optimize-2",
        # },
    )
    optimized = optimizer.compile(program, trainset=trainset, valset=valset)

    # Evaluate optimized
    logger.info("Evaluating optimized program...")
    optimized_score = evaluator(optimized)
    logger.info(f"Optimized F1: {optimized_score.score:.4f}")
    logger.info(f"Improvement: {optimized_score.score - baseline.score:+.4f}")

    # Save
    out_dir = HERE / "optimized"
    out_dir.mkdir(exist_ok=True)
    program_path = out_dir / f"entities_{opt_model}_{opt_reasoning}.json"
    optimized.save(str(program_path))
    logger.info(f"Saved: {program_path}")

    logger.info("=" * 80)
    logger.info(f"Done! {baseline.score:.4f} -> {optimized_score.score:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
