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

from kg_gen.steps._1_get_entities import TextEntities
import mlflow


# class TextEntities(dspy.Signature):
#     """
#     Your task is to extract the key entities from the wikipedia article contents. The extracted entities are subjects or objects.
#     The entities selected will be used to create a knowledge graph. Please be concise and accurate.

#     The entities should:
#     1. Have a stable identity
#     2. Be possible to referenc independently

#     """

#     source_text: str = dspy.InputField(desc="Wikipedia article contents")
#     entities: list[str] = dspy.OutputField(desc="List of key entities", required=True)


mlflow.dspy.autolog(
    log_compiles=True,  # Track optimization process
    log_evals=True,  # Track evaluation results
    log_traces_from_compile=True,  # Track program traces during optimization
)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("gepa-optimize")


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
    Compute F1 score for entity extraction.

    GEPA requires metrics to accept 5 arguments.
    """
    gold = example.entities
    predicted = pred.entities if hasattr(pred, "entities") else []

    if predicted is None:
        predicted = []

    gold_set = set(e.lower().strip() for e in gold)
    pred_set = set(e.lower().strip() for e in predicted)

    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0

    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gold_set) if gold_set else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


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

    threads = 100
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
        reasoning={"effort": "medium"},
    )

    dspy.configure(lm=student_lm)

    # Create program
    program = EntityExtractor()

    logger.info("Evaluating baseline...")
    evaluator = dspy.Evaluate(
        devset=valset, metric=entities_f1, display_progress=True, num_threads=threads
    )
    baseline = evaluator(program)
    logger.info(f"Baseline F1: {baseline.score:.4f}")
    # return

    # Optimize with GEPA
    logger.info(f"Running GEPA optimization (auto={auto})...")
    optimizer = dspy.GEPA(
        metric=entities_f1, auto=auto, reflection_lm=reflection_lm, num_threads=threads
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
