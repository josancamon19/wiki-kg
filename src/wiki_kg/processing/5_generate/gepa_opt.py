"""
Prompt optimization using GEPA.

Optimizes entity/relation extraction prompts for a cheaper model
to match the output quality of a more expensive model (ground truth).

Usage:
    python gepa_opt.py gpt-5.1 high gpt-5-nano minimal 100
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Optional

import gcsfs
import typer
from dotenv import load_dotenv
from openai import OpenAI

import gepa
from gepa import EvaluationBatch, GEPAAdapter

try:
    from .utils import (
        GCP_KG_PREFIX,
        build_subdir,
        extract_entities_from_text,
        extract_relations_from_text,
    )
except ImportError:
    from utils import (
        GCP_KG_PREFIX,
        build_subdir,
        extract_entities_from_text,
        extract_relations_from_text,
    )

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

# Load baseline prompts
PROMPTS_PATH = HERE / "prompts"
with open(PROMPTS_PATH / "entities.txt", "r") as f:
    ENTITIES_PROMPT_TEMPLATE = f.read()
with open(PROMPTS_PATH / "relations.txt", "r") as f:
    RELATIONS_PROMPT_TEMPLATE = f.read()

app = typer.Typer()


# --- Data Types ---


@dataclass
class DataInstance:
    """Single example for optimization."""

    article_id: str
    source_text: str
    ground_truth: Any  # entities list or relations list
    entities: list[str] | None = None  # For relations task


@dataclass
class Trajectory:
    """Execution trace for reflection."""

    input_text: str
    prompt_used: str
    raw_output: str
    parsed_output: Any
    ground_truth: Any
    score: float
    error: str | None = None


# --- Metrics ---


def compute_entities_f1(pred: list[str], gold: list[str]) -> float:
    """Compute F1 score for entity extraction."""
    gold_set = set(e.lower().strip() for e in gold)
    pred_set = set(e.lower().strip() for e in (pred or []))

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


def compute_relations_f1(pred: list[dict], gold: list[dict]) -> float:
    """Compute F1 score for relation extraction."""

    def normalize(r):
        if isinstance(r, dict):
            return (
                r.get("subject", "").lower().strip(),
                r.get("predicate", "").lower().strip(),
                r.get("object", "").lower().strip(),
            )
        return tuple(str(x).lower().strip() for x in r[:3])

    gold_set = set(normalize(r) for r in gold)
    pred_set = set(normalize(r) for r in (pred or []))

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


# --- GEPA Adapter for Entities ---


class EntitiesAdapter(GEPAAdapter[DataInstance, Trajectory, list[str]]):
    """GEPA adapter for entity extraction optimization."""

    def __init__(self, model: str, reasoning_effort: str):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI()

    def _call_model(self, prompt: str) -> str:
        """Call the model with the given prompt."""
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=100000,
            temperature=1.0,
        )
        return response.output_text

    def evaluate(
        self,
        batch: list[DataInstance],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, list[str]]:
        """Evaluate candidate prompt on batch."""
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        prompt_template = candidate["prompt"]

        for instance in batch:
            # Build prompt
            prompt = prompt_template.replace("{_source_text_}", instance.source_text)

            try:
                raw_output = self._call_model(prompt)
                parsed = extract_entities_from_text(raw_output) or []
                score = compute_entities_f1(parsed, instance.ground_truth)
                error = None
            except Exception as e:
                raw_output = ""
                parsed = []
                score = 0.0
                error = str(e)

            outputs.append(parsed)
            scores.append(score)

            if capture_traces:
                trajectories.append(
                    Trajectory(
                        input_text=instance.source_text[:500],
                        prompt_used=prompt[:1000],
                        raw_output=raw_output[:1000],
                        parsed_output=parsed,
                        ground_truth=instance.ground_truth,
                        score=score,
                        error=error,
                    )
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, list[str]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build reflective dataset for prompt improvement."""
        dataset = []

        for traj in eval_batch.trajectories or []:
            # Focus on failures for reflection
            if traj.score < 0.8:
                gold_set = set(e.lower() for e in traj.ground_truth)
                pred_set = set(e.lower() for e in traj.parsed_output)
                missed = gold_set - pred_set
                extra = pred_set - gold_set

                feedback_parts = [f"Score: {traj.score:.2f}"]
                if missed:
                    feedback_parts.append(f"Missed entities: {list(missed)[:5]}")
                if extra:
                    feedback_parts.append(f"Wrong entities: {list(extra)[:5]}")
                if traj.error:
                    feedback_parts.append(f"Error: {traj.error}")

                dataset.append(
                    {
                        "Inputs": {"source_text": traj.input_text[:300]},
                        "Generated Outputs": {"entities": traj.parsed_output[:10]},
                        "Feedback": " | ".join(feedback_parts),
                    }
                )

        return {"prompt": dataset}


# --- GEPA Adapter for Relations ---


class RelationsAdapter(GEPAAdapter[DataInstance, Trajectory, list[dict]]):
    """GEPA adapter for relation extraction optimization."""

    def __init__(self, model: str, reasoning_effort: str):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI()

    def _call_model(self, prompt: str) -> str:
        """Call the model with the given prompt."""
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=50000,
            temperature=1.0,
        )
        return response.output_text

    def evaluate(
        self,
        batch: list[DataInstance],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, list[dict]]:
        """Evaluate candidate prompt on batch."""
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        prompt_template = candidate["prompt"]

        for instance in batch:
            # Build prompt
            prompt = prompt_template.replace("{_source_text_}", instance.source_text)
            prompt = prompt.replace("{_entities_}", json.dumps(instance.entities or []))

            try:
                raw_output = self._call_model(prompt)
                parsed = extract_relations_from_text(raw_output) or []
                score = compute_relations_f1(parsed, instance.ground_truth)
                error = None
            except Exception as e:
                raw_output = ""
                parsed = []
                score = 0.0
                error = str(e)

            outputs.append(parsed)
            scores.append(score)

            if capture_traces:
                trajectories.append(
                    Trajectory(
                        input_text=instance.source_text[:500],
                        prompt_used=prompt[:1000],
                        raw_output=raw_output[:1000],
                        parsed_output=parsed,
                        ground_truth=instance.ground_truth,
                        score=score,
                        error=error,
                    )
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Trajectory, list[dict]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build reflective dataset for prompt improvement."""
        dataset = []

        def rel_key(r):
            return (
                r.get("subject", "").lower(),
                r.get("predicate", "").lower(),
                r.get("object", "").lower(),
            )

        for traj in eval_batch.trajectories or []:
            if traj.score < 0.8:
                gold_set = set(rel_key(r) for r in traj.ground_truth)
                pred_set = set(rel_key(r) for r in traj.parsed_output)
                missed = gold_set - pred_set
                extra = pred_set - gold_set

                feedback_parts = [f"Score: {traj.score:.2f}"]
                if missed:
                    feedback_parts.append(f"Missed relations: {list(missed)[:3]}")
                if extra:
                    feedback_parts.append(f"Wrong relations: {list(extra)[:3]}")
                if traj.error:
                    feedback_parts.append(f"Error: {traj.error}")

                dataset.append(
                    {
                        "Inputs": {"source_text": traj.input_text[:300]},
                        "Generated Outputs": {"relations": traj.parsed_output[:5]},
                        "Feedback": " | ".join(feedback_parts),
                    }
                )

        return {"prompt": dataset}


# --- Data Loading ---


def load_ground_truth(
    wiki: str,
    model: str,
    reasoning_effort: str,
    limit: Optional[int],
    task: str,
    fs: gcsfs.GCSFileSystem,
) -> list[dict]:
    """Load parsed ground truth data from GCS."""
    subdir = build_subdir(model, reasoning_effort, limit)
    path = f"{GCP_KG_PREFIX}/{wiki}/{task}/{subdir}/parsed.jsonl"

    logger.info(f"Loading {task} ground truth from: {path}")

    if not fs.exists(path):
        logger.error(f"Ground truth file not found: {path}")
        return []

    data = []
    with fs.open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} {task} samples from {model}/{reasoning_effort}")
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


# --- Main ---


@app.command()
def main(
    gt_model: Annotated[str, typer.Argument(help="Ground truth model")],
    gt_reasoning: Annotated[str, typer.Argument(help="Ground truth reasoning effort")],
    opt_model: Annotated[str, typer.Argument(help="Model to optimize")],
    opt_reasoning: Annotated[str, typer.Argument(help="Target reasoning effort")],
    limit: Annotated[int, typer.Argument(help="Number of samples")],  # 100 optimizer
    wiki: Annotated[str, typer.Option(help="Wiki identifier")] = "enwiki",
    task: Annotated[
        str, typer.Option(help="Task: entities, relations, or both")
    ] = "both",
    max_iterations: Annotated[int, typer.Option(help="Max GEPA iterations")] = 10,
    train_ratio: Annotated[float, typer.Option(help="Train/val split")] = 0.8,
):
    """
    Optimize prompts using GEPA.

    Example:
        python gepa_opt.py gpt-5.1 high gpt-5-nano minimal 100
    """
    fs = gcsfs.GCSFileSystem()

    logger.info("=" * 80)
    logger.info("GEPA Prompt Optimization")
    logger.info(f"Ground truth: {gt_model} ({gt_reasoning})")
    logger.info(f"Optimize for: {opt_model} ({opt_reasoning})")
    logger.info(f"Limit: {limit}, Task: {task}")
    logger.info("=" * 80)

    # --- ENTITIES ---
    if task in ("entities", "both"):
        logger.info("Optimizing ENTITIES prompt...")

        gt_data = load_ground_truth(wiki, gt_model, gt_reasoning, limit, "entities", fs)
        if not gt_data:
            logger.error("No ground truth entities data found. Skipping entities optimization.")
        else:
            article_ids = set(item["custom_id"] for item in gt_data)
            logger.info(f"Ground truth entities: {len(gt_data)} samples, {len(article_ids)} unique article IDs")
            
            articles = load_articles(article_ids, limit * 10)
            logger.info(f"Articles loaded: {len(articles)} / {len(article_ids)} requested")

            # Build dataset
            all_data = []
            missing_articles = 0
            empty_articles = 0
            for item in gt_data:
                aid = item["custom_id"]
                if aid not in articles:
                    missing_articles += 1
                    continue
                if not articles[aid]:
                    empty_articles += 1
                    continue
                all_data.append(
                    DataInstance(
                        article_id=aid,
                        source_text=articles[aid],
                        ground_truth=item["entities"],
                    )
                )

            logger.info(f"Dataset stats: {len(all_data)} matched, {missing_articles} missing articles, {empty_articles} empty articles")
            
            split = int(len(all_data)/10 * train_ratio)
            trainset, valset = all_data[:split], all_data[split:10]
            logger.info(f"Train: {len(trainset)}, Val: {len(valset)}")

            if trainset and valset:
                adapter = EntitiesAdapter(opt_model, opt_reasoning)
                seed = {"prompt": ENTITIES_PROMPT_TEMPLATE}

                result = gepa.optimize(
                    seed_candidate=seed,
                    trainset=trainset,
                    valset=valset,
                    adapter=adapter,
                    reflection_lm=gt_model,
                    max_metric_calls=max_iterations * len(trainset),
                    reflection_minibatch_size=min(3, len(trainset)),
                    display_progress_bar=True,
                )

                # Save optimized prompt
                out_path = PROMPTS_PATH / f"entities_opt_{opt_model}_{opt_reasoning}.txt"
                with open(out_path, "w") as f:
                    f.write(result.best_candidate["prompt"])
                logger.info(f"Saved: {out_path}")
                logger.info(f"Best score: {result.best_score:.4f}")
            else:
                logger.warning("Insufficient data for entities optimization (need both train and val sets)")

    # --- RELATIONS ---
    if task in ("relations", "both"):
        logger.info("Optimizing RELATIONS prompt...")

        gt_relations = load_ground_truth(
            wiki, gt_model, gt_reasoning, limit, "relations", fs
        )
        gt_entities = load_ground_truth(
            wiki, gt_model, gt_reasoning, limit, "entities", fs
        )
        
        if not gt_relations:
            logger.error("No ground truth relations data found. Skipping relations optimization.")
        elif not gt_entities:
            logger.error("No ground truth entities data found (needed for relations). Skipping relations optimization.")
        else:
            entities_map = {item["custom_id"]: item["entities"] for item in gt_entities}
            logger.info(f"Ground truth relations: {len(gt_relations)} samples")
            logger.info(f"Ground truth entities: {len(gt_entities)} samples ({len(entities_map)} unique IDs)")

            article_ids = set(item["custom_id"] for item in gt_relations)
            articles = load_articles(article_ids, limit * 10)
            logger.info(f"Articles loaded: {len(articles)} / {len(article_ids)} requested")

            all_data = []
            missing_articles = 0
            empty_articles = 0
            missing_entities = 0
            for item in gt_relations:
                aid = item["custom_id"]
                if aid not in articles:
                    missing_articles += 1
                    continue
                if not articles[aid]:
                    empty_articles += 1
                    continue
                if aid not in entities_map:
                    missing_entities += 1
                    continue
                all_data.append(
                    DataInstance(
                        article_id=aid,
                        source_text=articles[aid],
                        ground_truth=item["relations"],
                        entities=entities_map[aid],
                    )
                )

            logger.info(f"Dataset stats: {len(all_data)} matched, {missing_articles} missing articles, {empty_articles} empty articles, {missing_entities} missing entities")

            split = int(len(all_data) * train_ratio)
            trainset, valset = all_data[:split], all_data[split:]
            logger.info(f"Train: {len(trainset)}, Val: {len(valset)}")

            if trainset and valset:
                adapter = RelationsAdapter(opt_model, opt_reasoning)
                seed = {"prompt": RELATIONS_PROMPT_TEMPLATE}

                result = gepa.optimize(
                    seed_candidate=seed,
                    trainset=trainset,
                    valset=valset,
                    adapter=adapter,
                    reflection_lm=gt_model,
                    max_metric_calls=max_iterations * len(trainset),
                    reflection_minibatch_size=min(3, len(trainset)),
                    display_progress_bar=True,
                )

                out_path = PROMPTS_PATH / f"relations_opt_{opt_model}_{opt_reasoning}.txt"
                with open(out_path, "w") as f:
                    f.write(result.best_candidate["prompt"])
                logger.info(f"Saved: {out_path}")
                logger.info(f"Best score: {result.best_score:.4f}")
            else:
                logger.warning("Insufficient data for relations optimization (need both train and val sets)")

    logger.info("=" * 80)
    logger.info("Optimization complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
