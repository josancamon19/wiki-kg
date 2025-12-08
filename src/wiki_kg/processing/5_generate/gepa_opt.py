"""
Prompt optimization using GEPA.

Optimizes entity/relation extraction prompts for a cheaper model
to match the output quality of a more expensive model (ground truth).

Usage:
    python gepa_opt.py gpt-5.1 high gpt-5-nano minimal 100
"""

import json
import logging
import re
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
    from .utils import GCP_KG_PREFIX, build_filename
except ImportError:
    from utils import GCP_KG_PREFIX, build_filename

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
            max_output_tokens=50000,
            temperature=1.0,
        )
        return response.output_text

    def _parse_entities(self, output: str) -> list[str]:
        """Parse entities from model output."""
        # Look for the entities section
        match = re.search(
            r"\[\[\s*##\s*entities\s*##\s*\]\]\s*(.+?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)",
            output,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Fallback: try to find any JSON array
        arrays = re.findall(r"\[.*?\]", output, re.DOTALL)
        for arr in arrays:
            try:
                parsed = json.loads(arr)
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    return parsed
            except json.JSONDecodeError:
                continue

        return []

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
                parsed = self._parse_entities(raw_output)
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
                        input_text=instance.source_text[:500],  # Truncate for reflection
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

    def _parse_relations(self, output: str) -> list[dict]:
        """Parse relations from model output."""
        match = re.search(
            r"\[\[\s*##\s*relations\s*##\s*\]\]\s*(.+?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)",
            output,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Fallback: find JSON arrays of objects
        arrays = re.findall(r"\[.*?\]", output, re.DOTALL)
        for arr in arrays:
            try:
                parsed = json.loads(arr)
                if (
                    isinstance(parsed, list)
                    and parsed
                    and isinstance(parsed[0], dict)
                    and "subject" in parsed[0]
                ):
                    return parsed
            except json.JSONDecodeError:
                continue

        return []

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
                parsed = self._parse_relations(raw_output)
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
    filename = build_filename("parsed", model, reasoning_effort, limit)
    path = f"{GCP_KG_PREFIX}/{wiki}/{task}/{filename}"

    logger.info(f"Loading ground truth from: {path}")

    data = []
    with fs.open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    logger.info(f"Loaded {len(data)} ground truth samples")
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
    limit: Annotated[int, typer.Argument(help="Number of samples")],
    wiki: Annotated[str, typer.Option(help="Wiki identifier")] = "enwiki",
    task: Annotated[str, typer.Option(help="Task: entities, relations, or both")] = "both",
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
        article_ids = set(item["custom_id"] for item in gt_data)
        articles = load_articles(article_ids, limit * 10)

        # Build dataset
        all_data = []
        for item in gt_data:
            aid = item["custom_id"]
            if aid in articles and articles[aid]:
                all_data.append(
                    DataInstance(
                        article_id=aid,
                        source_text=articles[aid],
                        ground_truth=item["entities"],
                    )
                )

        split = int(len(all_data) * train_ratio)
        trainset, valset = all_data[:split], all_data[split:]
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

    # --- RELATIONS ---
    if task in ("relations", "both"):
        logger.info("Optimizing RELATIONS prompt...")

        gt_relations = load_ground_truth(
            wiki, gt_model, gt_reasoning, limit, "relations", fs
        )
        gt_entities = load_ground_truth(
            wiki, gt_model, gt_reasoning, limit, "entities", fs
        )
        entities_map = {item["custom_id"]: item["entities"] for item in gt_entities}

        article_ids = set(item["custom_id"] for item in gt_relations)
        articles = load_articles(article_ids, limit * 10)

        all_data = []
        for item in gt_relations:
            aid = item["custom_id"]
            if aid in articles and articles[aid] and aid in entities_map:
                all_data.append(
                    DataInstance(
                        article_id=aid,
                        source_text=articles[aid],
                        ground_truth=item["relations"],
                        entities=entities_map[aid],
                    )
                )

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

    logger.info("=" * 80)
    logger.info("Optimization complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
