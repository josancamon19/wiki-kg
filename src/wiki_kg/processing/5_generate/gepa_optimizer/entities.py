"""
Prompt optimization for entity extraction using GEPA (no DSPy dependency).

Optimizes the system prompt used by the cheaper model so that its extracted
entities align with the ground-truth entities from a stronger model.
"""

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import gcsfs
import litellm
import typer
from dotenv import load_dotenv
from gepa.api import optimize as gepa_optimize
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from kg_gen.steps._1_get_entities import EntitiesResponse, _load_entities_prompt
from pydantic import ValidationError

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


# --- Data structures ---


@dataclass
class EntitiesSample:
    custom_id: str
    source_text: str
    entities: list[str]


@dataclass
class EntitiesRollout:
    predicted_entities: list[str]
    raw_response: str


@dataclass
class EntitiesTrajectory:
    custom_id: str
    source_text: str
    gold_entities: list[str]
    predicted_entities: list[str]
    feedback: str
    raw_response: str


# --- Helpers ---


def load_base_system_prompt() -> str:
    """
    Load the seed system prompt.

    Prefer the prompt used by kg_gen (site-packages/kg_gen/prompts/entities.txt).
    """
    try:
        return _load_entities_prompt()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Falling back to local prompt: %s", exc)
        fallback = Path(__file__).resolve().parent.parent / "prompts" / "entities.txt"
        return fallback.read_text()


def entities_f1(gold: list[str], predicted: list[str]) -> tuple[float, str]:
    """Compute F1 and concise feedback for entity extraction."""
    gold_normalized = {e.lower().strip(): e for e in gold}
    pred_normalized = {e.lower().strip(): e for e in predicted}

    gold_set = set(gold_normalized)
    pred_set = set(pred_normalized)

    if not gold_set and not pred_set:
        return 1.0, "Correct: no entities expected, none extracted."

    if not pred_set:
        missing = [gold_normalized[e] for e in gold_set]
        return 0.0, f"No entities extracted. MISSING: {missing}"

    if not gold_set:
        extra = [pred_normalized[e] for e in pred_set]
        return 0.0, f"Extracted entities when none expected. EXTRA: {extra}"

    tp = gold_set & pred_set
    fn = gold_set - pred_set
    fp = pred_set - gold_set

    precision = len(tp) / len(pred_set) if pred_set else 0.0
    recall = len(tp) / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    parts = [f"F1={f1:.2f} P={precision:.2f} R={recall:.2f}"]
    if tp:
        parts.append(f"CORRECT: {[gold_normalized[e] for e in tp]}")
    if fn:
        parts.append(f"MISSING: {[gold_normalized[e] for e in fn]}")
    if fp:
        parts.append(f"EXTRA: {[pred_normalized[e] for e in fp]}")
    return f1, " | ".join(parts)


# --- GEPA Adapter ---


class EntitiesAdapter(GEPAAdapter[EntitiesSample, EntitiesTrajectory, EntitiesRollout]):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_source_chars: int = 4000,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_source_chars = max_source_chars

        # Align litellm configuration with provided overrides
        if api_key:
            litellm.api_key = api_key
        if api_base:
            litellm.api_base = api_base

        schema = EntitiesResponse.model_json_schema()
        schema["additionalProperties"] = False
        self.schema = schema

    def _predict_entities(
        self, system_prompt: str, source_text: str
    ) -> tuple[list[str], str, Optional[str]]:
        user_prompt = (
            "Here is the text to extract entities from:\n\n"
            "<article>\n"
            f"{source_text}\n"
            "</article>\n"
        )

        kwargs = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "entities_response",
                    "schema": self.schema,
                    "strict": True,
                }
            },
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = litellm.responses(**kwargs)
        raw_text = response.output[-1].content[0].text
        try:
            parsed = EntitiesResponse.model_validate_json(raw_text)
            return parsed.entities, raw_text, None
        except ValidationError as exc:
            return [], raw_text, f"ValidationError: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            return [], raw_text, f"Unexpected parse error: {exc}"

    def evaluate(
        self,
        batch: list[EntitiesSample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[EntitiesTrajectory, EntitiesRollout]:
        outputs: list[EntitiesRollout] = []
        scores: list[float] = []
        trajectories: list[EntitiesTrajectory] | None = [] if capture_traces else None

        system_prompt = candidate["system_prompt"]

        for sample in batch:
            # Truncate very long articles to stay under provider limits

            predicted_entities, raw_response, parse_error = self._predict_entities(
                system_prompt=system_prompt, source_text=sample.source_text
            )

            if parse_error:
                feedback = f"PARSING ERROR: {parse_error}"
                score = 0.0
            else:
                score, feedback = entities_f1(sample.entities, predicted_entities)

            outputs.append(
                EntitiesRollout(
                    predicted_entities=predicted_entities,
                    raw_response=raw_response,
                )
            )
            scores.append(score)

            if capture_traces:
                trajectories.append(
                    EntitiesTrajectory(
                        custom_id=sample.custom_id,
                        source_text=sample.source_text[:500]
                        if len(sample.source_text) > 500
                        else sample.source_text,
                        gold_entities=sample.entities,
                        predicted_entities=predicted_entities,
                        feedback=feedback,
                        raw_response=raw_response,
                    )
                )

        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=trajectories
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[EntitiesTrajectory, EntitiesRollout],
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        assert eval_batch.trajectories is not None, (
            "Trajectories required for reflection."
        )

        reflective_data: dict[str, list[dict]] = {}
        for component in components_to_update:
            items = []
            for traj, score in zip(
                eval_batch.trajectories, eval_batch.scores, strict=False
            ):
                items.append(
                    {
                        "Inputs": {
                            "custom_id": traj.custom_id,
                            "source_text": traj.source_text,
                        },
                        "Generated Outputs": {
                            "entities": traj.predicted_entities,
                            "raw_response": traj.raw_response,
                        },
                        "Feedback": traj.feedback,
                        "Score": score,
                        "Gold": traj.gold_entities,
                    }
                )
            if not items:
                raise ValueError("No reflective dataset items generated.")
            reflective_data[component] = items

        return reflective_data


# --- Data Loading ---


def load_ground_truth(
    wiki: str, model: str, reasoning_effort: str, limit: Optional[int]
) -> list[dict]:
    fs = gcsfs.GCSFileSystem()

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


def build_dataset(
    gt_data: list[dict], articles: dict[str, str]
) -> list[EntitiesSample]:
    """Build dataset from ground truth and articles."""
    dataset: list[EntitiesSample] = []

    for item in gt_data:
        aid = item["custom_id"]
        if aid not in articles or not articles[aid]:
            continue

        dataset.append(
            EntitiesSample(
                custom_id=aid, source_text=articles[aid], entities=item["entities"]
            )
        )

    logger.info(f"Built dataset with {len(dataset)} examples")
    return dataset


def evaluate_candidate(
    adapter: EntitiesAdapter, candidate: dict[str, str], dataset: list[EntitiesSample]
) -> float:
    """Evaluate a candidate prompt on a dataset and return mean F1."""
    batch = adapter.evaluate(dataset, candidate, capture_traces=False)
    return sum(batch.scores) / len(batch.scores)


# --- Main ---


@app.command()
def main(
    gt_model: Annotated[str, typer.Argument(help="Ground truth / teacher model")],
    gt_reasoning: Annotated[str, typer.Argument(help="Ground truth reasoning effort")],
    opt_model: Annotated[str, typer.Argument(help="Model to optimize")],
    opt_reasoning: Annotated[str, typer.Argument(help="Target reasoning effort")],
    limit: Annotated[int, typer.Argument(help="Number of samples")],
    wiki: Annotated[str, typer.Option(help="Wiki identifier")] = "enwiki",
    train_ratio: Annotated[float, typer.Option(help="Train/val split")] = 0.75,
    auto: Annotated[str, typer.Option(help="Budget: light, medium, heavy")] = "light",
    max_metric_calls: Annotated[
        Optional[int], typer.Option(help="Override GEPA max_metric_calls")
    ] = None,
    mlflow_tracking_uri: Annotated[
        str, typer.Option(help="MLflow tracking URI")
    ] = "http://127.0.0.1:5000",
    mlflow_experiment: Annotated[
        str, typer.Option(help="MLflow experiment name")
    ] = "gepa-optimize-entities",
):
    """
    Optimize the entity extraction system prompt with GEPA.

    Example:
        python entities.py gpt-5.1 high gpt-5-nano minimal 100
    """
    logger.info("=" * 80)
    logger.info("GEPA Optimization - Entity Extraction (system prompt)")
    logger.info(f"Teacher: {gt_model} ({gt_reasoning})")
    logger.info(f"Student: {opt_model} ({opt_reasoning})")
    logger.info(f"Budget: {auto}")
    logger.info("=" * 80)

    gt_data = load_ground_truth(wiki, gt_model, gt_reasoning, limit)
    assert len(gt_data) > 0, "gt_data empty"

    article_ids = {item["custom_id"] for item in gt_data}
    articles = load_articles(article_ids, limit * 10)

    all_data = build_dataset(gt_data, articles)
    assert len(all_data) >= 5, "Insufficient data"

    all_data = all_data[:50]
    random.seed(42)
    random.shuffle(all_data)
    split = int(len(all_data) * train_ratio)
    trainset, valset = all_data[:split], all_data[split:]
    logger.info(f"Train: {len(trainset)}, Val: {len(valset)}")

    base_prompt = load_base_system_prompt()
    seed_candidate = {"system_prompt": base_prompt}

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")

    adapter = EntitiesAdapter(
        model=opt_model,
        api_key=api_key,
        api_base=api_base,
        temperature=0.0,
    )

    logger.info("Evaluating baseline candidate...")
    baseline_f1 = evaluate_candidate(adapter, seed_candidate, valset)
    logger.info("Baseline F1: %.4f", baseline_f1)

    auto_budget = {"light": 200, "medium": 400, "heavy": 800}
    metric_budget = max_metric_calls or auto_budget.get(auto, 200)
    log_dir = HERE / "gepa_logs" / f"{opt_model}_{opt_reasoning}"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting GEPA optimize (max_metric_calls=%s)...", metric_budget)
    result = gepa_optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=gt_model,
        module_selector="round_robin",
        reflection_minibatch_size=3,
        max_metric_calls=metric_budget,
        run_dir=str(log_dir),
        use_mlflow=True,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment,
        track_best_outputs=True,
        display_progress_bar=True,
    )

    best_candidate = result.best_candidate
    best_prompt = best_candidate["system_prompt"]

    best_f1 = evaluate_candidate(adapter, best_candidate, valset)
    logger.info("Optimized F1: %.4f", best_f1)
    logger.info("Improvement: %+0.4f", best_f1 - baseline_f1)

    out_dir = HERE / "optimized"
    out_dir.mkdir(exist_ok=True)
    prompt_path = out_dir / f"entities_{opt_model}_{opt_reasoning}.txt"
    prompt_path.write_text(best_prompt)

    result_path = out_dir / f"entities_{opt_model}_{opt_reasoning}_result.json"
    result_path.write_text(
        json.dumps(
            {
                "baseline_f1": baseline_f1,
                "optimized_f1": best_f1,
                "improvement": best_f1 - baseline_f1,
                "best_candidate_idx": result.best_idx,
                "run_dir": result.run_dir,
            },
            indent=2,
        )
    )

    logger.info("=" * 80)
    logger.info("Saved optimized prompt to: %s", prompt_path)
    logger.info("Result summary written to: %s", result_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
