"""
Generate individual knowledge graphs from parsed entities and relations.
"""

import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Annotated

import gcsfs
import typer
from dotenv import load_dotenv
from kg_gen import KGGen
from kg_gen.kg_gen import DeduplicateMethod
from kg_gen.models import Graph

try:
    from .utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        build_filename,
    )
except ImportError:
    from utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
        build_filename,
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

app = typer.Typer()


def count_lines(file_path: str, fs: gcsfs.GCSFileSystem) -> int:
    """Count non-empty lines in a JSONL file on GCS."""
    count = 0
    with fs.open(file_path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_range(
    entities_path: str,
    relations_path: str,
    start_idx: int,
    end_idx: int,
    fs: gcsfs.GCSFileSystem,
) -> Tuple[Dict[str, List[str]], Dict[str, List[dict]]]:
    """Load entities and relations for a specific range of rows from GCS."""
    entities_dict = {}
    relations_dict = {}

    with fs.open(entities_path, "r") as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            if line.strip():
                data = json.loads(line)
                entities_dict[data["custom_id"]] = data["entities"]

    with fs.open(relations_path, "r") as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            if line.strip():
                data = json.loads(line)
                relations_dict[data["custom_id"]] = data["relations"]

    return entities_dict, relations_dict


def process_range(
    entities_path: str,
    relations_path: str,
    output_dir: str,
    start_idx: int,
    end_idx: int,
    worker_id: int,
) -> int:
    """Process graphs for a specific range of rows and save to GCS."""
    fs = gcsfs.GCSFileSystem()
    kggen = KGGen()

    entities_dict, relations_dict = load_range(
        entities_path, relations_path, start_idx, end_idx, fs
    )

    processed = 0
    logger.info(
        f"Worker {worker_id}: Processing {len(entities_dict)} items (rows {start_idx}-{end_idx})"
    )

    for custom_id in entities_dict.keys():
        if custom_id not in relations_dict:
            logger.warning(f"Worker {worker_id}: No relations found for {custom_id}")
            continue

        entities = entities_dict[custom_id]
        relations = relations_dict[custom_id]

        edges = {r["predicate"] for r in relations if r.get("predicate") is not None}

        graph = Graph(
            entities=set(entities),
            relations={
                (r["subject"], r["predicate"], r["object"])
                for r in relations
                if r.get("predicate") is not None
            },
            edges=edges,
        )

        graph = kggen.deduplicate(graph, method=DeduplicateMethod.SEMHASH)

        output_dir_clean = output_dir.replace("gs://", "")
        output_path = f"{output_dir}/{custom_id.replace('/', '_')}.json"
        fs.makedirs(output_dir_clean, exist_ok=True)

        graph_json = json.dumps(
            {
                "entities": list(graph.entities),
                "relations": [
                    {"subject": r[0], "predicate": r[1], "object": r[2]}
                    for r in graph.relations
                ],
                "edges": list(graph.edges),
            }
        )

        with fs.open(output_path, "w") as f:
            f.write(graph_json)

        processed += 1
        if processed % 10 == 0:
            logger.info(
                f"Worker {worker_id}: Processed {processed}/{len(entities_dict)} items"
            )

    logger.info(f"Worker {worker_id}: Completed {processed} items")
    return processed


@app.command()
def main(
    wiki: Annotated[str, typer.Option(help="Wiki identifier")] = "enwiki",
    model: Annotated[str, typer.Option(help="Model name")] = DEFAULT_MODEL,
    reasoning_effort: Annotated[
        str, typer.Option(help="Reasoning effort")
    ] = DEFAULT_REASONING_EFFORT,
    limit: Annotated[
        Optional[int], typer.Option(help="Limit used during generation")
    ] = None,
):
    """Generate individual knowledge graphs from entities and relations."""
    fs = gcsfs.GCSFileSystem()

    entities_filename = build_filename("parsed", model, reasoning_effort, limit)
    relations_filename = build_filename("parsed", model, reasoning_effort, limit)

    entities_path = f"{GCP_KG_PREFIX}/{wiki}/entities/{entities_filename}"
    relations_path = f"{GCP_KG_PREFIX}/{wiki}/relations/{relations_filename}"
    output_dir = f"{GCP_KG_PREFIX}/{wiki}/graphs"

    logger.info(f"Script: {__file__}")
    logger.info(
        f"Args: wiki={wiki}, model={model}, reasoning_effort={reasoning_effort}, limit={limit}"
    )
    logger.info("=" * 80)
    logger.info(f"Entities: {entities_path}")
    logger.info(f"Relations: {relations_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    if not fs.exists(entities_path):
        logger.error(f"Entities file not found: {entities_path}")
        raise typer.Exit(1)
    if not fs.exists(relations_path):
        logger.error(f"Relations file not found: {relations_path}")
        raise typer.Exit(1)

    logger.info("Counting total lines...")
    total_lines = count_lines(entities_path, fs)
    logger.info(f"Total lines: {total_lines}")

    num_workers = mp.cpu_count()
    logger.info(f"Using {num_workers} workers")

    chunk_size = (total_lines + num_workers - 1) // num_workers

    worker_args = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_lines)
        if start_idx >= total_lines:
            break
        worker_args.append(
            (entities_path, relations_path, output_dir, start_idx, end_idx, i)
        )

    logger.info("Starting parallel processing...")
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(process_range, worker_args)

    total_processed = sum(results)
    logger.info("=" * 80)
    logger.info(f"Total processed: {total_processed} graphs")
    logger.info(f"Saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
