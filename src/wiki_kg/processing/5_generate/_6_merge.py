"""
Merge all individual graphs into a single deduplicated graph.

TODO: Implement hierarchical merging for better scalability:
- Process in batches of 100-1000 graphs
- Merge and deduplicate in parallel
- Save intermediate results to GCS
- Final merge of intermediate results
"""

import json
import logging
from pathlib import Path
from typing import Optional, Annotated

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


def load_graph_from_gcs(file_path: str, fs: gcsfs.GCSFileSystem) -> Graph:
    """Load a graph from GCS."""
    with fs.open(file_path, "r") as f:
        data = json.load(f)

    return Graph(
        entities=set(data.get("entities", [])),
        relations={
            (r["subject"], r["predicate"], r["object"])
            for r in data.get("relations", [])
        },
        edges=set(data.get("edges", [])),
    )


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
    """Merge individual knowledge graphs into a single deduplicated graph."""
    fs = gcsfs.GCSFileSystem()
    kggen = KGGen()

    graphs_dir = f"{GCP_KG_PREFIX}/{wiki}/graphs"
    output_filename = build_filename("merged", model, reasoning_effort, limit, ext=".json")
    output_path = f"{GCP_KG_PREFIX}/{wiki}/{output_filename}"

    logger.info(f"Script: {__file__}")
    logger.info(
        f"Args: wiki={wiki}, model={model}, reasoning_effort={reasoning_effort}, limit={limit}"
    )
    logger.info("=" * 80)
    logger.info(f"Graphs directory: {graphs_dir}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 80)

    # Find all graph JSON files
    graphs_dir_clean = graphs_dir.replace("gs://", "")
    graph_files = fs.ls(graphs_dir_clean, detail=False)
    graph_files = [f"gs://{f}" for f in graph_files if f.endswith(".json")]

    logger.info(f"Found {len(graph_files)} graph files to merge")

    # Load all graphs
    graphs = []
    for i, graph_file in enumerate(graph_files):
        try:
            graph = load_graph_from_gcs(graph_file, fs)
            graphs.append(graph)

            if (i + 1) % 100 == 0:
                logger.info(f"Loaded {i + 1}/{len(graph_files)} graphs...")
        except Exception as e:
            logger.error(f"Error loading {graph_file}: {e}")
            continue

    logger.info(f"Successfully loaded {len(graphs)} graphs")

    # Aggregate all graphs
    logger.info("Aggregating graphs...")
    aggregated_graph = kggen.aggregate(graphs)
    logger.info(
        f"Aggregated: {len(aggregated_graph.entities)} entities, {len(aggregated_graph.relations)} relations"
    )

    # Deduplicate
    logger.info("Deduplicating with SEMHASH method...")
    final_graph = kggen.deduplicate(aggregated_graph, method=DeduplicateMethod.SEMHASH)
    logger.info(
        f"Final: {len(final_graph.entities)} entities, {len(final_graph.relations)} relations"
    )

    # Save
    logger.info(f"Saving merged graph to {output_path}...")
    graph_json = json.dumps(
        {
            "entities": list(final_graph.entities),
            "relations": [
                {"subject": r[0], "predicate": r[1], "object": r[2]}
                for r in final_graph.relations
            ],
            "edges": list(final_graph.edges),
        },
        indent=2,
    )

    output_dir = "/".join(output_path.replace("gs://", "").split("/")[:-1])
    fs.makedirs(output_dir, exist_ok=True)

    with fs.open(output_path, "w") as f:
        f.write(graph_json)

    logger.info("=" * 80)
    logger.info(f"Saved merged graph to {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
