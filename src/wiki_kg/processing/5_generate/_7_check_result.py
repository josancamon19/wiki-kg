"""Visualize a merged knowledge graph.

This script:
- Accepts the same (model, reasoning_effort, limit) parameters used by the 5_generate pipeline
- Loads the merged graph JSON (local path or GCS path)
- Converts it to ``kg_gen.models.Graph``
- Calls ``KGGen.visualize`` to generate an HTML visualization

Typical usage (GCS merged output):
    python -m wiki_kg.processing.5_generate._6_check_result --wiki enwiki \
      --model gpt-5-nano --reasoning-effort minimal --limit 100 \
      --open

Local JSON file:
    python -m wiki_kg.processing.5_generate._6_check_result \
      --graph-path ./result-5.1-high.json --output ./analysis/graph.html
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Annotated, Any

import gcsfs
import typer
from dotenv import load_dotenv
from kg_gen import KGGen
from kg_gen.models import Graph

try:
    from .utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
    )
except ImportError:
    from utils import (
        GCP_KG_PREFIX,
        DEFAULT_MODEL,
        DEFAULT_REASONING_EFFORT,
    )

load_dotenv()

app = typer.Typer(add_completion=False)

logger = logging.getLogger(__name__)


def _default_merged_graph_path(
    *, wiki: str, model: str, reasoning_effort: str, limit: Optional[int]
) -> str:
    """Path written by `_6_merge.py` for the merged graph."""
    limit_suffix = f"-l{limit}" if limit else ""
    filename = f"{model}-{reasoning_effort}{limit_suffix}.json"
    return f"{GCP_KG_PREFIX}/{wiki}/full/{filename}"


def _load_json(*, path: str, fs: Optional[gcsfs.GCSFileSystem]) -> dict[str, Any]:
    if path.startswith("gs://"):
        if fs is None:
            fs = gcsfs.GCSFileSystem()
        with fs.open(path, "r") as f:
            return json.load(f)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _graph_from_json(data: dict[str, Any]) -> Graph:
    entity_metadata = None
    if data.get("entity_metadata"):
        entity_metadata = {
            str(entity): set(article_ids)
            for entity, article_ids in data["entity_metadata"].items()
            if isinstance(article_ids, list)
        }

    return Graph(
        entities=set(data.get("entities", [])),
        relations={
            (r.get("subject"), r.get("predicate"), r.get("object"))
            for r in data.get("relations", [])
            if isinstance(r, dict)
            and r.get("subject")
            and r.get("predicate")
            and r.get("object")
        },
        edges=set(data.get("edges", [])),
        entity_metadata=entity_metadata,
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
    open_in_browser: Annotated[
        bool, typer.Option("--open", help="Open the HTML in a browser")
    ] = True,
):
    # Logging (keep simple: stdout only)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    fs: Optional[gcsfs.GCSFileSystem] = None

    resolved_graph_path = _default_merged_graph_path(
        wiki=wiki,
        model=model,
        reasoning_effort=reasoning_effort,
        limit=limit,
    )

    default_out_name = (
        f"{wiki}-{model}-{reasoning_effort}" + (f"-l{limit}" if limit else "") + ".html"
    )
    resolved_output = str(Path("analysis") / default_out_name)

    logger.info(
        "Loading graph JSON: wiki=%s model=%s reasoning_effort=%s limit=%s",
        wiki,
        model,
        reasoning_effort,
        limit,
    )
    logger.info("Graph path: %s", resolved_graph_path)

    if resolved_graph_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        if not fs.exists(resolved_graph_path):
            raise typer.BadParameter(f"Graph not found on GCS: {resolved_graph_path}")
    else:
        if not Path(resolved_graph_path).exists():
            raise typer.BadParameter(f"Graph file not found: {resolved_graph_path}")

    data = _load_json(path=resolved_graph_path, fs=fs)
    graph = _graph_from_json(data)

    logger.info(
        "Graph loaded: %d entities, %d relations, %d edges",
        len(graph.entities),
        len(graph.relations),
        len(graph.edges),
    )

    out_path = Path(resolved_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kggen = KGGen()
    logger.info("Visualizing to: %s", out_path)
    kggen.visualize(
        graph=graph, output_path=str(out_path), open_in_browser=open_in_browser
    )


if __name__ == "__main__":
    app()
