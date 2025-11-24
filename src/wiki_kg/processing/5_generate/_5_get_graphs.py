# take the result files from entities and relations, use KGGen() to create the graphs.
# deduplicate individually, and upload to GCP each one.

import json
from pathlib import Path
from kg_gen import KGGen
from kg_gen.kg_gen import DeduplicateMethod
from kg_gen.models import Graph
import multiprocessing as mp
from typing import Tuple


def count_lines(file_path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_range(
    entities_path: Path, relations_path: Path, start_idx: int, end_idx: int
) -> Tuple[dict, dict]:
    """Load entities and relations for a specific range of rows."""
    entities_dict = {}
    relations_dict = {}

    # Load entities for the range
    with open(entities_path, "r") as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            if line.strip():
                data = json.loads(line)
                custom_id = data["custom_id"]
                entities_dict[custom_id] = data["entities"]

    # Load relations for the range
    with open(relations_path, "r") as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if i >= end_idx:
                break
            if line.strip():
                data = json.loads(line)
                custom_id = data["custom_id"]
                relations_dict[custom_id] = data["relations"]

    return entities_dict, relations_dict


def process_range(
    entities_path: Path,
    relations_path: Path,
    output_dir: Path,
    start_idx: int,
    end_idx: int,
    worker_id: int,
) -> int:
    """Process graphs for a specific range of rows."""
    # Initialize KGGen for this process
    kggen = KGGen()

    # Load data for this range
    entities_dict, relations_dict = load_range(
        entities_path, relations_path, start_idx, end_idx
    )

    processed = 0
    print(
        f"Worker {worker_id}: Processing {len(entities_dict)} items (rows {start_idx}-{end_idx})"
    )

    # Process each custom_id in this range
    for custom_id in entities_dict.keys():
        if custom_id not in relations_dict:
            print(f"Worker {worker_id}: Warning: No relations found for {custom_id}")
            continue

        # Get entities and relations for this custom_id
        entities = entities_dict[custom_id]
        relations = relations_dict[custom_id]

        # Extract unique predicates (edges)
        edges = {
            relation["predicate"]
            for relation in relations
            if relation.get("predicate") is not None
        }

        # Create graph
        graph = Graph(
            entities=set(entities),
            relations={
                (r["subject"], r["predicate"], r["object"])
                for r in relations
                if r.get("predicate") is not None
            },
            edges=edges,
        )

        # Deduplicate the graph
        graph = kggen.deduplicate(graph, method=DeduplicateMethod.SEMHASH)

        # Prepare output path: results/graphs/enwiki/$id.json
        parts = custom_id.split("/")
        if len(parts) == 2:
            namespace, doc_id = parts
            output_path = output_dir / namespace / f"{doc_id}.json"
        else:
            # Fallback if format is unexpected
            output_path = output_dir / f"{custom_id.replace('/', '_')}.json"

        # Save the graph
        kggen.export_graph(graph, str(output_path))
        processed += 1

        if processed % 10 == 0:
            print(
                f"Worker {worker_id}: Processed {processed}/{len(entities_dict)} items"
            )

    print(f"Worker {worker_id}: Completed {processed} items")
    return processed


def main():
    # Define paths
    entities_path = Path(
        "src/wiki_kg/processing/5_generate/results/entities/parsed_entities.jsonl"
    )
    relations_path = Path(
        "src/wiki_kg/processing/5_generate/results/relations/parsed_relations.jsonl"
    )
    output_dir = Path("src/wiki_kg/processing/5_generate/results/graphs")

    # Count total lines
    print("Counting total lines...")
    total_lines = count_lines(entities_path)
    print(f"Total lines: {total_lines}")

    # Get number of CPUs
    num_workers = mp.cpu_count()
    print(f"Using {num_workers} workers")

    # Calculate chunk size
    chunk_size = (total_lines + num_workers - 1) // num_workers

    # Create worker arguments
    worker_args = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_lines)
        if start_idx >= total_lines:
            break
        worker_args.append(
            (entities_path, relations_path, output_dir, start_idx, end_idx, i)
        )

    # Process in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(process_range, worker_args)

    total_processed = sum(results)
    print(f"\nTotal processed: {total_processed} graphs")


if __name__ == "__main__":
    main()
