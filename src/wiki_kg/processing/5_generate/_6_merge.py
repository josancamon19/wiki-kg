# Merge all individual graphs into a single deduplicated graph
# TODO: this is definitely not scalable.
# data ls, ordered by id, then take every 100 kg's, merge with kg.aggregate, then with deduplicate(semhash)
# do this in parallel, in multiproc, every 10000 articles, save the temporal results locally, store temps in gcp as well {from_id}_{to_id}.json
# then merge those aggregate, and 1 single semhash, store this in GCP
# .. could also generate embeddings of all the titles, and group every 10000 using knn, and generate graphs for topics, then merge lthem

from pathlib import Path
from kg_gen import KGGen
from kg_gen.kg_gen import DeduplicateMethod

# Initialize KGGen
kggen = KGGen()

# Define paths
graphs_dir = Path("src/wiki_kg/processing/5_generate/results/graphs")
output_path = Path("src/wiki_kg/processing/5_generate/results/merged_graph.json")

# Find all graph JSON files
graph_files = list(graphs_dir.rglob("*.json"))
print(f"Found {len(graph_files)} graph files to merge")

# Load all graphs
graphs = []
for graph_file in graph_files:
    try:
        graph = KGGen.from_file(str(graph_file))
        graphs.append(graph)
    except Exception as e:
        print(f"Error loading {graph_file}: {e}")
        continue

print(f"Successfully loaded {len(graphs)} graphs")

# Aggregate all graphs
print("Aggregating graphs...")
aggregated_graph = kggen.aggregate(graphs)
print(
    f"Aggregated graph has {len(aggregated_graph.entities)} entities and {len(aggregated_graph.relations)} relations"
)

# Deduplicate using semhash method
print("Deduplicating with SEMHASH method...")
final_graph = kggen.deduplicate(aggregated_graph, method=DeduplicateMethod.SEMHASH)
print(
    f"Final graph has {len(final_graph.entities)} entities and {len(final_graph.relations)} relations"
)

# Save the final merged graph
kggen.export_graph(final_graph, str(output_path))
print(f"Saved merged graph to {output_path}")
