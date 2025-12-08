#!/bin/bash
set -e

# Usage: ./run.sh <model> <reasoning_effort> [limit] [--force]
# Example: ./run.sh gpt-5-nano minimal 100 --force

MODEL="${1:?Usage: ./run.sh <model> <reasoning_effort> [limit] [--force]}"
REASONING_EFFORT="${2:?Usage: ./run.sh <model> <reasoning_effort> [limit] [--force]}"
LIMIT=""
FORCE=""

# Parse remaining args
shift 2
for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE="--force"
    elif [ -z "$LIMIT" ] && [[ "$arg" =~ ^[0-9]+$ ]]; then
        LIMIT="$arg"
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$SCRIPT_DIR"

# Set GCS credentials if not already set
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ -f "$PROJECT_ROOT/google-credentials.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$PROJECT_ROOT/google-credentials.json"
    echo "Using credentials: $GOOGLE_APPLICATION_CREDENTIALS"
fi

# Build common args
COMMON_ARGS="--model $MODEL --reasoning-effort $REASONING_EFFORT"
if [ -n "$LIMIT" ]; then
    COMMON_ARGS="$COMMON_ARGS --limit $LIMIT"
fi

echo "========================================"
echo "Knowledge Graph Generation Pipeline"
echo "========================================"
echo "Model: $MODEL"
echo "Reasoning Effort: $REASONING_EFFORT"
echo "Limit: ${LIMIT:-none}"
echo "Force: ${FORCE:-no}"
echo "========================================"

wait_for_batch() {
    local batch_type=$1
    echo "Waiting for $batch_type batch to complete..."
    
    while true; do
        # Extract status using sed (works on macOS)
        status=$(uv run python batch_api.py status "$batch_type" $COMMON_ARGS 2>&1 | grep "Status:" | head -1 | sed 's/.*Status: //')
        echo "  Status: $status"
        
        if [ "$status" = "completed" ]; then
            echo "  ✓ $batch_type batch completed!"
            break
        elif [ "$status" = "failed" ] || [ "$status" = "expired" ] || [ "$status" = "cancelled" ]; then
            echo "  ✗ $batch_type batch $status!"
            exit 1
        fi
        
        echo "  Waiting 30 seconds..."
        sleep 30
    done
}

# Step 1: Generate entities batch file
echo ""
echo "[1/10] Generating entities batch file..."
uv run python _1_generate_entities.py $COMMON_ARGS $FORCE

# Step 2: Upload entities batch
echo ""
echo "[2/10] Uploading entities batch..."
uv run python batch_api.py upload entities $COMMON_ARGS 

# Step 3: Wait for entities batch to complete
echo ""
echo "[3/10] Waiting for entities batch..."
wait_for_batch entities

# Step 4: Download entities results
echo ""
echo "[4/10] Downloading entities results..."
uv run python batch_api.py download entities $COMMON_ARGS

# Step 5: Parse entities
echo ""
echo "[5/10] Parsing entities..."
uv run python _2_parse_entities.py $COMMON_ARGS

# Step 6: Generate relations batch file
echo ""
echo "[6/10] Generating relations batch file..."
uv run python _3_generate_relations.py $COMMON_ARGS $FORCE

# Step 7: Upload relations batch
echo ""
echo "[7/10] Uploading relations batch..."
uv run python batch_api.py upload relations $COMMON_ARGS 

# Step 8: Wait for relations batch to complete
echo ""
echo "[8/10] Waiting for relations batch..."
wait_for_batch relations

# Step 9: Download relations results
echo ""
echo "[9/10] Downloading relations results..."
uv run python batch_api.py download relations $COMMON_ARGS

# Step 10: Parse relations
echo ""
echo "[10/10] Parsing relations..."
uv run python _4_parse_relations.py $COMMON_ARGS

# Step 11: Get graphs
echo ""
echo "[11/12] Generating individual graphs..."
uv run python _5_get_graphs.py $COMMON_ARGS

# Step 12: Merge graphs
echo ""
echo "[12/12] Merging graphs..."
uv run python _6_merge.py $COMMON_ARGS

echo ""
echo "========================================"
echo "Pipeline completed successfully!"
echo "========================================"

