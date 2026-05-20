#!/usr/bin/env bash
set -euo pipefail

KOHONEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$KOHONEN_DIR")"
VENV="$REPO_ROOT/.venv/bin/activate"

if [[ ! -f "$VENV" ]]; then
    echo "ERROR: virtualenv not found at $VENV" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$VENV"

EXPERIMENTS=(
    grid_architecture
    neighborhood_functions
    decay
    initialization
    training_phases
    bmu_distance
)

echo "========================================"
echo " Kohonen — running all experiments"
echo " Seeds: 10 runs per variant (master=2025)"
echo "========================================"
echo ""

TOTAL=${#EXPERIMENTS[@]}
for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    echo "[$((i + 1))/$TOTAL] $exp"
    echo "----------------------------------------"
    python "$KOHONEN_DIR/experiments/$exp.py"
    echo ""
done

echo "========================================"
echo " All experiments complete."
echo " Results in: $KOHONEN_DIR/output/"
echo "========================================"
