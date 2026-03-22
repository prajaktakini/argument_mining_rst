"""
scripts/run_ablations.sh

Runs all five ablation conditions from Section 6.3 of the proposal and
collects results into a single comparison table.

Usage:
    bash scripts/run_ablations.sh

Results are written to outputs/<run_name>/test_results.json for each condition.
Run compare_ablations.py afterwards to print the full table.
"""

#!/usr/bin/env bash
set -e

echo "=== RST-AM Ablation Suite ==="

echo "--- [1/5] Full model ---"
python train.py --run_name full_model

echo "--- [2/5] Untyped edges ---"
python train.py --ablation_untyped_edges --run_name untyped_edges

echo "--- [3/5] No nuclearity weighting ---"
python train.py --ablation_no_nuclearity --run_name no_nuclearity

echo "--- [4/5] Single-task: relation only ---"
python train.py --single_task relation --run_name single_relation

echo "--- [5/5] RoBERTa baseline (no graph) ---"
python train.py --ablation_no_graph --run_name no_graph_baseline

echo "=== All ablations complete. Run: python scripts/compare_ablations.py ==="
