"""
scripts/compare_ablations.py

Loads test_results.json from each ablation run and prints a comparison table.

Usage:
    python scripts/compare_ablations.py --output_dir outputs/
"""

import json
import argparse
from pathlib import Path
from evaluate import compare_ablations

ABLATION_ORDER = [
    "full_model",
    "untyped_edges",
    "no_nuclearity",
    "single_relation",
    "no_graph_baseline",
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/")
    args = p.parse_args()

    results = {}
    for run_name in ABLATION_ORDER:
        path = Path(args.output_dir) / run_name / "test_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            results[run_name] = {k: v for k, v in data.items() if k != "run_name"}
        else:
            print(f"[WARN] No results found for {run_name} at {path}")

    if results:
        compare_ablations(results)
    else:
        print("No results found. Run scripts/run_ablations.sh first.")


if __name__ == "__main__":
    main()
