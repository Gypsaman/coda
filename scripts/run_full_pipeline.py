"""
CODA Full Pipeline Runner

Chains all four phases: Diagnose -> Classify -> Optimize -> Validate
for one or all case studies.

Usage:
    python scripts/run_full_pipeline.py --case a
    python scripts/run_full_pipeline.py --case all
    python scripts/run_full_pipeline.py --case a --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_diagnosis import run_diagnosis
from scripts.run_classification import run_classification
from scripts.run_optimization import run_optimization
from scripts.run_validation import run_validation


def run_pipeline(case_id: str, dry_run: bool = False):
    print(f"\n{'#'*60}")
    print(f"# CODA FULL PIPELINE - Case {case_id.upper()}")
    print(f"# Started: {datetime.now().isoformat()}")
    print(f"{'#'*60}")

    # Phase 1: Diagnose
    diagnosis = run_diagnosis(case_id, dry_run=dry_run)
    if dry_run:
        return

    # Find the results directory (most recent for this case)
    results_base = Path(__file__).parent.parent / "results" / f"case_{case_id}"
    results_dirs = sorted(results_base.iterdir(), reverse=True)
    if not results_dirs:
        print("ERROR: No results directory found after diagnosis.")
        return
    results_dir = str(results_dirs[0])

    # Check if optimization is needed
    if diagnosis["ppi"] >= 98:
        print(f"\nPPI is {diagnosis['ppi']:.1f} (GREEN zone). No optimization needed.")
        print("Pipeline complete.")
        return

    # Phase 2: Classify
    classification = run_classification(case_id, results_dir)

    if classification.get("status") == "no_failures_to_classify":
        print("\nNo failures to classify. Pipeline complete.")
        return

    # Phase 3: Optimize
    optimization = run_optimization(case_id, results_dir)

    # Phase 4: Validate
    validation = run_validation(case_id, results_dir)

    # Final summary
    print(f"\n{'#'*60}")
    print(f"# PIPELINE COMPLETE - Case {case_id.upper()}")
    print(f"# Status: {validation['status']}")
    print(f"# PPI: {diagnosis['ppi']:.1f} -> {validation['ppi_after_optimization']:.1f}")
    print(f"# Zone: {diagnosis['triage_zone'].upper()} -> "
          f"{validation['triage_zone_after'].upper()}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CODA Full Pipeline")
    parser.add_argument("--case", required=True, choices=["a", "b", "c", "all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cases = ["a", "b", "c"] if args.case == "all" else [args.case]
    for case_id in cases:
        run_pipeline(case_id, dry_run=args.dry_run)
