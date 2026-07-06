"""
Roll up every case's latest results into the paper's table format.

Reads results/case_<id>/<latest-timestamp>/{diagnosis_report,validation_report,
summary}.json for each case and emits both a CSV and LaTeX-table-row rendering
matching coda.md's results table (old/new MHS, PPI_init, optimized MHS,
PPI_final, outcome).

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --cases a b c
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from evaluators.metrics import METRIC_MAPPING, compute_mhs, map_to_weighted

ALL_CASES = ["a", "b", "c", "d", "e", "f", "g"]


def _latest_results_dir(case_id: str) -> Path | None:
    case_dir = ROOT / "results" / f"case_{case_id}"
    if not case_dir.exists():
        return None
    subdirs = sorted((d for d in case_dir.iterdir() if d.is_dir()), reverse=True)
    return subdirs[0] if subdirs else None


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def build_row(case_id: str, weights: dict[str, float]) -> dict | None:
    results_dir = _latest_results_dir(case_id)
    if results_dir is None:
        return None

    diagnosis = _load_json(results_dir / "diagnosis_report.json")
    if diagnosis is None:
        return None
    validation = _load_json(results_dir / "validation_report.json")

    row = {
        "case": case_id.upper(),
        "old_mhs": diagnosis["mhs_baseline"],
        "new_mhs": diagnosis["mhs_new_model"],
        "ppi_init": diagnosis["ppi"],
        "opt_mhs": np.nan,
        "ppi_final": np.nan,
        "outcome": "Diagnosed only (no optimization run yet)",
    }

    if diagnosis["triage_zone"] == "green":
        row["outcome"] = "No optimization needed"
    elif validation is not None:
        opt_metrics = validation["metrics"]["optimized_new_model"]
        mapped = map_to_weighted(opt_metrics, METRIC_MAPPING, weights.keys())
        row["opt_mhs"] = compute_mhs(mapped, weights) if opt_metrics else np.nan
        row["ppi_final"] = validation["ppi_after_optimization"]
        row["outcome"] = "Recovered" if validation["status"] == "PASS" else "Failed -- optimization insufficient"

    return row


def to_latex_row(row: dict) -> str:
    def fmt(v):
        return "---" if pd.isna(v) else f"{v:.2f}" if isinstance(v, float) and v < 10 else f"{v:.1f}"

    return (
        f"{row['case']} & {fmt(row['old_mhs'])} & {fmt(row['new_mhs'])} & "
        f"{fmt(row['ppi_init'])} & {fmt(row['opt_mhs'])} & {fmt(row['ppi_final'])} & "
        f"{row['outcome']} \\\\"
    )


def main(cases: list[str]):
    with open(ROOT / "config" / "models.yaml") as f:
        models_config = yaml.safe_load(f)

    rows = [
        r for r in (
            build_row(c, models_config["cases"][c]["evaluator_config"]["metric_weights"])
            for c in cases
        )
        if r is not None
    ]
    if not rows:
        print("No results found for any requested case. Run the pipeline first.")
        return

    df = pd.DataFrame(rows)
    out_csv = ROOT / "results" / "aggregate_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    print("\nLaTeX table rows:\n")
    for row in rows:
        print(to_latex_row(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate CODA results across cases")
    parser.add_argument("--cases", nargs="+", default=ALL_CASES, choices=ALL_CASES)
    args = parser.parse_args()
    main(args.cases)
