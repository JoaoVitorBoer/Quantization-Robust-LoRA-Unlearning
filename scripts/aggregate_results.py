#!/usr/bin/env python3
"""
aggregate_results.py
--------------------
Aggregate TOFU evaluation results from Muon and Adam unlearning runs into a
single CSV file.

Expected directory layout (4 levels deep under each root):
    {root}/{method}/{data_split}/{model}/{lr_folder}/TOFU_SUMMARY.json

Example paths:
    saves/unlearn/muon_results/SimNPO/forget01/Llama-3.2-1B-Instruct/lr-1e-5/TOFU_SUMMARY.json
    saves/unlearn/adam_results/SimNPO/forget01/Llama-3.2-1B-Instruct/lr-1e-5_ep-5/TOFU_SUMMARY.json

Usage:
    python scripts/aggregate_results.py \
        --muon-root  saves/unlearn/muon_results \
        --adam-root  saves/unlearn/adam_results \
        --output     results/tofu_aggregated.csv

    # Only one root is required; omit the other if unavailable.
    python scripts/aggregate_results.py \
        --muon-root saves/unlearn/muon_results \
        --output    results/muon_only.csv
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns used to compute the memorization score.
MEM_SOURCE_COLS = [
    "extraction_strength",
    "exact_memorization",
    "forget_Q_A_PARA_Prob",
    "forget_truth_ratio",
]

# Metadata levels relative to the root (root / method / split / model / lr_folder / file)
LEVELS = {
    "method":         -5,   # e.g. SimNPO
    "data_split":     -4,   # e.g. forget01
    "model":          -3,   # e.g. Llama-3.2-1B-Instruct
    "learning_rate":  -2,   # e.g. lr-1e-5  or  lr-1e-5_ep-5
}

SORT_COLS = ["optimizer_type", "method", "data_split", "model", "learning_rate"]


# ---------------------------------------------------------------------------
# Helper: row-wise harmonic mean across a DataFrame
# ---------------------------------------------------------------------------

def harmonic_mean_frame(df: pd.DataFrame) -> pd.Series:
    """
    Compute the harmonic mean across columns for every row.

    harmonic_mean(x1, x2, ..., xn) = n / sum(1/xi)

    Rows with any zero or NaN value will yield NaN.
    """
    n = df.shape[1]
    # reciprocals; 0 and NaN → NaN (replace so we don't divide by zero)
    recip = df.replace(0, np.nan).rdiv(1.0)   # 1 / each cell
    return n / recip.sum(axis=1)


# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------

def parse_metadata(json_path: Path, optimizer_type: str) -> dict:
    """
    Extract metadata from the path of a TOFU_SUMMARY.json file.

    Expected structure (counting from the end of path parts):
        index -5 → method
        index -4 → data_split
        index -3 → model
        index -2 → learning_rate (lr_folder)
        index -1 → TOFU_SUMMARY.json
    """
    parts = json_path.parts
    meta = {"optimizer_type": optimizer_type}
    for field, idx in LEVELS.items():
        try:
            meta[field] = parts[idx]
        except IndexError:
            warnings.warn(
                f"Cannot extract '{field}' from path (not enough components): {json_path}"
            )
            meta[field] = None
    return meta


# ---------------------------------------------------------------------------
# Single-file loader
# ---------------------------------------------------------------------------

def load_summary(json_path: Path, optimizer_type: str) -> dict | None:
    """
    Load one TOFU_SUMMARY.json and attach metadata.
    Returns None (with a warning) if the file is missing or malformed.
    """
    try:
        with open(json_path, "r") as fh:
            metrics = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"Skipping {json_path}: {exc}")
        return None

    if not isinstance(metrics, dict):
        warnings.warn(f"Skipping {json_path}: expected a JSON object, got {type(metrics)}")
        return None

    row = parse_metadata(json_path, optimizer_type)
    row.update(metrics)
    return row


# ---------------------------------------------------------------------------
# Root crawler
# ---------------------------------------------------------------------------

def collect_rows(root: Path, optimizer_type: str) -> list[dict]:
    """
    Recursively find every TOFU_SUMMARY.json under *root* and load each one.
    """
    rows = []
    summaries = sorted(root.rglob("TOFU_SUMMARY.json"))

    if not summaries:
        warnings.warn(f"No TOFU_SUMMARY.json files found under: {root}")

    for path in summaries:
        row = load_summary(path, optimizer_type)
        if row is not None:
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

def compute_memorization(df: pd.DataFrame) -> pd.Series:
    """
    memorization = harmonic_mean(
        1 - extraction_strength,
        1 - exact_memorization,
        1 - forget_Q_A_PARA_Prob,
        1 - forget_truth_ratio,
    )

    Reproduces the logic from the original codebase:
        df['mem'] = harmonic_mean_frame(1.0 - df[MEM_SOURCE_COLS])
    """
    missing = [c for c in MEM_SOURCE_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: cannot compute mem; missing columns: {missing}")
        return pd.Series(np.nan, index=df.index)

    return harmonic_mean_frame(1.0 - df[MEM_SOURCE_COLS])


def compute_agg_memorization(df: pd.DataFrame) -> pd.Series:
    """
    agg_memorization = harmonic_mean(model_utility, memorization)

    Returns NaN for rows where either component is NaN.
    """
    if "model_utility" not in df.columns or "memorization" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    pair = df[["model_utility", "memorization"]]
    return harmonic_mean_frame(pair)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataframe(muon_root: Path | None, adam_root: Path | None) -> pd.DataFrame:
    """
    Collect all rows from both roots, compute derived metrics, and return a
    sorted DataFrame.
    """
    rows: list[dict] = []

    if muon_root is not None:
        rows.extend(collect_rows(muon_root, optimizer_type="muon"))

    if adam_root is not None:
        rows.extend(collect_rows(adam_root, optimizer_type="adam"))

    if not rows:
        print("No results found. Output CSV will be empty.", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    # memorization  (same logic as original: harmonic mean of 1 - col)
    df["memorization"] = compute_memorization(df)

    # agg_memorization: harmonic mean of model_utility and memorization
    df["agg_memorization"] = compute_agg_memorization(df)

    # ------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------
    existing_sort_cols = [c for c in SORT_COLS if c in df.columns]
    df = df.sort_values(existing_sort_cols, ignore_index=True)

    return df


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Write the aggregated DataFrame to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate TOFU evaluation results from Muon and/or Adam runs "
            "into a single CSV file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--muon-root",
        type=Path,
        default=None,
        help="Root directory for Muon results (e.g. saves/unlearn/muon_results).",
    )
    parser.add_argument(
        "--adam-root",
        type=Path,
        default=None,
        help="Root directory for Adam results (e.g. saves/unlearn/adam_results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tofu_aggregated.csv"),
        help="Path to the output CSV file (default: results/tofu_aggregated.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.muon_root is None and args.adam_root is None:
        print(
            "ERROR: Provide at least one of --muon-root or --adam-root.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate that provided roots actually exist
    for label, root in [("--muon-root", args.muon_root), ("--adam-root", args.adam_root)]:
        if root is not None and not root.is_dir():
            warnings.warn(f"{label} does not exist or is not a directory: {root}")

    df = build_dataframe(
        muon_root=args.muon_root,
        adam_root=args.adam_root,
    )

    write_csv(df, args.output)


if __name__ == "__main__":
    main()
