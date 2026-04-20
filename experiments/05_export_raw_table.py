#!/usr/bin/env python3
"""
Export a flat CSV table with per-question confidence scores from ALL methods.

One row per (dataset, question_type, question_id, model).
Columns include confidence from sampling, verbalized, and HEDGE methods.

Usage:
    python experiments/05_export_raw_table.py
    python experiments/05_export_raw_table.py --output colm_results/raw_calibration_table.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medvlm.configs import MODELS, MODEL_SHORT_KEYS
from experiments.config import COLM_RESULTS, SAMPLING_DIR, VERBALIZED_DIR


# =========================================================================
# Loading helpers (priority: clustered > judged > regular)
# =========================================================================

def _load_best(base_dir):
    base = Path(base_dir)
    for fname in ["detailed_results_clustered.json",
                   "detailed_results_judged.json",
                   "detailed_results.json"]:
        p = base / fname
        if p.exists():
            with open(p) as f:
                return json.load(f), fname
    jsonl = base / "detailed_results.jsonl"
    if jsonl.exists():
        with open(jsonl) as f:
            return [json.loads(line) for line in f], "detailed_results.jsonl"
    return None, None


def qt_suffix(question_type):
    return "" if question_type == "closed" else f"_{question_type}"


# =========================================================================
# Main
# =========================================================================

def build_table():
    models = list(MODELS.keys())
    datasets = ["vqa_rad", "slake", "vqa_med_2019", "vqa_med_2021"]

    sampling_prompts = ["base", "cot"]
    verbalized_variants = ["vanilla", "vanilla_cot", "punish", "top_k", "two_stage", "linguistic"]

    rows = []

    for model_key in models:
        model_short = MODEL_SHORT_KEYS[model_key]
        model_id = MODELS[model_key]

        for ds in datasets:
            if ds == "vqa_med_2021":
                qt_list = ["open"]
            else:
                qt_list = ["closed", "open"]

            for qt in qt_list:
                suf = qt_suffix(qt)

                # Use first available source to get question list
                ref_data = None

                ref_dir = SAMPLING_DIR / f"base_{model_short}_{ds}{suf}_sampling"
                ref_data, ref_fname = _load_best(ref_dir)

                if not ref_data:
                    continue

                n_questions = len(ref_data)

                # Pre-load all data sources
                sampling_data = {}
                for prompt in sampling_prompts:
                    d, _ = _load_best(SAMPLING_DIR / f"{prompt}_{model_short}_{ds}{suf}_sampling")
                    sampling_data[prompt] = d

                verb_data = {}
                for variant in verbalized_variants:
                    d, _ = _load_best(VERBALIZED_DIR / variant / f"{model_key}_{ds}{suf}")
                    verb_data[variant] = d

                hedge_data = None
                hedge_path = COLM_RESULTS / "hedge_vase" / f"{model_short}_{ds}{suf}" / "detailed_results.json"
                if hedge_path.exists():
                    with open(hedge_path) as f:
                        hedge_data = json.load(f)

                # Build rows
                for i in range(n_questions):
                    row = OrderedDict()
                    row["dataset"] = ds
                    row["question_type"] = qt
                    row["question_id"] = i
                    row["question"] = ref_data[i].get("question", "")
                    row["golden_answer"] = ref_data[i].get("ground_truth", "")
                    row["model_key"] = model_key
                    row["model_short"] = model_short
                    row["model_id"] = model_id

                    # ---- Sampling confidences ----
                    for prompt in sampling_prompts:
                        col = f"sampling_{prompt}_conf"
                        cor_col = f"sampling_{prompt}_correct"
                        d = sampling_data[prompt]
                        if d and i < len(d):
                            row[col] = d[i].get("confidence")
                            row[cor_col] = d[i].get("is_correct")
                        else:
                            row[col] = None
                            row[cor_col] = None

                    # ---- Verbalized confidences ----
                    for variant in verbalized_variants:
                        col = f"verbalized_{variant}_conf"
                        cor_col = f"verbalized_{variant}_correct"
                        d = verb_data.get(variant)
                        if d and i < len(d):
                            row[col] = d[i].get("confidence")
                            row[cor_col] = d[i].get("is_correct")
                        else:
                            row[col] = None
                            row[cor_col] = None

                    # ---- HEDGE/VASE scores ----
                    if hedge_data and i < len(hedge_data):
                        row["hedge_SE"] = hedge_data[i].get("SE")
                        row["hedge_RadFlag"] = hedge_data[i].get("RadFlag")
                        row["hedge_VASE"] = hedge_data[i].get("VASE")
                    else:
                        row["hedge_SE"] = None
                        row["hedge_RadFlag"] = None
                        row["hedge_VASE"] = None

                    rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Export raw calibration table")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = args.output or str(COLM_RESULTS / "raw_calibration_table.csv")

    print("Building table...")
    df = build_table()

    print(f"\nTable shape: {df.shape}")
    print(f"Models: {df['model_short'].nunique()}")
    print(f"Datasets: {df['dataset'].nunique()}")
    print(f"Question types: {df['question_type'].value_counts().to_dict()}")
    print(f"\nColumn coverage (non-null %):")
    conf_cols = [c for c in df.columns
                 if c.endswith("_conf") or c.endswith("_correct") or c.startswith("hedge_")]
    for c in conf_cols:
        pct = df[c].notna().mean() * 100
        print(f"  {c:<35} {pct:5.1f}%")

    df.to_csv(output, index=False)
    print(f"\nSaved to {output}")

    pq_path = output.replace(".csv", ".parquet")
    df.to_parquet(pq_path, index=False)
    print(f"Saved to {pq_path}")


if __name__ == "__main__":
    main()
