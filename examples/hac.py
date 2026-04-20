#!/usr/bin/env python3
"""
Example 3: HAC improves both ECE and AUROC.

Hallucination-Aware Calibration (HAC) combines a model's self-reported
confidence with an independent hallucination signal to produce calibrated
probabilities.  Unlike standard post-hoc methods (Platt, isotonic), HAC
can improve AUROC (discrimination) because the hallucination score provides
information that the original confidence does not capture.

This example:
  1. Computes sampling-based confidence (N=20)
  2. Computes VASE hallucination scores via ``compute_hedge_scores``
     (greedy + N high-temp + N distorted-image samples, clustered)
  3. Compares standard Platt scaling vs HAC-Platt vs HAC-Gate

HAC methods:
  - HAC-Platt:  sigma(a*c + b*h + d),  a>=0, b<=0
  - HAC-Gate:   c * sigma(-a*h + b),    a>=0

Expected output (Qwen3-VL-8B, 100 questions, 30 val / 70 test):
    Standard Platt:  AUROC unchanged (~0.54)
    HAC-Platt:       AUROC improved  (~0.58, +0.04)
    HAC-Gate:        AUROC improved  (~0.57, +0.03)

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/hac.py
    CUDA_VISIBLE_DEVICES=0 python examples/hac.py --model qwen3vl_8b
"""

import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

import medvlm
from medvlm import CalibrationPipeline, train_val_test_split


def auroc(corr, conf):
    try:
        return roc_auc_score(corr, conf)
    except ValueError:
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3vl_2b")
    parser.add_argument("--dataset", default="vqa_rad")
    parser.add_argument("--n-questions", type=int, default=100,
                        help="Subsample size (0 = full dataset)")
    parser.add_argument("--val-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Load and split
    kwargs = dict(split="test", question_type="closed", seed=args.seed)
    if args.n_questions > 0:
        kwargs["subsample_size"] = args.n_questions
    dataset = medvlm.load_dataset(args.dataset, **kwargs)
    val_set, test_set = train_val_test_split(
        dataset, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Dataset: {args.dataset} ({len(val_set)} val, {len(test_set)} test)")

    # 2. Load model
    model, processor, config = medvlm.load_model(args.model)
    print(f"Model: {config.model_id}\n")

    # 3. Sampling confidence on both splits
    print("Computing sampling confidence (N=20)...")
    val_results = medvlm.compute_confidence(
        model, processor, config, val_set,
        method="sampling", num_samples=20, temperature=0.7,
    )
    test_results = medvlm.compute_confidence(
        model, processor, config, test_set,
        method="sampling", num_samples=20, temperature=0.7,
    )

    val_conf = np.array([r.confidence for r in val_results])
    val_corr = np.array([r.is_correct for r in val_results])
    test_conf = np.array([r.confidence for r in test_results])
    test_corr = np.array([r.is_correct for r in test_results])

    # 4. Compute VASE hallucination scores (uses hedge-bench under the hood)
    print("Computing HEDGE/VASE scores (N=10)...")
    val_hedge = medvlm.compute_hedge_scores(
        model, processor, config, val_set,
        question_type="closed", n_samples=10,
    )
    test_hedge = medvlm.compute_hedge_scores(
        model, processor, config, test_set,
        question_type="closed", n_samples=10,
    )
    val_h = np.array([r.VASE for r in val_hedge])
    test_h = np.array([r.VASE for r in test_hedge])
    print(f"VASE: mean={np.mean(test_h):.3f}, std={np.std(test_h):.3f}\n")

    # 5. Compare methods
    raw_report = medvlm.evaluate_calibration(test_corr, test_conf)
    raw_auroc = auroc(test_corr, test_conf)

    print("=" * 70)
    print("  HAC vs STANDARD CALIBRATION (fit on val, evaluate on test)")
    print("=" * 70)
    print(f"\n  {'Method':<25} {'ECE':>7} {'dECE':>8} {'AUROC':>7} {'dAUROC':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Raw (uncalibrated)':<25} {raw_report['ece']:>7.4f} {'':>8} "
          f"{raw_auroc:>7.4f}")

    methods = [
        # Standard (no hallucination scores)
        ("platt", False),
        ("isotonic", False),
        # HAC (uses hallucination scores)
        ("hac_platt", True),
        ("hac_gate", True),
    ]

    for method, uses_h in methods:
        cal = CalibrationPipeline(method=method)
        if uses_h:
            cal.fit(val_conf, val_corr, hallucination_scores=val_h)
            cal_conf = cal.transform(test_conf, hallucination_scores=test_h)
        else:
            cal.fit(val_conf, val_corr)
            cal_conf = cal.transform(test_conf)

        report = medvlm.evaluate_calibration(test_corr, cal_conf)
        auc = auroc(test_corr, cal_conf)
        d_ece = report["ece"] - raw_report["ece"]
        d_auc = auc - raw_auroc

        marker = " *" if uses_h else ""
        print(f"  {method + marker:<25} {report['ece']:>7.4f} {d_ece:>+8.4f} "
              f"{auc:>7.4f} {d_auc:>+8.4f}")

    print(f"\n  * = HAC method (uses hallucination scores)")
    print(f"\n  Standard Platt/isotonic preserve AUROC (monotonic transforms).")
    print(f"  HAC methods can improve AUROC because they incorporate an")
    print(f"  independent hallucination signal that helps discriminate")
    print(f"  correct from incorrect predictions.")
    print(f"\n  Hallucination score: VASE (from compute_hedge_scores). Swap to")
    print(f"  r.SE or r.RadFlag if you prefer semantic-entropy or RadFlag.")


if __name__ == "__main__":
    main()
