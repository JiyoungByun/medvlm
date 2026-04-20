#!/usr/bin/env python3
"""
Example 2: Post-hoc calibration reduces ECE.

Splits a dataset into val/test, computes confidence on both, fits four
post-hoc calibration methods on val, and evaluates on test.  Shows that
simple calibration (temperature scaling, Platt, isotonic) substantially
reduces ECE on the held-out test set.

Expected output (Qwen3-VL-2B, 100 questions, 30 val / 70 test):
    Raw test ECE:    0.17
    Platt ECE:       0.05  (-0.12)
    Isotonic ECE:    0.04  (-0.13)

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/calibration.py
    CUDA_VISIBLE_DEVICES=0 python examples/calibration.py --model qwen3vl_8b --n-questions 0
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

    # 1. Load and split dataset
    kwargs = dict(split="test", question_type="closed", seed=args.seed)
    if args.n_questions > 0:
        kwargs["subsample_size"] = args.n_questions
    dataset = medvlm.load_dataset(args.dataset, **kwargs)
    val_set, test_set = train_val_test_split(
        dataset, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Dataset: {args.dataset} ({len(dataset)} total, "
          f"{len(val_set)} val, {len(test_set)} test)")

    # 2. Load model
    model, processor, config = medvlm.load_model(args.model)
    print(f"Model: {config.model_id}\n")

    # 3. Compute confidence on both splits
    print("Computing sampling confidence (N=20) on val...")
    val_results = medvlm.compute_confidence(
        model, processor, config, val_set,
        method="sampling", num_samples=20, temperature=0.7,
    )
    print("Computing sampling confidence (N=20) on test...")
    test_results = medvlm.compute_confidence(
        model, processor, config, test_set,
        method="sampling", num_samples=20, temperature=0.7,
    )

    val_conf = np.array([r.confidence for r in val_results])
    val_corr = np.array([r.is_correct for r in val_results])
    test_conf = np.array([r.confidence for r in test_results])
    test_corr = np.array([r.is_correct for r in test_results])

    # 4. Evaluate raw and calibrated
    raw_report = medvlm.evaluate_calibration(test_corr, test_conf)
    raw_auroc = auroc(test_corr, test_conf)

    print("=" * 65)
    print("  POST-HOC CALIBRATION (fit on val, evaluate on test)")
    print("=" * 65)
    print(f"\n  {'Method':<25} {'ECE':>7} {'delta':>8} {'AUROC':>7} {'delta':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Raw (uncalibrated)':<25} {raw_report['ece']:>7.4f} {'':>8} "
          f"{raw_auroc:>7.4f}")

    for method in ["temperature_scaling", "platt", "isotonic", "histogram_binning"]:
        cal = CalibrationPipeline(method=method)
        cal.fit(val_conf, val_corr)
        cal_conf = cal.transform(test_conf)

        cal_report = medvlm.evaluate_calibration(test_corr, cal_conf)
        cal_auroc = auroc(test_corr, cal_conf)
        d_ece = cal_report["ece"] - raw_report["ece"]
        d_auc = cal_auroc - raw_auroc

        print(f"  {method:<25} {cal_report['ece']:>7.4f} {d_ece:>+8.4f} "
              f"{cal_auroc:>7.4f} {d_auc:>+8.4f}")

    print(f"\n  Post-hoc calibration reduces ECE without retraining.")
    print(f"  Note: AUROC is unchanged by monotonic methods (Platt, temp scaling)")
    print(f"  because they preserve the ranking of confidence scores.")


if __name__ == "__main__":
    main()
