#!/usr/bin/env python3
"""
Example 1: Medical VLMs are overconfident.

Loads a VLM, runs sampling-based and verbalized confidence estimation on
VQA-RAD (closed questions), and prints a calibration report showing that
the model's mean confidence consistently exceeds its accuracy.

Expected output (Qwen3-VL-2B, 50 questions):
    Sampling:   Acc=0.76  MeanConf=0.94  Overconfidence=+0.19
    Verbalized: Acc=0.68  MeanConf=0.95  Overconfidence=+0.27

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/overconfidence.py
    CUDA_VISIBLE_DEVICES=0 python examples/overconfidence.py --model qwen3vl_8b
"""

import argparse
import numpy as np

import medvlm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3vl_2b",
                        help="Model key (e.g. qwen3vl_2b, qwen3vl_8b, internvl3_8b)")
    parser.add_argument("--dataset", default="vqa_rad")
    parser.add_argument("--n-questions", type=int, default=50,
                        help="Subsample size (0 = full dataset)")
    args = parser.parse_args()

    # 1. Load dataset
    kwargs = dict(split="test", question_type="closed")
    if args.n_questions > 0:
        kwargs["subsample_size"] = args.n_questions
    dataset = medvlm.load_dataset(args.dataset, **kwargs)
    print(f"Dataset: {args.dataset}, {len(dataset)} closed questions")

    # 2. Load model
    model, processor, config = medvlm.load_model(args.model)
    print(f"Model: {config.model_id}\n")

    # 3. Sampling-based confidence (N=20)
    sampling_results = medvlm.compute_confidence(
        model, processor, config, dataset,
        method="sampling", num_samples=20, temperature=0.7,
    )
    s_conf = np.array([r.confidence for r in sampling_results])
    s_corr = np.array([r.is_correct for r in sampling_results])
    s_report = medvlm.evaluate_calibration(s_corr, s_conf)

    # 4. Verbalized confidence (vanilla)
    verb_results = medvlm.compute_confidence(
        model, processor, config, dataset,
        method="verbalized", variant="vanilla", batch_size=4,
    )
    v_conf = np.array([r.confidence for r in verb_results])
    v_corr = np.array([r.is_correct for r in verb_results])
    v_report = medvlm.evaluate_calibration(v_corr, v_conf)

    # 5. Print results
    print("=" * 60)
    print("  OVERCONFIDENCE ANALYSIS")
    print("=" * 60)

    for name, report in [("Sampling (N=20)", s_report),
                         ("Verbalized (vanilla)", v_report)]:
        acc = report["accuracy"]
        conf = report["mean_confidence"]
        oc = report["overconfidence"]
        ece = report["ece"]
        print(f"\n  {name}:")
        print(f"    Accuracy:        {acc:.4f}")
        print(f"    Mean Confidence: {conf:.4f}")
        print(f"    Overconfidence:  {oc:+.4f}  (conf - acc = {conf - acc:+.4f})")
        print(f"    ECE:             {ece:.4f}")

    print(f"\n  Both methods show overconfidence: the model reports")
    print(f"  higher confidence than its actual accuracy warrants.")


if __name__ == "__main__":
    main()
