#!/usr/bin/env python3
"""Generate HEDGE / RadFlag / VASE hallucination scores for Medical VQA.

Thin CLI wrapper around ``medvlm.compute_hedge_scores``. Loads the model and
dataset, runs the full pipeline (greedy + N high-temp + N distorted), and
saves results to JSON.

Usage:
    python experiments/03_generate_hedge.py --model_key qwen3vl_2b --dataset vqa_rad --gpu 4
    python experiments/03_generate_hedge.py --model_key internvl3_2b --dataset slake --gpu 4
"""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

# Parse GPU first, before importing torch. Default None so the user's
# external CUDA_VISIBLE_DEVICES is respected; pass --gpu explicitly to override.
_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu", type=str, default=None)
_gpu_args, _ = _gpu_parser.parse_known_args()
if _gpu_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_args.gpu
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={_gpu_args.gpu}")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medvlm.configs import (
    ModelConfig, DataConfig, DatasetName, QuestionType,
    MODELS, MODEL_SHORT_KEYS, SEED,
)
from medvlm.data import get_dataset
from medvlm.models import load_model
from medvlm.utils import set_seed
from medvlm.confidence import compute_hedge_scores

from experiments.config import COLM_RESULTS, DATA_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Generate HEDGE/VASE scores")
    p.add_argument("--model_key", required=True, choices=list(MODELS.keys()))
    p.add_argument("--dataset", required=True,
                   choices=["vqa_rad", "slake", "vqa_med_2019",
                            "vqa_med_2020", "vqa_med_2021"])
    p.add_argument("--question_type", default="closed", choices=["closed", "open"])
    p.add_argument("--gpu", type=str, default=None)
    p.add_argument("--n_samples", type=int, default=10,
                   help="Number of high-temp samples AND distortions")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="VASE alpha parameter for contrastive weighting")
    p.add_argument("--batch_size", type=int, default=5,
                   help="Batch size for distorted-image generation")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing results")
    p.add_argument("--subsample-size", type=int, default=None,
                   help="Cap dataset to N examples (smoke testing).")
    return p.parse_args()


_DATASET_NAME_MAP = {
    "vqa_rad": DatasetName.VQA_RAD,
    "slake": DatasetName.SLAKE,
    "vqa_med_2019": DatasetName.VQA_MED_2019,
    "vqa_med_2020": DatasetName.VQA_MED_2020,
    "vqa_med_2021": DatasetName.VQA_MED_2021,
}
_QT_MAP = {"closed": QuestionType.CLOSED, "open": QuestionType.OPEN}


def _data_path_for(dataset: str) -> str | None:
    if dataset == "slake":
        return str(DATA_DIR / "Slake1.0")
    if dataset == "vqa_med_2019":
        return str(DATA_DIR / "vqa_med_2019" / "VQAMed2019Test")
    if dataset == "vqa_med_2020":
        return str(DATA_DIR / "vqa_med_2020"
                   / "VQA-TestSet-ReferenceAnswers-VQAMed2020-Task1")
    if dataset == "vqa_med_2021":
        return str(DATA_DIR / "vqa_med_2021"
                   / "Task1-VQA-2021-TestSet-w-GroundTruth")
    return None


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    model_id = MODELS[args.model_key]
    model_short = MODEL_SHORT_KEYS[args.model_key]

    dir_name = f"{model_short}_{args.dataset}"
    if args.question_type != "closed":
        dir_name += f"_{args.question_type}"
    output_path = Path(args.output_dir or COLM_RESULTS / "hedge_vase" / dir_name)

    marker = output_path / "detailed_results.json"
    if marker.exists() and not args.force:
        print(f"Results already exist at {output_path}. Use --force to overwrite.")
        return
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model and dataset
    # flash_attention disabled due to hedge-bench upgrading torch (ABI break)
    model_config = ModelConfig(model_id=model_id, use_flash_attention=False)
    model, processor = load_model(model_config)
    model.eval()

    data_config = DataConfig(
        dataset_name=_DATASET_NAME_MAP[args.dataset],
        question_type=_QT_MAP[args.question_type],
        split="test",
        data_path=_data_path_for(args.dataset),
        subsample_size=args.subsample_size,
    )
    dataset = get_dataset(data_config).load()

    print(f"Model: {model_id}")
    print(f"Dataset: {args.dataset} ({len(dataset)} {args.question_type} questions)")
    print(f"N samples/distortions: {args.n_samples}, alpha: {args.alpha}")
    print(f"Output: {output_path}")

    results = compute_hedge_scores(
        model, processor, model_config, dataset,
        question_type=args.question_type,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        alpha=args.alpha,
        batch_size=args.batch_size,
    )

    with open(output_path / "detailed_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    config = {
        "model_id": model_id,
        "model_key": args.model_key,
        "dataset": args.dataset,
        "question_type": args.question_type,
        "method": "hedge_vase",
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    accuracy = float(np.mean([r.is_correct for r in results]))
    mean_se = float(np.mean([r.SE for r in results]))
    mean_radflag = float(np.mean([r.RadFlag for r in results]))
    mean_vase = float(np.mean([r.VASE for r in results]))

    print(f"\nDone. {len(results)} results saved to {output_path}")
    print(f"  Accuracy:     {accuracy:.3f}")
    print(f"  Mean SE:      {mean_se:.4f}")
    print(f"  Mean RadFlag: {mean_radflag:.4f}")
    print(f"  Mean VASE:    {mean_vase:.4f}")


if __name__ == "__main__":
    main()
