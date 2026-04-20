#!/usr/bin/env python3
"""
Sampling-based evaluation for BASE models.

Generates N=20 samples per question using the BASE sampling prompt,
counts answer frequencies, and computes sampling-based confidence.

Supports all question types: closed, open.

Usage:
    python experiments/01_generate_sampling.py --gpu 0 --model qwen3vl_8b --dataset vqa_rad
    python experiments/01_generate_sampling.py --gpu 0 --model all --dataset all --question-type closed
    python experiments/01_generate_sampling.py --dry-run
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Parse GPU before importing torch
_parser_gpu = argparse.ArgumentParser(add_help=False)
_parser_gpu.add_argument("--gpu", type=str, default=None)
_args_gpu, _ = _parser_gpu.parse_known_args()
if _args_gpu.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _args_gpu.gpu
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={_args_gpu.gpu}")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medvlm.configs import (
    MODELS,
    DATASETS,
    QUESTION_TYPES,
    BASE_SAMPLING_PROMPT,
    COT_SAMPLING_PROMPT,
    SAMPLING_N,
    SAMPLING_TEMP,
    SEED,
    MODEL_SHORT_KEYS,
)
from experiments.config import SAMPLING_DIR, DATA_DIR, results_exist

SAMPLING_PROMPTS = {
    "base": BASE_SAMPLING_PROMPT,
    "cot": COT_SAMPLING_PROMPT,
}
# CoT needs more tokens for reasoning
PROMPT_MAX_TOKENS = {
    "base": 64,
    "cot": 256,
}
from medvlm.configs import ModelConfig, DataConfig, DatasetName, QuestionType, ModelFamily
from medvlm.data import get_dataset
from medvlm.models.two_stage_loader import smart_load_model
from medvlm.utils import set_seed


# =============================================================================
# Answer Parsing
# =============================================================================

def parse_answer_text(text):
    """Extract answer from 'Answer: XXX' format.

    Returns normalized (lowercased, stripped) answer or None.
    """
    # Pattern: "Answer: XXX" (take until newline or end)
    match = re.search(r"answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().rstrip(".")
        return answer.lower()

    # Fallback: short response (1-3 words), use the whole thing
    text_stripped = text.strip().rstrip(".")
    if len(text_stripped.split()) <= 3:
        return text_stripped.lower()

    return None


def check_answer_correct(predicted, ground_truth, question_type="closed"):
    """Check if predicted answer matches ground truth.

    For closed: exact match (yes/no).
    For open: containment-based matching.
    """
    if predicted is None or predicted == "unknown":
        return False

    if question_type == "closed":
        return predicted == ground_truth

    pred = predicted.lower().strip()
    gt = ground_truth.lower().strip()

    if pred == gt:
        return True
    if gt in pred:
        return True
    if pred in gt:
        return True
    return False


def normalize_answer(answer, question_type="closed"):
    """Normalize parsed answer for comparison.

    For closed: map to yes/no.
    For open: lowercase strip.
    """
    if answer is None:
        return None

    answer = answer.lower().strip()

    if question_type == "closed":
        # Try to map to yes/no
        if answer in ("yes", "yes.", "yes,"):
            return "yes"
        if answer in ("no", "no.", "no,"):
            return "no"
        if answer.startswith("yes"):
            return "yes"
        if answer.startswith("no"):
            return "no"
        # Ambiguous
        if "yes" in answer and "no" not in answer:
            return "yes"
        if "no" in answer and "yes" not in answer:
            return "no"
        return None  # Unparseable for yes/no

    # For open: return as-is
    return answer


# =============================================================================
# Generation
# =============================================================================

def format_prompt(processor, model_config, image, prompt_text):
    """Format a single prompt for generation."""
    if model_config.model_family == ModelFamily.QWEN_VL:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
    elif model_config.model_family == ModelFamily.INTERNVL:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(images=image, text=text, return_tensors="pt")
    elif model_config.model_family in [ModelFamily.LLAVA, ModelFamily.LLAVA_NEXT]:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=text, images=image, return_tensors="pt")
    else:
        raise ValueError(f"Unsupported model family: {model_config.model_family}")
    return inputs


def generate_samples(model, processor, inputs, num_sequences, temperature=0.7, max_new_tokens=64):
    """Generate multiple responses from a single prompt using num_return_sequences."""
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_sequences,
            pad_token_id=pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    responses = []
    for i in range(num_sequences):
        generated = outputs[i, prompt_len:]
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(text)
    return responses


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_single(model, processor, model_config, sample, question_type,
                    prompt_template, max_new_tokens=64,
                    num_samples=20, samples_per_batch=25, temperature=0.7,
                    effective_batch_size=None):
    """Evaluate a single question via sampling.

    Returns (result_dict, effective_batch_size) -- the batch size may be reduced on OOM.
    """
    question = sample["question"]
    ground_truth = sample["answer"].lower().strip()

    prompt_text = prompt_template.format(question=question)
    inputs = format_prompt(processor, model_config, sample["image"], prompt_text)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    answer_counts = {}
    unknown_count = 0
    raw_responses = []

    samples_remaining = num_samples
    current_batch_size = effective_batch_size or samples_per_batch
    while samples_remaining > 0:
        batch_size = min(current_batch_size, samples_remaining)
        try:
            responses = generate_samples(
                model, processor, inputs, batch_size,
                temperature=temperature, max_new_tokens=max_new_tokens,
            )
            for response in responses:
                raw_responses.append(response)
                raw_answer = parse_answer_text(response)
                normalized = normalize_answer(raw_answer, question_type)
                if normalized is not None:
                    answer_counts[normalized] = answer_counts.get(normalized, 0) + 1
                else:
                    unknown_count += 1
            samples_remaining -= batch_size
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                current_batch_size = max(1, batch_size // 2)
                print(f"  OOM at batch_size={batch_size}, retrying with {current_batch_size}")
                continue
            raise
        except Exception as e:
            print(f"  Batch failed (n={batch_size}): {type(e).__name__}: {e}")
            unknown_count += batch_size
            samples_remaining -= batch_size

    total = sum(answer_counts.values()) + unknown_count
    valid_count = sum(answer_counts.values())
    valid_response_rate = valid_count / total if total > 0 else 0

    if valid_count > 0:
        predicted = max(answer_counts, key=answer_counts.get)
        confidence = answer_counts[predicted] / valid_count
    else:
        predicted = "unknown"
        confidence = 0.5

    is_correct = check_answer_correct(predicted, ground_truth, question_type)

    return {
        "question": question,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "confidence": confidence,
        "is_correct": is_correct,
        "answer_counts": answer_counts,
        "unknown_count": unknown_count,
        "valid_response_rate": valid_response_rate,
        "raw_responses": raw_responses,
        "method": "sampling",
    }, current_batch_size


def save_results(results, output_dir, config):
    """Save sampling results."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    n = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    avg_conf = np.mean([r["confidence"] for r in results]) if n else 0
    print(f"  Saved {n} results to {output_dir}")
    print(f"  Acc={correct/n:.4f} mean_conf={avg_conf:.4f}" if n else "")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Sampling evaluation for BASE models")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--dataset", type=str, default="all",
                        choices=DATASETS + ["all"])
    parser.add_argument("--question-type", type=str, default="closed",
                        choices=QUESTION_TYPES,
                        help="Question type to evaluate (closed, open)")
    parser.add_argument("--prompt-mode", type=str, default="base",
                        choices=list(SAMPLING_PROMPTS.keys()),
                        help="Sampling prompt mode: base (direct) or cot (chain-of-thought)")
    parser.add_argument("--num-samples", type=int, default=SAMPLING_N)
    parser.add_argument("--samples-per-batch", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=SAMPLING_TEMP)
    parser.add_argument("--subsample-size", type=int, default=None,
                        help="Cap dataset to N examples (smoke testing).")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (faster on high-VRAM GPUs)")
    parser.add_argument("--use-8bit", action="store_true",
                        help="Use 8-bit quantization instead of 4-bit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    question_type = args.question_type
    prompt_mode = args.prompt_mode
    prompt_template = SAMPLING_PROMPTS[prompt_mode]
    max_new_tokens = PROMPT_MAX_TOKENS[prompt_mode]

    # Output dir suffix for non-closed question types
    qt_suffix = f"_{question_type}" if question_type != "closed" else ""
    # Prompt mode prefix: "base" for backward compat, "cot" for new
    prompt_prefix = "base" if prompt_mode == "base" else f"{prompt_mode}"

    # Build job list
    jobs = []
    for model_key in models:
        for dataset in datasets:
            short_key = MODEL_SHORT_KEYS.get(model_key, model_key)
            output_dir = SAMPLING_DIR / f"{prompt_prefix}_{short_key}_{dataset}{qt_suffix}_sampling"

            if not args.force and results_exist(output_dir):
                print(f"  [SKIP] {prompt_prefix}_{short_key}_{dataset}{qt_suffix}_sampling")
                continue

            jobs.append({
                "model_key": model_key,
                "model_id": MODELS[model_key],
                "dataset": dataset,
                "output_dir": str(output_dir),
            })

    if not jobs:
        print("No jobs to run (all results exist)")
        return

    print(f"\n{len(jobs)} jobs to run (question_type={question_type}, prompt_mode={prompt_mode}):")
    for j in jobs:
        short_key = MODEL_SHORT_KEYS.get(j["model_key"], j["model_key"])
        print(f"  {prompt_prefix}_{short_key}_{j['dataset']}{qt_suffix}_sampling")

    if args.dry_run:
        print("\n[DRY-RUN] No models loaded, no evaluation performed.")
        return

    # Group by model to avoid reloading
    from collections import defaultdict
    jobs_by_model = defaultdict(list)
    for j in jobs:
        jobs_by_model[j["model_key"]].append(j)

    for model_key, model_jobs in jobs_by_model.items():
        model_id = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"Loading: {model_id}")
        print(f"{'='*60}")

        model_config = ModelConfig(model_id=model_id, use_4bit=not args.no_4bit, use_8bit=args.use_8bit)
        model, processor = smart_load_model(model_config, adapter_path=None)
        model.eval()

        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

        # Cache datasets
        dataset_cache = {}

        for job in model_jobs:
            dataset_name = job["dataset"]
            output_dir = job["output_dir"]

            short_key = MODEL_SHORT_KEYS.get(model_key, model_key)
            print(f"\n--- {prompt_prefix}_{short_key}_{dataset_name}{qt_suffix}_sampling ---")

            if dataset_name not in dataset_cache:
                data_config_kwargs = {
                    "dataset_name": DatasetName(dataset_name),
                    "question_type": QuestionType(question_type),
                    "split": "test",
                    "seed": SEED,
                }
                if dataset_name == "slake":
                    data_config_kwargs["data_path"] = str(DATA_DIR / "Slake1.0")
                elif dataset_name == "vqa_med_2019":
                    data_config_kwargs["data_path"] = str(DATA_DIR / "vqa_med_2019" / "VQAMed2019Test")
                elif dataset_name == "vqa_med_2020":
                    data_config_kwargs["data_path"] = str(DATA_DIR / "vqa_med_2020" / "VQA-TestSet-ReferenceAnswers-VQAMed2020-Task1")
                elif dataset_name == "vqa_med_2021":
                    data_config_kwargs["data_path"] = str(DATA_DIR / "vqa_med_2021" / "Task1-VQA-2021-TestSet-w-GroundTruth")
                if args.subsample_size is not None:
                    data_config_kwargs["subsample_size"] = args.subsample_size
                data_config = DataConfig(**data_config_kwargs)
                dataset_cache[dataset_name] = get_dataset(data_config).load()

            dataset = dataset_cache[dataset_name]
            print(f"  Loaded {len(dataset)} {question_type} questions")

            # Incremental JSONL saving with resume support
            os.makedirs(output_dir, exist_ok=True)
            jsonl_path = os.path.join(output_dir, "detailed_results.jsonl")
            results = []
            done_indices = set()
            if os.path.exists(jsonl_path):
                with open(jsonl_path) as jf:
                    for line in jf:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        results.append(r)
                        done_indices.add(r["dataset_index"])
                print(f"  Resuming: {len(results)} questions already done")
            jsonl_file = open(jsonl_path, "a")
            eff_batch_size = None  # tracks OOM-adjusted batch size across questions
            for idx, sample in enumerate(dataset):
                if idx in done_indices:
                    continue
                result, eff_batch_size = evaluate_single(
                    model, processor, model_config, sample, question_type,
                    prompt_template=prompt_template,
                    max_new_tokens=max_new_tokens,
                    num_samples=args.num_samples,
                    samples_per_batch=args.samples_per_batch,
                    temperature=args.temperature,
                    effective_batch_size=eff_batch_size,
                )
                result["dataset_index"] = idx
                results.append(result)
                jsonl_file.write(json.dumps(result) + "\n")
                jsonl_file.flush()
                n_done = len(results)
                if n_done % 20 == 0 or (idx + 1) == len(dataset):
                    n_correct = sum(1 for r in results if r["is_correct"])
                    avg_conf = np.mean([r["confidence"] for r in results])
                    print(f"  [{idx+1}/{len(dataset)}] (done={n_done}) acc={n_correct/n_done:.2%} "
                          f"mean_conf={avg_conf:.3f}")
            jsonl_file.close()
            config = {
                "model_id": model_id,
                "model_key": model_key,
                "dataset": dataset_name,
                "question_type": question_type,
                "method": "sampling",
                "prompt_mode": prompt_mode,
                "prompt_template": prompt_template,
                "num_samples": args.num_samples,
                "samples_per_batch": args.samples_per_batch,
                "temperature": args.temperature,
                "max_new_tokens": max_new_tokens,
                "seed": SEED,
            }

            save_results(results, output_dir, config)

        # Free GPU memory
        del model, processor
        torch.cuda.empty_cache()

    print("\nAll sampling evaluations complete.")


if __name__ == "__main__":
    main()
