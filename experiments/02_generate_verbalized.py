#!/usr/bin/env python3
"""
Verbalized confidence generation.

Generates verbalized confidence responses for 6 prompt variants x models x datasets.
Each run produces detailed_results.json and config.json.

Usage:
    python experiments/02_generate_verbalized.py --gpu 0 --model qwen3vl_8b --dataset vqa_rad --variant vanilla
    python experiments/02_generate_verbalized.py --gpu 0 --model all --dataset all --variant all
    python experiments/02_generate_verbalized.py --dry-run
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
    VERBALIZED_VARIANTS,
    VERBALIZED_PROMPTS,
    LINGUISTIC_MAP,
    SEED,
    ModelConfig, DataConfig, DatasetName, QuestionType, ModelFamily,
)
from experiments.config import VERBALIZED_DIR, DATA_DIR, results_exist
from medvlm.data import get_dataset
from medvlm.models.two_stage_loader import smart_load_model
from medvlm.utils import set_seed


# =============================================================================
# Confidence Parsing
# =============================================================================

def parse_answer_yes_no(text):
    """Extract yes/no answer from verbalized response.

    Looks for 'Answer: yes/no' pattern first, then falls back to
    simple yes/no detection.
    """
    text_lower = text.lower().strip()

    # Pattern: "Answer: yes" or "Answer: no"
    match = re.search(r"answer:\s*(yes|no)\b", text_lower)
    if match:
        return match.group(1)

    # Fallback: check if response is just yes/no
    if text_lower in ("yes", "no", "yes.", "no."):
        return text_lower.rstrip(".")

    # Check start
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"

    # Contains (unambiguous)
    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"

    return None


def parse_answer_general(text):
    """Extract answer text from response (any question type).

    Looks for 'Answer: XXX' pattern, then falls back to short responses.
    Returns normalized (lowercased, stripped) answer text or None.
    """
    # Pattern: "Answer: XXX" (take until newline or end)
    match = re.search(r"answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        answer = match.group(1).strip().rstrip(".")
        return answer.lower()

    # Fallback: if response is short (1-5 words), use the whole thing
    text_stripped = text.strip().rstrip(".")
    if len(text_stripped.split()) <= 5:
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

    # For open: more lenient matching
    pred = predicted.lower().strip()
    gt = ground_truth.lower().strip()

    # Exact match
    if pred == gt:
        return True

    # Ground truth contained in predicted (e.g., "coronal plane" in "the coronal plane")
    if gt in pred:
        return True

    # Predicted contained in ground truth
    if pred in gt:
        return True

    return False


def parse_confidence_numeric(text):
    """Extract numeric confidence from 'Confidence: XX%' pattern.

    Returns float in [0, 1] or None if not found.
    """
    # Match "Confidence: 85%" or "Confidence: 85.5%" or "Confidence: 85"
    match = re.search(r"confidence:\s*(\d+(?:\.\d+)?)\s*%?", text.lower())
    if match:
        val = float(match.group(1))
        if val > 1.0:
            val /= 100.0
        return max(0.0, min(1.0, val))
    return None


def parse_confidence_linguistic(text, mapping=LINGUISTIC_MAP):
    """Map linguistic confidence term to numerical value.

    Returns float in [0, 1] or None if no term matched.
    """
    text_lower = text.lower()

    # Look for "Confidence: <term>" pattern
    conf_match = re.search(r"confidence:\s*(.+?)(?:\.|$)", text_lower)
    search_text = conf_match.group(1).strip() if conf_match else text_lower

    # Check each term (longer terms first to avoid partial matches)
    for term in sorted(mapping.keys(), key=len, reverse=True):
        if term in search_text:
            return mapping[term]

    return None


def parse_top_k(text, force_yes_no=True):
    """Extract top guess and its probability from top-K format.

    Args:
        text: Model response text.
        force_yes_no: If True, normalize answer to yes/no. If False, return raw answer.

    Returns (answer, confidence) or (None, None).
    """
    # Match "Guess 1: Yes (Probability: 85%)"
    matches = re.findall(
        r"guess\s*\d+:\s*(.*?)\s*\(probability:\s*(\d+(?:\.\d+)?)\s*%?\)",
        text.lower()
    )
    if matches:
        # Return the guess with highest probability
        best_answer, best_prob = None, -1
        for answer, prob_str in matches:
            prob = float(prob_str)
            if prob > 1.0:
                prob /= 100.0
            if prob > best_prob:
                best_prob = prob
                best_answer = answer.strip()
            prob = max(0.0, min(1.0, prob))
        if force_yes_no:
            ans = parse_answer_yes_no(best_answer) if best_answer else None
        else:
            ans = best_answer
        return ans, max(0.0, min(1.0, best_prob)) if best_prob >= 0 else None
    return None, None


def parse_two_stage_confidence(stage2_text):
    """Extract confidence from stage 2 response."""
    return parse_confidence_numeric(stage2_text)


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_prompt_for_model(processor, model_config, image, prompt_text):
    """Format prompt for a specific model family. Returns tokenized inputs."""
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


def generate_response(model, processor, inputs, max_new_tokens=1024, do_sample=False, temperature=1.0):
    """Generate a single greedy response."""
    with torch.no_grad():
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": processor.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        outputs = model.generate(**gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated = outputs[0, prompt_len:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True)


# =============================================================================
# Batched Generation
# =============================================================================

def format_prompts_batch(processor, model_config, images, prompt_texts):
    """Format and tokenize a batch of image+text prompts for generation.

    Uses the same pattern as training collators but with left padding
    for batched generation.
    """
    texts = []
    for image, prompt_text in zip(images, prompt_texts):
        if model_config.model_family == ModelFamily.QWEN_VL:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ]}]
        else:
            # InternVL, LLaVA, LLaVA-Next all use placeholder
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    # Left padding for batched generation
    orig_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    processor.tokenizer.padding_side = orig_padding_side
    return inputs


def generate_batch(model, processor, inputs, max_new_tokens=256, do_sample=False, temperature=1.0):
    """Generate responses for a batch of inputs. Returns list of strings."""
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        outputs = model.generate(**gen_kwargs)

    # With left padding, all prompts end at the same position
    prompt_len = inputs["input_ids"].shape[1]
    responses = []
    for i in range(outputs.shape[0]):
        generated = outputs[i, prompt_len:]
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(text)
    return responses


# =============================================================================
# Main Evaluation Logic
# =============================================================================

def _parse_variant_response(variant, response, question_type="closed"):
    """Parse predicted answer and confidence from a response based on variant type.

    Args:
        variant: Prompt variant name.
        response: Raw model response text.
        question_type: "closed" uses yes/no parsing, others use general parsing.
    """
    use_yes_no = (question_type == "closed")

    if variant == "top_k":
        predicted, confidence = parse_top_k(response, force_yes_no=use_yes_no)
        if predicted is None:
            predicted = parse_answer_yes_no(response) if use_yes_no else parse_answer_general(response)
    elif variant == "linguistic":
        predicted = parse_answer_yes_no(response) if use_yes_no else parse_answer_general(response)
        confidence = parse_confidence_linguistic(response)
    else:
        # vanilla, vanilla_cot, punish
        predicted = parse_answer_yes_no(response) if use_yes_no else parse_answer_general(response)
        confidence = parse_confidence_numeric(response)
    return predicted, confidence


def evaluate_verbalized(
    model, processor, model_config, dataset, variant, output_dir,
    max_new_tokens=256, do_sample=False, temperature=1.0, batch_size=1,
    question_type="closed", **kwargs,
):
    """Run verbalized confidence evaluation with batched generation.

    Returns detailed_results list.
    """
    detailed_results = []
    dataset_list = list(dataset)
    n = len(dataset_list)

    # Ensure pad_token_id is set for generation
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_samples = dataset_list[batch_start:batch_end]

        questions = [s["question"] for s in batch_samples]
        ground_truths = [s["answer"].lower().strip() for s in batch_samples]
        images = [s["image"] for s in batch_samples]

        if variant == "two_stage":
            # Stage 1: batch generate answers
            s1_prompts = [VERBALIZED_PROMPTS["two_stage_s1"].format(question=q)
                          for q in questions]
            s1_inputs = format_prompts_batch(processor, model_config, images, s1_prompts)
            s1_inputs = {k: v.to(model.device) for k, v in s1_inputs.items()}
            s1_responses = generate_batch(
                model, processor, s1_inputs, max_new_tokens, do_sample, temperature
            )

            # Parse S1 answers
            if question_type == "closed":
                s1_answers = [parse_answer_yes_no(r) or r.strip() for r in s1_responses]
            else:
                s1_answers = [parse_answer_general(r) or r.strip() for r in s1_responses]

            # Stage 2: batch generate confidence
            s2_prompts = [VERBALIZED_PROMPTS["two_stage_s2"].format(question=q, answer=a)
                          for q, a in zip(questions, s1_answers)]
            s2_inputs = format_prompts_batch(processor, model_config, images, s2_prompts)
            s2_inputs = {k: v.to(model.device) for k, v in s2_inputs.items()}
            s2_responses = generate_batch(
                model, processor, s2_inputs, max_new_tokens=256,
                do_sample=do_sample, temperature=temperature,
            )

            for i in range(len(batch_samples)):
                if question_type == "closed":
                    predicted = parse_answer_yes_no(s1_responses[i])
                else:
                    predicted = parse_answer_general(s1_responses[i])
                confidence = parse_two_stage_confidence(s2_responses[i])
                if predicted is None:
                    predicted = "unknown"
                parse_success = confidence is not None
                if confidence is None:
                    confidence = 0.5
                detailed_results.append({
                    "dataset_index": batch_start + i,
                    "question": questions[i],
                    "ground_truth": ground_truths[i],
                    "predicted": predicted,
                    "confidence": confidence,
                    "is_correct": check_answer_correct(predicted, ground_truths[i], question_type),
                    "parse_success": parse_success,
                    "raw_response": s1_responses[i],
                    "raw_stage2": s2_responses[i],
                    "method": f"verbalized_{variant}",
                    "variant": variant,
                })
        else:
            # All single-stage variants: vanilla, vanilla_cot, punish, top_k, linguistic
            prompts = [VERBALIZED_PROMPTS[variant].format(question=q)
                       for q in questions]
            inputs = format_prompts_batch(processor, model_config, images, prompts)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            responses = generate_batch(
                model, processor, inputs, max_new_tokens, do_sample, temperature
            )

            for i in range(len(batch_samples)):
                predicted, confidence = _parse_variant_response(variant, responses[i], question_type)
                if predicted is None:
                    predicted = "unknown"
                parse_success = confidence is not None
                if confidence is None:
                    confidence = 0.5
                detailed_results.append({
                    "dataset_index": batch_start + i,
                    "question": questions[i],
                    "ground_truth": ground_truths[i],
                    "predicted": predicted,
                    "confidence": confidence,
                    "is_correct": check_answer_correct(predicted, ground_truths[i], question_type),
                    "parse_success": parse_success,
                    "raw_response": responses[i],
                    "method": f"verbalized_{variant}",
                    "variant": variant,
                })

        # Progress print
        if batch_end % 20 <= batch_size or batch_end == n:
            n_parsed = sum(1 for r in detailed_results if r["parse_success"])
            print(f"  [{batch_end}/{n}] parse_rate={n_parsed/batch_end:.2%}")

    return detailed_results


def save_verbalized_results(detailed_results, output_dir, config):
    """Save verbalized results."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(detailed_results, f, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    parse_rate = sum(1 for r in detailed_results if r["parse_success"]) / len(detailed_results) if detailed_results else 0
    n_correct = sum(1 for r in detailed_results if r["is_correct"])
    print(f"  Saved {len(detailed_results)} results to {output_dir}")
    print(f"  Acc={n_correct/len(detailed_results):.4f} parse_rate={parse_rate:.2%}" if detailed_results else "")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Verbalized confidence generation")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--dataset", type=str, default="all",
                        choices=DATASETS + ["all"])
    parser.add_argument("--variant", type=str, default="all",
                        choices=VERBALIZED_VARIANTS + ["all"])
    parser.add_argument("--question-type", type=str, default="closed",
                        choices=QUESTION_TYPES,
                        help="Question type to evaluate (closed, open)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for generation (higher = faster, uses more VRAM)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (faster on high-VRAM GPUs)")
    parser.add_argument("--use-8bit", action="store_true",
                        help="Use 8-bit quantization instead of 4-bit")
    parser.add_argument("--subsample-size", type=int, default=None,
                        help="Cap dataset to N examples (smoke testing).")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    models = list(MODELS.keys()) if args.model == "all" else [args.model]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    variants = VERBALIZED_VARIANTS if args.variant == "all" else [args.variant]
    question_type = args.question_type

    # Output dir suffix for non-closed question types
    qt_suffix = f"_{question_type}" if question_type != "closed" else ""

    # Build job list
    jobs = []
    for model_key in models:
        for dataset in datasets:
            for variant in variants:
                output_dir = VERBALIZED_DIR / variant / f"{model_key}_{dataset}{qt_suffix}"
                if not args.force and results_exist(output_dir):
                    print(f"  [SKIP] {variant}/{model_key}_{dataset}")
                    continue
                jobs.append((model_key, dataset, variant, str(output_dir)))

    if not jobs:
        print("No jobs to run (all results exist)")
        return

    print(f"\n{len(jobs)} jobs to run:")
    for model_key, dataset, variant, output_dir in jobs:
        print(f"  {variant}/{model_key}_{dataset}")

    if args.dry_run:
        print("\n[DRY-RUN] No models loaded, no generation performed.")
        return

    # Group jobs by model to avoid reloading
    from collections import defaultdict
    jobs_by_model = defaultdict(list)
    for j in jobs:
        jobs_by_model[j[0]].append(j)

    for model_key, model_jobs in jobs_by_model.items():
        model_id = MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"Loading model: {model_id}")
        print(f"{'='*60}")

        model_config = ModelConfig(model_id=model_id, use_4bit=not args.no_4bit, use_8bit=args.use_8bit)
        model, processor = smart_load_model(model_config, adapter_path=None)
        model.eval()

        # Cache loaded datasets
        dataset_cache = {}

        for model_key, dataset_name, variant, output_dir in model_jobs:
            print(f"\n--- {variant}/{model_key}_{dataset_name} ---")

            # Load dataset (cache across variants for same model+dataset)
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
            print(f"  Dataset: {len(dataset)} {question_type} questions")

            detailed = evaluate_verbalized(
                model, processor, model_config, dataset, variant, output_dir,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                question_type=question_type,
            )

            config = {
                "model_id": model_id,
                "model_key": model_key,
                "dataset": dataset_name,
                "variant": variant,
                "question_type": question_type,
                "method": f"verbalized_{variant}",
                "max_new_tokens": args.max_new_tokens,
                "num_questions": len(dataset),
                "seed": SEED,
            }

            save_verbalized_results(detailed, output_dir, config)

        # Free GPU memory before loading next model
        del model, processor
        torch.cuda.empty_cache()

    print("\nAll verbalized generation complete.")


if __name__ == "__main__":
    main()
