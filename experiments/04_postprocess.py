#!/usr/bin/env python3
"""
Post-processing: LLM-as-a-Judge re-scoring + semantic clustering.

Processes open results that need semantic matching:
  1. Verbalized results → detailed_results_judged.json
  2. Sampling results → detailed_results_clustered.json

Uses a small LLM judge (Qwen3-4B) for:
  - Verbalized: correctness re-scoring (tight string match + LLM fallback)
  - Sampling: pairwise equivalence clustering + correctness judgment

Usage:
    python experiments/04_postprocess.py --gpu 0
    python experiments/04_postprocess.py --gpu 0 --force
    python experiments/04_postprocess.py --gpu 0 --verbalized-only
    python experiments/04_postprocess.py --gpu 0 --sampling-only
    python experiments/04_postprocess.py --dry-run
"""

import argparse
import json
import os
import re
import sys
import fcntl
from pathlib import Path
from collections import defaultdict

# Parse GPU before importing torch
_parser_gpu = argparse.ArgumentParser(add_help=False)
_parser_gpu.add_argument("--gpu", type=str, default=None)
_args_gpu, _ = _parser_gpu.parse_known_args()
if _args_gpu.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _args_gpu.gpu
    print(f"[GPU] Set CUDA_VISIBLE_DEVICES={_args_gpu.gpu}")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medvlm.configs import JUDGE_MODEL_ID
from experiments.config import COLM_RESULTS, VERBALIZED_DIR, SAMPLING_DIR


# =============================================================================
# Concurrent-Safe Cache I/O
# =============================================================================

def load_cache_safe(cache_path):
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        return data
    except (json.JSONDecodeError, IOError):
        return {}


def save_cache_safe(local_cache, cache_path, compact=False):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    disk_cache = load_cache_safe(cache_path)
    disk_cache.update(local_cache)
    indent = None if (compact or len(disk_cache) > 100_000) else 2
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(disk_cache, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)
        os.rename(tmp_path, cache_path)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        with open(cache_path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(disk_cache, f, indent=indent)
            fcntl.flock(f, fcntl.LOCK_UN)
    return disk_cache


# =============================================================================
# Cache Key Helpers
# =============================================================================

def make_equiv_key(question, answer_a, answer_b):
    a, b = sorted([answer_a.strip(), answer_b.strip()])
    return f"{question.strip()}|||{a}|||{b}"


def make_judge_key(question, ground_truth, predicted):
    return f"{question}|||{ground_truth}|||{predicted}"


# =============================================================================
# String Normalization
# =============================================================================

def normalize_for_matching(text):
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r'^(the|a|an)\s+', '', text)
    text = re.sub(r'[.,;:!?\'"()\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tight_match(predicted, ground_truth):
    if predicted is None or ground_truth is None:
        return False
    p = normalize_for_matching(predicted)
    g = normalize_for_matching(ground_truth)
    if not p or not g:
        return False
    if p == g:
        return True
    if len(p) >= 3 and len(g) >= 3:
        if g in p or p in g:
            return True
    return False


# =============================================================================
# LLM Judge
# =============================================================================

JUDGE_PROMPT_TEMPLATE = (
    "Given a medical VQA question, determine if the predicted answer is "
    "semantically equivalent to the ground truth.\n\n"
    "Question: {question}\n"
    "Ground truth: {ground_truth}\n"
    "Predicted: {predicted}\n\n"
    "Are these semantically equivalent? Reply with exactly one word: yes or no."
)

EQUIV_PROMPT_TEMPLATE = (
    "In the context of a medical VQA question, determine if two answers "
    "are semantically equivalent (same medical meaning, even if worded differently).\n\n"
    "Question: {question}\n"
    "Answer A: {answer_a}\n"
    "Answer B: {answer_b}\n\n"
    "Are these semantically equivalent? Reply with exactly one word: yes or no."
)


def load_judge_model(model_id=JUDGE_MODEL_ID):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading judge model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Judge model loaded")
    return model, tokenizer


def judge_batch_generic(model, tokenizer, prompts_and_keys, batch_size=32):
    import torch

    results = {}
    for batch_start in range(0, len(prompts_and_keys), batch_size):
        batch = prompts_and_keys[batch_start:batch_start + batch_size]

        texts = []
        for prompt_text, _ in batch:
            messages = [{"role": "user", "content": prompt_text}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(text)

        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        tokenizer.padding_side = orig_side
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for i, (_, key) in enumerate(batch):
            generated = outputs[i, prompt_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip().lower()
            results[key] = response.startswith("yes")

    return results


# =============================================================================
# Part 1: Verbalized Re-scoring
# =============================================================================

def find_verbalized_dirs():
    result_dirs = []
    if not VERBALIZED_DIR.exists():
        return result_dirs
    for root, dirs, files in os.walk(str(VERBALIZED_DIR)):
        if "detailed_results.json" in files:
            dir_path = Path(root)
            if "_archive" in str(dir_path) or "_backup" in dir_path.name:
                continue
            config_path = dir_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                qt = config.get("question_type", "closed")
                if qt == "open":
                    result_dirs.append(dir_path)
                    continue
            if "_open" in dir_path.name:
                result_dirs.append(dir_path)
    return sorted(result_dirs)


def rescore_verbalized(detailed_results, judge_model, judge_tokenizer, judge_cache,
                       batch_size=32):
    updated = []
    stats = {"total": 0, "tight_match": 0, "judge_correct": 0,
             "judge_incorrect": 0, "unparseable": 0}

    needs_judge = []
    for i, r in enumerate(detailed_results):
        stats["total"] += 1
        entry = dict(r)
        entry["is_correct_original"] = r.get("is_correct", False)

        if not r.get("parse_success", True):
            entry["confidence"] = 0.0
            entry["is_correct"] = False
            entry["match_method"] = "unparseable"
            stats["unparseable"] += 1
            updated.append(entry)
            continue

        predicted = r.get("predicted", "")
        ground_truth = r.get("ground_truth", "")

        if tight_match(predicted, ground_truth):
            stats["tight_match"] += 1
            entry["is_correct"] = True
            entry["match_method"] = "tight_string"
            updated.append(entry)
        else:
            cache_key = make_judge_key(r.get("question", ""), ground_truth, predicted)
            if cache_key in judge_cache:
                entry["is_correct"] = judge_cache[cache_key]
                entry["match_method"] = "judge_cached"
                updated.append(entry)
            else:
                needs_judge.append((i, r, cache_key))
                updated.append(None)

    if needs_judge and judge_model is not None:
        print(f"  Running LLM judge on {len(needs_judge)} verbalized entries...")
        prompts_and_keys = []
        for _, r, ck in needs_judge:
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=r.get("question", ""),
                ground_truth=r.get("ground_truth", ""),
                predicted=r.get("predicted", ""),
            )
            prompts_and_keys.append((prompt, ck))

        judge_results = judge_batch_generic(
            judge_model, judge_tokenizer, prompts_and_keys, batch_size
        )

        for idx, r, cache_key in needs_judge:
            is_match = judge_results.get(cache_key, False)
            judge_cache[cache_key] = is_match
            entry = dict(r)
            entry["is_correct_original"] = r.get("is_correct", False)
            entry["is_correct"] = is_match
            entry["match_method"] = "judge"
            updated[idx] = entry
            if is_match:
                stats["judge_correct"] += 1
            else:
                stats["judge_incorrect"] += 1
    elif needs_judge:
        for idx, r, _ in needs_judge:
            entry = dict(r)
            entry["is_correct_original"] = r.get("is_correct", False)
            entry["match_method"] = "original"
            updated[idx] = entry

    return updated, stats


# =============================================================================
# Part 2: Sampling Clustering
# =============================================================================

def find_sampling_dirs():
    result_dirs = []
    if not SAMPLING_DIR.exists():
        return result_dirs
    for d in sorted(SAMPLING_DIR.iterdir()):
        if not d.is_dir() or "_archive" in d.name or "_chunk_" in d.name:
            continue
        if "_open" not in d.name:
            continue
        if (d / "detailed_results.json").exists():
            result_dirs.append(d)
    return result_dirs


def pre_cluster_by_string(unique_answers):
    n = len(unique_answers)
    norms = [normalize_for_matching(a) for a in unique_answers]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    norm_to_idx = {}
    for i, norm in enumerate(norms):
        if norm in norm_to_idx:
            union(i, norm_to_idx[norm])
        else:
            norm_to_idx[norm] = i

    clusters = defaultdict(set)
    for i in range(n):
        clusters[find(i)].add(i)
    return list(clusters.values())


def cluster_one_question(question, unique_answers, counts, model, tokenizer,
                         equiv_cache, batch_size=32):
    n = len(unique_answers)
    if n <= 1:
        return [(unique_answers[0], counts[0], unique_answers)] if n == 1 else []

    pre_clusters = pre_cluster_by_string(unique_answers)
    reps = []
    for cluster in pre_clusters:
        best_idx = max(cluster, key=lambda i: counts[i])
        total = sum(counts[i] for i in cluster)
        reps.append((unique_answers[best_idx], total, cluster))

    if len(reps) <= 1:
        members = [unique_answers[i] for i in reps[0][2]]
        return [(reps[0][0], reps[0][1], members)]

    K = len(reps)
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    uncached_pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            ekey = make_equiv_key(question, reps[i][0], reps[j][0])
            if ekey in equiv_cache:
                if equiv_cache[ekey]:
                    union(i, j)
            else:
                uncached_pairs.append((i, j, ekey))

    if uncached_pairs:
        prompts_and_keys = []
        for i, j, ekey in uncached_pairs:
            prompt = EQUIV_PROMPT_TEMPLATE.format(
                question=question, answer_a=reps[i][0], answer_b=reps[j][0],
            )
            prompts_and_keys.append((prompt, ekey))

        llm_results = judge_batch_generic(model, tokenizer, prompts_and_keys, batch_size)
        for i, j, ekey in uncached_pairs:
            is_equiv = llm_results.get(ekey, False)
            equiv_cache[ekey] = is_equiv
            if is_equiv:
                union(i, j)

    groups = defaultdict(list)
    for i in range(K):
        groups[find(i)].append(i)

    final = []
    for member_indices in groups.values():
        best = max(member_indices, key=lambda i: reps[i][1])
        total_count = sum(reps[i][1] for i in member_indices)
        all_members = []
        for i in member_indices:
            for orig_idx in reps[i][2]:
                all_members.append(unique_answers[orig_idx])
        final.append((reps[best][0], total_count, all_members))

    final.sort(key=lambda x: -x[1])
    return final


def process_sampling_entry(entry, model, tokenizer, equiv_cache, judge_cache,
                           batch_size=32):
    answer_counts = entry.get("answer_counts", {})
    if not answer_counts:
        return entry

    unique_answers = list(answer_counts.keys())
    counts = [answer_counts[a] for a in unique_answers]
    total_valid = sum(counts)
    unknown_count = entry.get("unknown_count", 0)
    total_n = total_valid + unknown_count

    if total_n == 0:
        return entry

    question = entry.get("question", "")
    clusters = cluster_one_question(
        question, unique_answers, counts, model, tokenizer, equiv_cache, batch_size
    )

    if not clusters:
        return entry

    top_answer = clusters[0][0]
    top_count = clusters[0][1]
    confidence = top_count / total_n

    ground_truth = entry.get("ground_truth", "")
    jkey = make_judge_key(question, ground_truth, top_answer)

    if jkey in judge_cache:
        is_correct = judge_cache[jkey]
    else:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question, ground_truth=ground_truth, predicted=top_answer,
        )
        result = judge_batch_generic(model, tokenizer, [(prompt, jkey)], batch_size=1)
        is_correct = result.get(jkey, False)
        judge_cache[jkey] = is_correct

    updated = dict(entry)
    updated["predicted"] = top_answer
    updated["confidence"] = confidence
    updated["is_correct"] = is_correct
    updated["clustered_answer_counts"] = {rep: cnt for rep, cnt, _ in clusters}
    updated["num_semantic_clusters"] = len(clusters)
    return updated


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-processing: judge re-scoring + semantic clustering")
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbalized-only", action="store_true")
    parser.add_argument("--sampling-only", action="store_true")
    parser.add_argument("--include", type=str, default=None,
                        help="Regex filter for directory names")
    return parser.parse_args()


def main():
    args = parse_args()

    # Discover directories
    verb_dirs = [] if args.sampling_only else find_verbalized_dirs()
    samp_dirs = [] if args.verbalized_only else find_sampling_dirs()

    if args.include:
        pattern = re.compile(args.include)
        verb_dirs = [d for d in verb_dirs if pattern.search(d.name)]
        samp_dirs = [d for d in samp_dirs if pattern.search(d.name)]

    if not args.force:
        verb_dirs = [d for d in verb_dirs
                     if not (d / "detailed_results_judged.json").exists()]
        samp_dirs = [d for d in samp_dirs
                     if not (d / "detailed_results_clustered.json").exists()]

    total = len(verb_dirs) + len(samp_dirs)
    if total == 0:
        print("Nothing to process (all results exist, use --force to redo).")
        return

    print(f"\n{len(verb_dirs)} verbalized + {len(samp_dirs)} sampling dirs to process")
    for d in verb_dirs:
        print(f"  [verb] {d.relative_to(COLM_RESULTS)}")
    for d in samp_dirs:
        print(f"  [samp] {d.name}")

    if args.dry_run:
        print("\n[DRY-RUN] No models loaded.")
        return

    # Load caches
    judge_cache_path = str(COLM_RESULTS / "judge_cache.json")
    equiv_cache_path = str(COLM_RESULTS / "equiv_cache.json")
    judge_cache = load_cache_safe(judge_cache_path)
    equiv_cache = load_cache_safe(equiv_cache_path)
    print(f"Judge cache: {len(judge_cache)} entries")
    print(f"Equiv cache: {len(equiv_cache)} entries")

    # Load judge model
    import torch
    model, tokenizer = load_judge_model(args.judge_model)

    # --- Part 1: Verbalized re-scoring ---
    for d in verb_dirs:
        rel = d.relative_to(COLM_RESULTS)
        print(f"\n{'='*60}")
        print(f"[Verbalized] {rel}")

        with open(d / "detailed_results.json") as f:
            detailed = json.load(f)

        updated, stats = rescore_verbalized(
            detailed, model, tokenizer, judge_cache, args.batch_size
        )

        with open(d / "detailed_results_judged.json", "w") as f:
            json.dump(updated, f, indent=2)

        save_cache_safe(judge_cache, judge_cache_path)

        n_correct = sum(1 for r in updated if r and r.get("is_correct", False))
        print(f"  {stats['total']} entries → {n_correct} correct "
              f"(tight={stats['tight_match']}, judge={stats['judge_correct']})")

    # --- Part 2: Sampling clustering ---
    for d in samp_dirs:
        print(f"\n{'='*60}")
        print(f"[Sampling] {d.name}")

        with open(d / "detailed_results.json") as f:
            detailed = json.load(f)

        updated_results = []
        for i, entry in enumerate(detailed):
            updated = process_sampling_entry(
                entry, model, tokenizer, equiv_cache, judge_cache, args.batch_size
            )
            updated_results.append(updated)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(detailed)}]")

        equiv_cache = save_cache_safe(equiv_cache, equiv_cache_path)
        judge_cache = save_cache_safe(judge_cache, judge_cache_path)

        with open(d / "detailed_results_clustered.json", "w") as f:
            json.dump(updated_results, f, indent=2)

        n_correct = sum(1 for r in updated_results if r.get("is_correct", False))
        print(f"  Done: {n_correct}/{len(detailed)} correct")

    del model, tokenizer
    torch.cuda.empty_cache()

    print(f"\nAll post-processing complete.")
    print(f"Judge cache: {len(judge_cache)}, Equiv cache: {len(equiv_cache)}")


if __name__ == "__main__":
    main()
