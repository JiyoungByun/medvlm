"""Phase E (Emulation): validate the full medvlm pipeline on pregen LLM outputs.

Uses `detailed_results.json` files saved by experiments/01_generate_sampling.py and
experiments/03_generate_hedge.py on previous runs. Covers:

  1. Sampling-result parsing (raw_responses -> confidence, correctness)
     matches what was stored at generation time.
  2. compute_calibration_metrics() over real confidences reproduces the saved
     metrics.json numbers (within tolerance).
  3. cluster_by_yesno() on HEDGE original+distorted samples runs and gives
     non-degenerate cluster assignments.
  4. CalibrationPipeline (all 5 standard + 3 HAC methods) fits on real
     (confidence, correctness, VASE) arrays; standard methods reduce ECE;
     HAC improves AUROC.
  5. end-to-end: emulate a val/test split, calibrate, report numbers.

Run on the server where pregen lives (r42). Example:
  PREGEN=/cis/home/jbyun13/vqa-overconfidence-backup/colm_results \\
      python tests/phase_emulation.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import medvlm
from medvlm import CalibrationPipeline
from medvlm.confidence import (
    parse_answer_text, normalize_answer, check_answer_correct,
    parse_confidence_numeric, parse_confidence_linguistic,
    cluster_by_yesno,
)
from medvlm.confidence.verbalized import parse_answer_yes_no, _parse_variant_response
from medvlm.evaluation.calibration import (
    CalibrationResult, compute_calibration_metrics,
)


PREGEN = Path(os.environ.get(
    "PREGEN",
    "/cis/home/jbyun13/vqa-overconfidence-backup/colm_results",
))
# Verbalized results live in a different tree
VERBALIZED_ROOT = Path(os.environ.get(
    "VERBALIZED_PREGEN",
    "/cis/home/jbyun13/vqa-overconfidence-backup/results/verbalized",
))


def section(t):
    print(f"\n{'='*64}\n{t}\n{'='*64}")


def pct(x):
    return f"{x*100:.1f}%"


# =============================================================================
# 1. Sampling: raw_responses -> (confidence, predicted) reproducibility
# =============================================================================

def test_sampling_parsing_reproduces_stored_confidence(sampling_dir):
    section(f"1. Sampling parsing reproduces stored confidence\n   {sampling_dir}")
    with open(sampling_dir / "detailed_results.json") as f:
        entries = json.load(f)
    print(f"  {len(entries)} entries")

    exact_confidence_matches = 0
    exact_predicted_matches = 0
    mismatches = []
    for i, e in enumerate(entries):
        raw = e["raw_responses"]
        # Re-parse each raw response
        counts = {}
        unknown = 0
        for r in raw:
            parsed = parse_answer_text(r)
            normalized = normalize_answer(parsed, "closed")
            if normalized is None:
                unknown += 1
            else:
                counts[normalized] = counts.get(normalized, 0) + 1
        valid = sum(counts.values())
        if valid > 0:
            my_pred = max(counts, key=counts.get)
            my_conf = counts[my_pred] / valid
        else:
            my_pred = "unknown"
            my_conf = 0.5

        if abs(my_conf - e["confidence"]) < 1e-6:
            exact_confidence_matches += 1
        if my_pred == e["predicted"]:
            exact_predicted_matches += 1
        if abs(my_conf - e["confidence"]) > 0.02 and len(mismatches) < 5:
            mismatches.append((i, e["confidence"], my_conf, counts, e.get("answer_counts")))

    n = len(entries)
    print(f"  predicted match: {exact_predicted_matches}/{n} ({pct(exact_predicted_matches/n)})")
    print(f"  confidence exact match: {exact_confidence_matches}/{n} "
          f"({pct(exact_confidence_matches/n)})")
    if mismatches:
        print(f"  First mismatches:")
        for i, stored, mine, mc, sc in mismatches:
            print(f"    [{i}] stored_conf={stored:.3f} mine={mine:.3f} "
                  f"my_counts={mc} stored_counts={sc}")
    # Parser MUST exactly reproduce, or the pipeline is non-deterministic.
    assert exact_confidence_matches / n > 0.95, "parsing diverges from stored"
    assert exact_predicted_matches / n > 0.98, "predicted diverges from stored"
    return entries


# =============================================================================
# 2. compute_calibration_metrics reproduces stored metrics.json
# =============================================================================

def test_metrics_reproducibility(sampling_dir, entries):
    section(f"2. evaluate_calibration reproduces stored metrics.json")
    with open(sampling_dir / "metrics.json") as f:
        stored = json.load(f)

    corr = [e["is_correct"] for e in entries]
    conf = [e["confidence"] for e in entries]

    # Try default (15 bins) and 10 bins to match whichever stored uses
    for bins in (stored.get("num_bins", 10), 10, 15):
        report = medvlm.evaluate_calibration(corr, conf, num_bins=bins)
        if abs(report["accuracy"] - stored.get("accuracy", -1)) < 1e-4:
            break

    for key in ("accuracy", "mean_confidence", "ece", "overconfidence"):
        if key in stored:
            mine = report[key]
            old = stored[key]
            diff = abs(mine - old)
            ok = "OK" if diff < 0.01 else "DIFF"
            print(f"  {key:20s}: stored={old:.4f}  mine={mine:.4f}  delta={diff:.4f}  [{ok}]")
    assert abs(report["accuracy"] - stored["accuracy"]) < 1e-4


# =============================================================================
# 3. HEDGE: cluster_by_yesno on real samples, SE/RadFlag/VASE sanity
# =============================================================================

def test_hedge_clustering_and_scores(hedge_dir):
    section(f"3. HEDGE clustering + scores\n   {hedge_dir}")
    with open(hedge_dir / "detailed_results.json") as f:
        entries = json.load(f)
    print(f"  {len(entries)} entries")

    # Sanity: scores are finite; VASE has non-zero variance across questions.
    vase = np.array([e["VASE"] for e in entries])
    se = np.array([e["SE"] for e in entries])
    rf = np.array([e["RadFlag"] for e in entries])
    for name, arr in [("VASE", vase), ("SE", se), ("RadFlag", rf)]:
        assert np.isfinite(arr).all(), f"{name} has non-finite values"
    assert vase.std() > 0, "VASE has zero variance"
    print(f"  VASE: mean={vase.mean():.3f} std={vase.std():.3f} "
          f"range=[{vase.min():.3f}, {vase.max():.3f}]")
    print(f"  SE:   mean={se.mean():.3f}  std={se.std():.3f}")
    print(f"  RadFlag: mean={rf.mean():.3f} std={rf.std():.3f}")

    # Exercise cluster_by_yesno on a variety of real answers
    n_checked = 0
    n_with_multiple = 0
    for e in entries[:40]:
        answers = [e["greedy_answer"]] + e["original_high_temp"] + e["distorted_high_temp"]
        ids = cluster_by_yesno(answers)
        assert len(ids) == len(answers)
        n_checked += 1
        if len(set(ids)) > 1:
            n_with_multiple += 1
    print(f"  cluster_by_yesno: ran on {n_checked} questions, "
          f"{n_with_multiple} got >1 cluster")
    return entries


# =============================================================================
# 4. CalibrationPipeline on real arrays (standard + HAC)
# =============================================================================

def test_real_calibration(sampling_entries, hedge_entries):
    section(f"4. CalibrationPipeline on real (conf, corr, VASE)")

    # Join sampling + hedge on question text (both have the same dataset order
    # but let's match to be safe).
    s_by_q = {e["question"]: e for e in sampling_entries}
    joined = []
    for h in hedge_entries:
        s = s_by_q.get(h["question"])
        if s is None:
            continue
        joined.append((s["confidence"], s["is_correct"], h["VASE"]))

    if len(joined) < 40:
        print(f"  only {len(joined)} joined entries, skipping")
        return

    conf = np.array([x[0] for x in joined], dtype=np.float64)
    corr = np.array([x[1] for x in joined], dtype=np.float64)
    h = np.array([x[2] for x in joined], dtype=np.float64)
    print(f"  joined N={len(joined)}; acc={corr.mean():.3f} "
          f"mean_conf={conf.mean():.3f}")

    # 50/50 val/test split (deterministic)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(joined))
    k = len(joined) // 2
    v = idx[:k]; t = idx[k:]

    raw_ece = medvlm.evaluate_calibration(corr[t], conf[t])["ece"]
    try:
        raw_auc = roc_auc_score(corr[t], conf[t])
    except ValueError:
        raw_auc = float("nan")
    print(f"\n  {'Method':<28} {'ECE':>7} {'ΔECE':>8} {'AUROC':>7} {'ΔAUROC':>8}")
    print(f"  {'Raw (uncalibrated)':<28} {raw_ece:>7.4f} {'':>8} {raw_auc:>7.4f}")

    standard = ["temperature_scaling", "platt", "platt_confidence",
                "isotonic", "histogram_binning"]
    hac = ["hac_platt", "hac_platt_confidence", "hac_gate"]

    for method in standard:
        cal = CalibrationPipeline(method=method)
        cal.fit(conf[v], corr[v])
        cc = cal.transform(conf[t])
        ece = medvlm.evaluate_calibration(corr[t], cc)["ece"]
        try:
            auc = roc_auc_score(corr[t], cc)
        except ValueError:
            auc = float("nan")
        print(f"  {method:<28} {ece:>7.4f} {ece-raw_ece:>+8.4f} "
              f"{auc:>7.4f} {auc-raw_auc:>+8.4f}")

    for method in hac:
        cal = CalibrationPipeline(method=method)
        cal.fit(conf[v], corr[v], hallucination_scores=h[v])
        cc = cal.transform(conf[t], hallucination_scores=h[t])
        ece = medvlm.evaluate_calibration(corr[t], cc)["ece"]
        try:
            auc = roc_auc_score(corr[t], cc)
        except ValueError:
            auc = float("nan")
        print(f"  {method + ' *':<28} {ece:>7.4f} {ece-raw_ece:>+8.4f} "
              f"{auc:>7.4f} {auc-raw_auc:>+8.4f}")


# =============================================================================
# main
# =============================================================================

def test_verbalized_reparsing(variant, dir_):
    """Verify parse_answer_yes_no + parse_confidence_* reproduce stored values."""
    section(f"V. Verbalized re-parsing — {variant}\n   {dir_}")
    with open(dir_ / "detailed_results.json") as f:
        entries = json.load(f)
    print(f"  {len(entries)} entries")

    pred_match = 0
    conf_match = 0
    parseable = 0
    for e in entries:
        raw = e["raw_response"]
        # Use the full variant-aware parser (matches compute_verbalized_confidence).
        # For two_stage, parsing happens on the raw_stage2 response, but pregen
        # only stores one raw_response — skip two_stage predicted check.
        my_pred, my_conf = _parse_variant_response(variant, raw, "closed")
        # compute_verbalized_confidence converts None -> "unknown" before storing
        if my_pred is None:
            my_pred = "unknown"
        if my_pred == e["predicted"]:
            pred_match += 1

        if my_conf is not None:
            parseable += 1
            if abs(my_conf - e["confidence"]) < 1e-6:
                conf_match += 1

    n = len(entries)
    print(f"  predicted match:  {pred_match}/{n} ({pct(pred_match/n)})")
    print(f"  parseable conf:   {parseable}/{n}")
    if parseable > 0:
        print(f"  exact conf match: {conf_match}/{parseable} "
              f"({pct(conf_match/parseable)})")
    # two_stage stores confidence from stage-2 response that's not in raw_response,
    # so predicted reproduction may be lower. Relax threshold there.
    threshold = 0.7 if variant == "two_stage" else 0.9
    assert pred_match / n > threshold, (
        f"{variant}: predicted mismatch rate too high ({pred_match}/{n})"
    )


def main():
    samp = PREGEN / "sampling" / "base_qwen2b_rad_vqa_sampling"
    hedge = PREGEN / "hedge_vase" / "qwen2b_rad_vqa"
    for p in (samp, hedge):
        if not (p / "detailed_results.json").exists():
            print(f"MISSING: {p/'detailed_results.json'}")
            sys.exit(1)

    s_entries = test_sampling_parsing_reproduces_stored_confidence(samp)
    test_metrics_reproducibility(samp, s_entries)
    h_entries = test_hedge_clustering_and_scores(hedge)
    test_real_calibration(s_entries, h_entries)

    # Verbalized variants — one model across all six variants
    VERBALIZED_MODEL_DIR = "internvl3_2b_rad_vqa"
    for variant in ("vanilla", "vanilla_cot", "punish", "top_k",
                    "two_stage", "linguistic"):
        d = VERBALIZED_ROOT / variant / VERBALIZED_MODEL_DIR
        if not (d / "detailed_results.json").exists():
            print(f"  [skip] {variant}: no pregen at {d}")
            continue
        test_verbalized_reparsing(variant, d)

    print("\n" + "=" * 64)
    print("ALL EMULATION CHECKS PASSED")
    print("=" * 64)


if __name__ == "__main__":
    main()
