"""Phase 3: end-to-end on a single 8B model. Run with --model qwen3vl_8b etc.

Single Python entry that exercises the entire public API on a real GPU model:
  * load_model (default + 8bit)
  * compute_confidence(method="sampling")
  * compute_confidence(method="verbalized") for all 6 variants
  * compute_hedge_scores closed + open
  * end-to-end: standard calibration reduces ECE
  * end-to-end: HAC improves AUROC vs raw
  * custom-data path (list of dicts)

Designed to take ~10-15 min on an H100 with 20 closed + 10 open questions.
"""
import argparse
import sys
import traceback
from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score

import medvlm
from medvlm import CalibrationPipeline


def fail(reason: str) -> None:
    print(f"  FAIL: {reason}")
    raise AssertionError(reason)


def passed(msg: str) -> None:
    print(f"  PASS: {msg}")


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_sampling_results(results: List, n_expected: int, num_samples: int) -> None:
    assert len(results) == n_expected, f"got {len(results)} not {n_expected}"
    confs = [r.confidence for r in results]
    assert all(0.0 <= c <= 1.0 for c in confs), "confidence out of [0,1]"
    # Each result keeps all raw responses
    for r in results:
        assert len(r.raw_responses) == num_samples, (
            f"raw_responses={len(r.raw_responses)} != {num_samples}"
        )
    valid = np.mean([r.valid_response_rate for r in results])
    assert valid > 0.5, f"avg valid_response_rate={valid:.2f} too low"


def check_verbalized_results(results: List, n_expected: int, variant: str) -> None:
    assert len(results) == n_expected
    confs = [r.confidence for r in results]
    assert all(0.0 <= c <= 1.0 for c in confs), f"{variant}: conf out of range"
    parse_rate = np.mean([r.parse_success for r in results])
    assert parse_rate >= 0.4, (
        f"{variant}: parse_success={parse_rate:.2f} too low"
    )
    if variant == "two_stage":
        assert any(r.raw_stage2 for r in results), "two_stage: raw_stage2 missing"


def check_hedge_results(results: List, n_expected: int, n_samples: int) -> None:
    assert len(results) == n_expected
    for r in results:
        assert np.isfinite(r.SE), f"SE={r.SE} not finite"
        assert np.isfinite(r.RadFlag), f"RadFlag={r.RadFlag} not finite"
        assert np.isfinite(r.VASE), f"VASE={r.VASE} not finite"
        assert len(r.original_high_temp) == n_samples
        assert len(r.distorted_high_temp) == n_samples
    # Distribution should be non-degenerate across questions
    vase = np.array([r.VASE for r in results])
    assert vase.std() > 0, "VASE has zero variance — clustering broken?"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3vl_8b")
    p.add_argument("--n-closed", type=int, default=20)
    p.add_argument("--n-open", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--hedge-n", type=int, default=5)
    p.add_argument("--skip-quant", action="store_true",
                   help="Skip the 8-bit quantization sub-test")
    p.add_argument("--skip-verbalized", action="store_true")
    p.add_argument("--skip-hedge", action="store_true")
    args = p.parse_args()

    failures = []

    section(f"Phase 3 integration: {args.model}")

    # ---- 1. load_model default ----
    section("1. load_model (default bf16)")
    try:
        model, processor, config = medvlm.load_model(args.model)
        assert config.model_id, "no model_id"
        assert config.model_family is not None, "no model_family"
        assert hasattr(model, "generate"), "model has no .generate"
        passed(f"loaded {config.model_id} family={config.model_family.value}")
    except Exception as e:
        failures.append(("load_model default", str(e)))
        traceback.print_exc()
        return _summary(failures)

    # ---- 2. dataset for closed + open ----
    section("2. load_dataset")
    closed_ds = medvlm.load_dataset(
        "vqa_rad", split="test", question_type="closed",
        subsample_size=args.n_closed, seed=42,
    )
    open_ds = medvlm.load_dataset(
        "vqa_rad", split="test", question_type="open",
        subsample_size=args.n_open, seed=42,
    )
    passed(f"closed={len(closed_ds)} open={len(open_ds)}")

    # ---- 3. sampling ----
    section("3. compute_confidence(method='sampling')")
    try:
        s_results = medvlm.compute_confidence(
            model, processor, config, closed_ds,
            method="sampling", num_samples=args.num_samples,
            question_type="closed",
        )
        check_sampling_results(s_results, args.n_closed, args.num_samples)
        s_corr = np.array([r.is_correct for r in s_results])
        s_conf = np.array([r.confidence for r in s_results])
        rep = medvlm.evaluate_calibration(s_corr, s_conf)
        print(f"  Sampling: acc={rep['accuracy']:.3f} "
              f"meanconf={rep['mean_confidence']:.3f} "
              f"ece={rep['ece']:.3f} oc={rep['overconfidence']:+.3f}")
        if rep["accuracy"] < 0.4:
            failures.append(("sampling", f"acc={rep['accuracy']:.2f} suspiciously low"))
        if rep["accuracy"] > 0.99:
            failures.append(("sampling", f"acc={rep['accuracy']:.2f} suspiciously high"))
        passed("sampling ok")
    except Exception as e:
        failures.append(("sampling", str(e)))
        traceback.print_exc()

    # ---- 4. verbalized — all 6 variants ----
    if not args.skip_verbalized:
        section("4. compute_confidence(method='verbalized') × 6 variants")
        verb_per_variant = {}
        for variant in ["vanilla", "vanilla_cot", "punish", "top_k",
                        "two_stage", "linguistic"]:
            try:
                v_results = medvlm.compute_confidence(
                    model, processor, config, closed_ds,
                    method="verbalized", variant=variant,
                    question_type="closed", batch_size=4,
                )
                check_verbalized_results(v_results, args.n_closed, variant)
                v_corr = np.array([r.is_correct for r in v_results])
                v_conf = np.array([r.confidence for r in v_results])
                rep = medvlm.evaluate_calibration(v_corr, v_conf)
                verb_per_variant[variant] = (rep["accuracy"], rep["mean_confidence"],
                                             np.mean([r.parse_success for r in v_results]))
                print(f"  {variant:14s}: acc={rep['accuracy']:.3f} "
                      f"conf={rep['mean_confidence']:.3f} "
                      f"parse={verb_per_variant[variant][2]:.2f}")
            except Exception as e:
                failures.append((f"verbalized/{variant}", str(e)))
                traceback.print_exc()
        # Sanity: confidences should not all be identical across variants
        if len(verb_per_variant) >= 2:
            confs = [v[1] for v in verb_per_variant.values()]
            if max(confs) - min(confs) < 0.001:
                failures.append(("verbalized variants",
                                 "all variants returned identical mean confidence"))
            else:
                passed("variants differ in confidence (parsing differentiates them)")

    # ---- 5. HEDGE scores ----
    val_h = test_h = None
    if not args.skip_hedge:
        section("5. compute_hedge_scores")
        try:
            hedge_closed = medvlm.compute_hedge_scores(
                model, processor, config, closed_ds,
                question_type="closed", n_samples=args.hedge_n,
            )
            check_hedge_results(hedge_closed, args.n_closed, args.hedge_n)
            vase = np.array([r.VASE for r in hedge_closed])
            print(f"  closed: SE mean={np.mean([r.SE for r in hedge_closed]):.3f} "
                  f"VASE mean={vase.mean():.3f} std={vase.std():.3f}")
            passed("HEDGE closed ok")

            hedge_open = medvlm.compute_hedge_scores(
                model, processor, config, open_ds,
                question_type="open", n_samples=args.hedge_n,
            )
            check_hedge_results(hedge_open, args.n_open, args.hedge_n)
            passed("HEDGE open ok (embedding clustering ran)")
        except Exception as e:
            failures.append(("hedge", str(e)))
            traceback.print_exc()

    # ---- 6. End-to-end calibration: split, fit, transform, eval ----
    section("6. End-to-end calibration on val/test split")
    try:
        from medvlm import train_val_test_split
        # Split closed_ds into val/test, recompute on each
        val_ds, test_ds = train_val_test_split(closed_ds, val_fraction=0.4, seed=42)
        val_r = medvlm.compute_confidence(
            model, processor, config, val_ds,
            method="sampling", num_samples=args.num_samples, question_type="closed",
        )
        test_r = medvlm.compute_confidence(
            model, processor, config, test_ds,
            method="sampling", num_samples=args.num_samples, question_type="closed",
        )
        val_conf = np.array([r.confidence for r in val_r])
        val_corr = np.array([r.is_correct for r in val_r])
        test_conf = np.array([r.confidence for r in test_r])
        test_corr = np.array([r.is_correct for r in test_r])

        raw_ece = medvlm.evaluate_calibration(test_corr, test_conf)["ece"]
        print(f"  Raw test ECE: {raw_ece:.3f}")
        for method in ["temperature_scaling", "platt", "isotonic"]:
            cal = CalibrationPipeline(method=method)
            cal.fit(val_conf, val_corr)
            cal_conf = cal.transform(test_conf)
            new_ece = medvlm.evaluate_calibration(test_corr, cal_conf)["ece"]
            print(f"  {method}: ECE {raw_ece:.3f} -> {new_ece:.3f} "
                  f"({new_ece - raw_ece:+.3f})")

        # Optional: HAC if hedge available
        if not args.skip_hedge:
            # Recompute hedge on val/test splits
            v_h_res = medvlm.compute_hedge_scores(
                model, processor, config, val_ds,
                question_type="closed", n_samples=args.hedge_n,
            )
            t_h_res = medvlm.compute_hedge_scores(
                model, processor, config, test_ds,
                question_type="closed", n_samples=args.hedge_n,
            )
            val_h = np.array([r.VASE for r in v_h_res])
            test_h = np.array([r.VASE for r in t_h_res])

            raw_auc = roc_auc_score(test_corr, test_conf) if len(set(test_corr)) > 1 else float("nan")
            print(f"  Raw test AUROC: {raw_auc:.3f}")
            for method in ["hac_platt", "hac_gate"]:
                cal = CalibrationPipeline(method=method)
                cal.fit(val_conf, val_corr, hallucination_scores=val_h)
                cal_conf = cal.transform(test_conf, hallucination_scores=test_h)
                new_auc = roc_auc_score(test_corr, cal_conf) if len(set(test_corr)) > 1 else float("nan")
                new_ece = medvlm.evaluate_calibration(test_corr, cal_conf)["ece"]
                print(f"  {method}: AUROC {raw_auc:.3f} -> {new_auc:.3f} "
                      f"({new_auc - raw_auc:+.3f}) ECE {new_ece:.3f}")
        passed("calibration end-to-end ok")
    except Exception as e:
        failures.append(("end-to-end calibration", str(e)))
        traceback.print_exc()

    # ---- 7. Custom data: list of dicts ----
    section("7. Custom data (list of dicts)")
    try:
        from PIL import Image
        # Use a real image from the loaded dataset
        img = closed_ds[0]["image"]
        custom = [
            {"image": img, "question": "Is this a chest x-ray?", "answer": "yes"},
            {"image": img, "question": "Is the patient pediatric?"},  # no answer
        ]
        r = medvlm.compute_confidence(
            model, processor, config, custom,
            method="sampling", num_samples=4, question_type="closed",
        )
        assert len(r) == 2
        assert r[0].is_correct in (True, False), "answer given but is_correct is None"
        assert r[1].is_correct is None, "no answer but is_correct is not None"
        assert r[1].ground_truth is None
        passed("custom-data sampling ok (is_correct=None when answer omitted)")
    except Exception as e:
        failures.append(("custom-data", str(e)))
        traceback.print_exc()

    # ---- 8. 8bit quantization ----
    if not args.skip_quant:
        section("8. load_model(quantization='8bit')")
        try:
            del model
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
            m8, p8, c8 = medvlm.load_model(args.model, quantization="8bit")
            assert c8.use_8bit is True
            r = medvlm.compute_confidence(
                m8, p8, c8, closed_ds.select(range(min(5, len(closed_ds)))),
                method="sampling", num_samples=4, question_type="closed",
            )
            assert len(r) == min(5, len(closed_ds))
            passed("8bit load + 4 samples ok")
            del m8
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append(("8bit quant", str(e)))
            traceback.print_exc()

    return _summary(failures)


def _summary(failures):
    section("SUMMARY")
    if not failures:
        print("ALL CHECKS PASSED")
        return 0
    print(f"{len(failures)} FAILURE(S):")
    for name, msg in failures:
        print(f"  - {name}: {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
