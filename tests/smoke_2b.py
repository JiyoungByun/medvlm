"""Minimal 2B inference smoke: 5 closed questions, verify load + generate works.

Confirms that `medvlm.load_model` + `compute_confidence(method='sampling')`
produces sensible output on a real GPU. After this passes, skip full-inference
phases and use phase_emulation.py instead.

Usage:  python tests/smoke_2b.py --model qwen3vl_2b
"""
import argparse
import sys

import numpy as np
import medvlm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3vl_2b")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=4)
    args = p.parse_args()

    print(f"Loading {args.model}...")
    model, proc, cfg = medvlm.load_model(args.model)
    print(f"  ok: {cfg.model_id} family={cfg.model_family.value}")

    import torch
    mem_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else -1
    print(f"  cuda? {torch.cuda.is_available()} mem_alloc={mem_mb:.0f} MB")

    ds = medvlm.load_dataset(
        "vqa_rad", split="test", question_type="closed",
        subsample_size=args.n, seed=0,
    )
    print(f"  dataset: {len(ds)} questions")

    r = medvlm.compute_confidence(
        model, proc, cfg, ds,
        method="sampling", num_samples=args.num_samples, question_type="closed",
    )

    assert len(r) == args.n
    for i, x in enumerate(r):
        assert 0.0 <= x.confidence <= 1.0, f"conf out of range: {x.confidence}"
        assert len(x.raw_responses) == args.num_samples
        print(f"  [{i}] Q={x.question[:50]!r:<55s} "
              f"gt={x.ground_truth} pred={x.predicted} "
              f"conf={x.confidence:.2f} correct={x.is_correct} "
              f"raw0={x.raw_responses[0][:30]!r}")

    acc = np.mean([x.is_correct for x in r])
    conf = np.mean([x.confidence for x in r])
    print(f"\n  acc={acc:.2f} mean_conf={conf:.2f}  (N={args.n} only — not meaningful)")
    print("SMOKE PASSED — inference works. Proceed with emulation.")


if __name__ == "__main__":
    sys.exit(main())
