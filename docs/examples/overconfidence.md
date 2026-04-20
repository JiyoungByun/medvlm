# Example 1: Overconfidence

**Script:** [`examples/overconfidence.py`](https://github.com/jiyoungbyun/medvlm/blob/main/examples/overconfidence.py)

Demonstrates that medical VLMs are systematically overconfident: their self-reported confidence consistently exceeds their actual accuracy.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python examples/overconfidence.py
CUDA_VISIBLE_DEVICES=0 python examples/overconfidence.py --model qwen3vl_8b
```

## What it does

1. Loads VQA-RAD closed questions (50 by default)
2. Computes **sampling-based** confidence (N=20 samples per question)
3. Computes **verbalized** confidence (vanilla prompt variant)
4. Prints accuracy, mean confidence, and overconfidence gap

## Expected output

```
  Sampling (N=20):
    Accuracy:        0.7600
    Mean Confidence: 0.9400
    Overconfidence:  +0.1940  (conf - acc = +0.1800)
    ECE:             0.2080

  Verbalized (vanilla):
    Accuracy:        0.6800
    Mean Confidence: 0.9500
    Overconfidence:  +0.2700  (conf - acc = +0.2700)
    ECE:             0.2700
```

Both methods show positive overconfidence: the model's confidence exceeds its accuracy by 18--27 percentage points.
