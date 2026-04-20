# Example 2: Post-hoc Calibration

**Script:** [`examples/calibration.py`](https://github.com/jiyoungbyun/medvlm/blob/main/examples/calibration.py)

Demonstrates that simple post-hoc calibration methods can substantially reduce ECE on held-out test data without retraining the model.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python examples/calibration.py
CUDA_VISIBLE_DEVICES=0 python examples/calibration.py --model qwen3vl_8b --n-questions 0
```

## What it does

1. Loads VQA-RAD and splits into val (30%) / test (70%)
2. Computes sampling confidence (N=20) on both splits
3. Fits 4 calibration methods on val: temperature scaling, Platt, isotonic, histogram binning
4. Evaluates calibrated ECE and AUROC on test

## Expected output

```
  Method                      ECE   delta   AUROC    delta
  ---------------------------------------------------------
  Raw (uncalibrated)        0.1694           0.5195
  temperature_scaling       0.0252  -0.1442  0.5195  +0.0000
  platt                     0.0537  -0.1157  0.5195  +0.0000
  isotonic                  0.0370  -0.1324  0.5195  +0.0000
  histogram_binning         0.0524  -0.1170  0.5195  +0.0000
```

All methods reduce ECE by 50--85%. AUROC is unchanged because these are monotonic transforms that preserve the ranking of confidence scores.
