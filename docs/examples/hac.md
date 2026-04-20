# Example 3: HAC (Hallucination-Aware Calibration)

**Script:** [`examples/hac.py`](https://github.com/jiyoungbyun/medvlm/blob/main/examples/hac.py)

Demonstrates that HAC methods can improve AUROC (discrimination between correct and incorrect predictions), which standard post-hoc calibration cannot do.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python examples/hac.py
CUDA_VISIBLE_DEVICES=0 python examples/hac.py --model qwen3vl_8b
```

## What it does

1. Loads VQA-RAD and splits into val (30%) / test (70%)
2. Computes sampling confidence (N=20)
3. Computes **VASE hallucination scores** via `medvlm.compute_hedge_scores` (greedy + N high-temp + N distorted-image samples, clustered)
4. Compares standard Platt/isotonic vs **HAC-Platt**, **HAC-Gate**

## HAC methods

**HAC-Platt:** $\sigma(a \cdot c + b \cdot h + d)$ with $a \geq 0, b \leq 0$

The hallucination score $h$ enters as an additive term with a negative coefficient, so higher hallucination lowers the calibrated confidence. L2 regularization on $b$ prevents overfitting.

**HAC-Gate:** $c \cdot \sigma(-a \cdot h + b)$ with $a \geq 0$

The original confidence is multiplied by a sigmoid gate controlled by $h$. When hallucination is high, the gate closes and confidence is attenuated.

## Why HAC improves AUROC

Standard calibration (Platt, isotonic, temperature scaling) applies monotonic transforms to confidence. This can fix ECE but **cannot change the ranking** of predictions --- so AUROC stays the same.

HAC methods incorporate hallucination scores as a **second input**, which provides independent information about whether a prediction is correct. This allows HAC to re-rank predictions: a high-confidence but high-hallucination prediction gets downweighted, while a moderate-confidence but low-hallucination prediction is preserved. This re-ranking improves AUROC.

## Expected output

```
  Method                      ECE    dECE   AUROC   dAUROC
  ----------------------------------------------------------
  Raw (uncalibrated)        0.1171           0.5832
  platt                     0.1456  +0.0284  0.5832  +0.0000
  isotonic                  0.1714  +0.0543  0.6032  +0.0200
  hac_platt *               0.1363  +0.0192  0.6425  +0.0593
  hac_gate *                0.1428  +0.0257  0.6364  +0.0532
```

*Qwen3-VL-8B, verbalized vanilla, VQA-RAD closed, val=30/test=70.*

HAC-Platt achieves +0.06 AUROC improvement over raw confidence here.

## Using pre-computed HEDGE scores

For large sweeps you typically want to pre-generate HEDGE/VASE scores with
the CLI wrapper and load them later:

```bash
python experiments/03_generate_hedge.py --model_key qwen3vl_8b --dataset vqa_rad
```

```python
import json
with open("colm_results/hedge_vase/qwen_vqa_rad/detailed_results.json") as f:
    hedge = json.load(f)
h_scores = np.array([r["VASE"] for r in hedge])

cal = CalibrationPipeline(method="hac_platt")
cal.fit(val_conf, val_corr, hallucination_scores=val_h)
calibrated = cal.transform(test_conf, hallucination_scores=test_h)
```
