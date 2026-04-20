# Calibration

## `CalibrationPipeline`

Fit/transform interface for post-hoc calibration.

```python
from medvlm import CalibrationPipeline

cal = CalibrationPipeline(method="platt")
cal.fit(val_confidences, val_correctness)
calibrated = cal.transform(test_confidences)
```

### Standard Methods

These only use confidence scores:

| Method | Description |
|--------|-------------|
| `temperature_scaling` | Learns temperature $T$: $\sigma(\text{logit}(c) / T)$ |
| `platt` | Logistic regression on $\mathrm{logit}(c) = \log(c/(1-c))$: $\sigma(a \cdot \mathrm{logit}(c) + b)$ (textbook Platt) |
| `platt_confidence` | Logistic regression on raw confidence: $\sigma(a \cdot c + b)$ |
| `isotonic` | Non-parametric monotonic mapping via isotonic regression |
| `histogram_binning` | Replaces confidence with observed bin accuracy |

### HAC Methods

These require an additional hallucination score $h$ (e.g., HEDGE/VASE semantic entropy).
Higher $h$ = more likely hallucinated.

| Method | Formula | Constraints |
|--------|---------|-------------|
| `hac_platt` | $\sigma(a \cdot \mathrm{logit}(c) + b \cdot h + d)$ | $a \geq 0$, $b \leq 0$, L2 reg on $b$ |
| `hac_platt_confidence` | $\sigma(a \cdot c + b \cdot h + d)$ | $a \geq 0$, $b \leq 0$, L2 reg on $b$ |
| `hac_gate` | $c \cdot \sigma(-a \cdot h + b)$ | $a \geq 0$, L2 reg on $a$ |

**HAC-Platt** adds $h$ as a linear term alongside $c$. The constraint $b \leq 0$
ensures higher hallucination lowers the calibrated confidence.

**HAC-Gate** multiplies the original confidence by a sigmoid gate controlled by $h$.
When hallucination is high, the gate closes and confidence is attenuated.

### `platt` vs `platt_confidence` (logit vs confidence input)

By default (`platt`, `hac_platt`) we feed $\mathrm{logit}(c) = \log(c/(1-c))$
into Platt scaling --- the "textbook" Platt input. The `_confidence` variants
feed the raw confidence $c$ instead.

On our benchmarks (5-fold CV across 3 datasets × 8 models × 2 question types,
240 folds per method), the two variants are nearly interchangeable.
The logit-input default is marginally better with sampling-based confidence;
raw $c$ is marginally better with verbalized confidence on ECE.

| Conf source | Method | ECE ↓ | ACE ↓ | AUROC ↑ |
|---|---|---|---|---|
| Sampling | `platt_confidence` | .093 | .179 | .653 |
| Sampling | `platt` | **.084** | **.178** | .653 |
| Sampling | `hac_platt_confidence` | .106 | .174 | .685 |
| Sampling | `hac_platt` | **.103** | .175 | **.688** |
| Verbalized | `platt_confidence` | **.063** | **.178** | .606 |
| Verbalized | `platt` | .069 | .181 | .606 |
| Verbalized | `hac_platt_confidence` | .101 | **.178** | .671 |
| Verbalized | `hac_platt` | **.100** | .180 | .671 |

### Usage with HAC

```python
cal = CalibrationPipeline(method="hac_platt")

# fit() requires hallucination_scores for HAC methods
cal.fit(val_conf, val_corr, hallucination_scores=val_h)

# transform() also requires hallucination_scores
calibrated = cal.transform(test_conf, hallucination_scores=test_h)
```

Check whether a method needs hallucination scores:
```python
cal = CalibrationPipeline(method="hac_gate")
print(cal.requires_hallucination_scores)  # True
```

### Regularization

HAC-Platt and HAC-Gate use L2 regularization ($\lambda = 0.01$) on the hallucination
coefficient to prevent overfitting when the hallucination signal is weak.
Set `CalibrationPipeline.HAC_LAMBDA` to change:

```python
CalibrationPipeline.HAC_LAMBDA = 0.1  # stronger regularization
```

## `medvlm.evaluate_calibration()`

Compute calibration metrics from confidence and correctness arrays:

```python
report = medvlm.evaluate_calibration(correctness, confidences)
```

Returns a dictionary:

| Key | Description |
|-----|-------------|
| `ece` | Expected Calibration Error |
| `mce` | Maximum Calibration Error |
| `overconfidence` | Overconfidence Error (only positive gaps) |
| `accuracy` | Fraction correct |
| `mean_confidence` | Mean predicted confidence |
| `num_samples` | Number of samples |
| `bin_data` | Per-bin statistics (accuracy, confidence, gap, size) |

## `train_val_test_split()`

Split a HuggingFace Dataset into validation and test sets:

```python
from medvlm import train_val_test_split

val_set, test_set = train_val_test_split(dataset, val_fraction=0.3, seed=42)
```
