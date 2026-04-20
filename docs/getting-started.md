# Getting Started

## Installation

### Quickstart

```bash
pip install medvlm
```

### Python version

medvlm requires **Python >= 3.10**. If your system default is older, create
a dedicated virtualenv first:

```bash
python3.10 -m venv .venv && source .venv/bin/activate
```

### PyTorch and your NVIDIA driver

`pip install medvlm` will pull the default PyPI PyTorch wheel, which
currently tracks the newest CUDA build (torch 2.11 needs NVIDIA
driver 555+). On older drivers this installs but fails at import with
a CUDA error. To avoid that, install torch yourself first from the
index URL matching your CUDA toolkit, then install medvlm:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install medvlm
```

medvlm is tested against `torch>=2.4,<2.12`; pick the newest wheel your
driver supports within that range.

### InternVL extras

The InternVL loader depends on `timm`, which is not pulled in by the
base install. If you plan to use `internvl3_*` models:

```bash
pip install 'medvlm[internvl]'
```

### Flash Attention (optional)

For faster inference on CUDA GPUs:

```bash
pip install flash-attn --no-build-isolation --no-deps
```

## Basic Usage

```python
import medvlm

# Load data (auto-downloads from HuggingFace)
dataset = medvlm.load_dataset("vqa_rad", split="test", question_type="closed")

# Load model
model, processor, config = medvlm.load_model("qwen3vl_2b")

# Compute sampling-based confidence (N=20 samples per question)
results = medvlm.compute_confidence(
    model, processor, config, dataset,
    method="sampling", num_samples=20,
)

# Evaluate calibration
report = medvlm.evaluate_calibration(
    [r.is_correct for r in results],
    [r.confidence for r in results],
)
print(f"ECE: {report['ece']:.4f}")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Mean Confidence: {report['mean_confidence']:.4f}")
print(f"Overconfidence: {report['overconfidence']:.4f}")
```

## Confidence Methods

### Sampling-based

Generates N independent answers per question and computes confidence as P(majority answer):

```python
results = medvlm.compute_confidence(
    model, processor, config, dataset,
    method="sampling",
    num_samples=20,       # N samples per question
    temperature=0.7,      # sampling temperature
    prompt_mode="base",   # "base" (direct) or "cot" (chain-of-thought)
)
```

Each result has `.confidence`, `.is_correct`, `.answer_counts`, `.raw_responses`.

### Verbalized

Prompts the model to self-report its confidence (0--100%):

```python
results = medvlm.compute_confidence(
    model, processor, config, dataset,
    method="verbalized",
    variant="vanilla",    # see below for all variants
    batch_size=4,         # questions per forward pass
)
```

Each result has `.confidence`, `.is_correct`, `.parse_success`, `.raw_response`.

**Variants:**

| Variant | Description |
|---------|-------------|
| `vanilla` | "Provide your answer and confidence (0--100%)" |
| `vanilla_cot` | Same with "think step by step" prefix |
| `punish` | Adds punishment framing for overconfidence |
| `top_k` | Top-3 guesses with probabilities |
| `two_stage` | Answer first, then rate confidence separately |
| `linguistic` | "almost certain", "likely", etc. mapped to numbers |

## Post-hoc Calibration

Fit a calibrator on a validation split, apply to test:

```python
from medvlm import CalibrationPipeline, train_val_test_split
import numpy as np

# Split dataset
val_set, test_set = train_val_test_split(dataset, val_fraction=0.3, seed=42)

# Compute confidence on both splits
val_results = medvlm.compute_confidence(model, processor, config, val_set, ...)
test_results = medvlm.compute_confidence(model, processor, config, test_set, ...)

val_conf = np.array([r.confidence for r in val_results])
val_corr = np.array([r.is_correct for r in val_results])
test_conf = np.array([r.confidence for r in test_results])
test_corr = np.array([r.is_correct for r in test_results])

# Fit and transform
cal = CalibrationPipeline(method="platt")
cal.fit(val_conf, val_corr)
calibrated = cal.transform(test_conf)

# Evaluate
report = medvlm.evaluate_calibration(test_corr, calibrated)
```

## HAC Calibration

HAC methods require hallucination scores (e.g., from HEDGE/VASE):

```python
cal = CalibrationPipeline(method="hac_platt")
cal.fit(val_conf, val_corr, hallucination_scores=val_h)
calibrated = cal.transform(test_conf, hallucination_scores=test_h)
```

See [Calibration API](api/calibration.md) for all available methods.

## Available Models

```python
# List all registered models
print(medvlm.MODEL_REGISTRY)

# Load by short key
model, processor, config = medvlm.load_model("qwen3vl_8b")

# Load by full HuggingFace ID
model, processor, config = medvlm.load_model("Qwen/Qwen3-VL-8B-Instruct")

# With quantization
model, processor, config = medvlm.load_model("qwen3vl_8b", quantization="8bit")
```

## Available Datasets

```python
# Auto-download from HuggingFace
dataset = medvlm.load_dataset("vqa_rad", split="test")
dataset = medvlm.load_dataset("slake", split="test", question_type="open")

# Subsample for quick experiments
dataset = medvlm.load_dataset("vqa_rad", subsample_size=50, seed=42)
```
