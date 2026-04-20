<div align="center">

# MedVLM: Inference, Confidence, and Calibration in Medical VQA

[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://github.com/jiyoungbyun/medvlm)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Project Page](https://jiyoungbyun.github.io/vlm-hac/) |
[Documentation](https://jiyoungbyun.github.io/medvlm) |
[Installation](#installation) |
[Quick Start](#quick-start) |
[Examples](#examples) |
[Models & Datasets](#models--datasets) |
[Citation](#citation)

</div>

## Installation

```bash
pip install medvlm
```

Requires Python >= 3.10 and a CUDA-capable GPU. See
[Installation](https://jiyoungbyun.github.io/medvlm/getting-started/#installation)
for the full matrix (matching PyTorch to your NVIDIA driver, the
`medvlm[internvl]` extras, and Flash Attention).

## Quick Start

```python
import medvlm

# 1. Load data (auto-downloads from HuggingFace)
dataset = medvlm.load_dataset("vqa_rad", split="test", question_type="closed")

# 2. Load model
model, processor, config = medvlm.load_model("qwen3vl_2b")

# 3. Compute confidence (sampling-based, N=20)
results = medvlm.compute_confidence(
    model, processor, config, dataset,
    method="sampling", num_samples=20,
)

# 4. Evaluate calibration
report = medvlm.evaluate_calibration(
    [r.is_correct for r in results],
    [r.confidence for r in results],
)
print(f"ECE: {report['ece']:.4f}, Accuracy: {report['accuracy']:.4f}")
```

### Post-hoc calibration

```python
from medvlm import CalibrationPipeline, train_val_test_split

val_set, test_set = train_val_test_split(dataset, val_fraction=0.3)

# ... compute confidence on both splits ...

cal = CalibrationPipeline(method="platt")
cal.fit(val_conf, val_corr)
calibrated = cal.transform(test_conf)
```

### Hallucination scores (SE / RadFlag / VASE)

Compute per-question hallucination scores via the full HEDGE pipeline
(greedy + $N$ high-temperature samples + $N$ distorted-image samples,
clustered; powered by [hedge-bench](https://github.com/SushantGautam/HEDGE)).
Useful on their own (for ranking or filtering questions by uncertainty) or
as the $h$ input to HAC calibration below.

```python
hedge = medvlm.compute_hedge_scores(
    model, processor, config, dataset,
    question_type="closed",    # "closed" -> yes/no clustering, "open" -> embedding
    n_samples=10,
    alpha=1.0,                 # VASE contrastive weight
)
h = np.array([r.VASE for r in hedge])   # or r.SE, r.RadFlag
```

Each `HedgeResult` also carries the raw samples and their log-likelihoods,
so you can re-cluster or try a different score later without re-generating.

### HAC calibration (with hallucination scores)

```python
cal = CalibrationPipeline(method="hac_platt")
cal.fit(val_conf, val_corr, hallucination_scores=val_h)
calibrated = cal.transform(test_conf, hallucination_scores=test_h)
```

### Custom data

`compute_confidence` and `compute_hedge_scores` accept any iterable of dicts
with `image`, `question`, and optional `answer` --- you don't need to go
through `load_dataset`:

```python
examples = [
    {"image": pil_image, "question": "Is there a fracture?", "answer": "yes"},
    {"image": pil_image_2, "question": "What organ is shown?"},  # answer optional
]
results = medvlm.compute_confidence(
    model, processor, config, examples,
    method="sampling", num_samples=20,
)
hedge = medvlm.compute_hedge_scores(
    model, processor, config, examples,
    question_type="closed", n_samples=10,
)
```

If `answer` is omitted, `ground_truth` and `is_correct` on each result are `None`.

For a val/test split on a plain list, split it yourself --- `train_val_test_split`
is a thin wrapper around HuggingFace `Dataset` methods:

```python
import random
random.Random(42).shuffle(examples)
k = int(0.3 * len(examples))
val_set, test_set = examples[:k], examples[k:]
```

## Examples

Three self-contained scripts that reproduce the paper's main findings:

| Script | What it shows | GPU time |
|--------|---------------|----------|
| [`overconfidence.py`](examples/overconfidence.py) | VLMs are overconfident | ~1 min (2B) |
| [`calibration.py`](examples/calibration.py) | Post-hoc calibration reduces ECE | ~2 min (2B) |
| [`hac.py`](examples/hac.py) | HAC improves AUROC | ~2 min (2B) |

```bash
CUDA_VISIBLE_DEVICES=0 python examples/overconfidence.py
CUDA_VISIBLE_DEVICES=0 python examples/calibration.py
CUDA_VISIBLE_DEVICES=0 python examples/hac.py --model qwen3vl_8b
```

## Confidence Methods

| Method | Description | API |
|--------|-------------|-----|
| **Sampling** | Generate N answers, confidence = P(majority) | `method="sampling", num_samples=20` |
| **Verbalized** | Model self-reports confidence (0--100%) | `method="verbalized", variant="vanilla"` |

Verbalized variants: `vanilla`, `vanilla_cot`, `punish`, `top_k`, `two_stage`, `linguistic`.

## Calibration Methods

### Standard (confidence only)

| Method | Description |
|--------|-------------|
| `temperature_scaling` | Scale logits by learned temperature T |
| `platt` | Logistic regression on $\mathrm{logit}(c) = \log(c/(1-c))$ (textbook Platt) |
| `platt_confidence` | Logistic regression on raw confidence $c$: $\sigma(a \cdot c + b)$ |
| `isotonic` | Non-parametric monotonic mapping |
| `histogram_binning` | Bin-wise accuracy replacement |

### HAC (confidence + hallucination scores)

| Method | Formula | Key idea |
|--------|---------|----------|
| `hac_platt` | $\sigma(a \cdot \mathrm{logit}(c) + b \cdot h + d)$ | Hallucination lowers confidence (b <= 0) |
| `hac_platt_confidence` | $\sigma(a \cdot c + b \cdot h + d)$ | Same, but feeds raw $c$ instead of $\mathrm{logit}(c)$ |
| `hac_gate` | $c \cdot \sigma(-a \cdot h + b)$ | Hallucination gates (attenuates) confidence |

> **`platt` vs `platt_confidence`:** by default we feed $\mathrm{logit}(c) = \log(c/(1-c))$
> into Platt scaling and HAC-Platt --- the "textbook" Platt input. The
> `_confidence` variants feed the raw confidence $c$ instead. On our benchmarks
> (5-fold CV across 3 datasets × 8 models × 2 question types = 240 folds/method)
> the two are nearly interchangeable; the logit-input default is marginally
> better with sampling-based confidence, while raw $c$ is marginally better with
> verbalized confidence on ECE:
>
> | Conf source | Method | ECE ↓ | ACE ↓ | AUROC ↑ |
> |---|---|---|---|---|
> | Sampling | `platt_confidence` | .093 | .179 | .653 |
> | Sampling | `platt` | **.084** | **.178** | .653 |
> | Sampling | `hac_platt_confidence` | .106 | .174 | .685 |
> | Sampling | `hac_platt` | **.103** | .175 | **.688** |
> | Verbalized | `platt_confidence` | **.063** | **.178** | .606 |
> | Verbalized | `platt` | .069 | .181 | .606 |
> | Verbalized | `hac_platt_confidence` | .101 | **.178** | .671 |
> | Verbalized | `hac_platt` | **.100** | .180 | .671 |

## Models & Datasets

### Models

| Key | Model | Parameters |
|-----|-------|-----------|
| `qwen3vl_2b` | Qwen/Qwen3-VL-2B-Instruct | 2B |
| `qwen3vl_8b` | Qwen/Qwen3-VL-8B-Instruct | 8B |
| `qwen3vl_32b` | Qwen/Qwen3-VL-32B-Instruct | 32B |
| `internvl3_2b` | OpenGVLab/InternVL3-2B-hf | 2B |
| `internvl3_8b` | OpenGVLab/InternVL3-8B-hf | 8B |
| `internvl3_38b` | OpenGVLab/InternVL3-38B-hf | 38B |
| `llava_next_7b` | llava-hf/llava-v1.6-mistral-7b-hf | 7B |
| `llava_next_34b` | llava-hf/llava-v1.6-34b-hf | 34B |

```python
model, processor, config = medvlm.load_model("qwen3vl_8b")
model, processor, config = medvlm.load_model("qwen3vl_8b", quantization="8bit")
```

### Datasets

| Dataset | Key | Questions | Source |
|---------|-----|-----------|--------|
| [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | `vqa_rad` | 451 | auto-download (HF) |
| [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) | `slake` | 1,061 | auto-download (HF) |
| [VQA-Med-2019](https://github.com/abachaa/VQA-Med-2019) | `vqa_med_2019` | 500 | manual download |
| [VQA-Med-2020](https://github.com/abachaa/VQA-Med-2020) | `vqa_med_2020` | 500 | manual download |
| [VQA-Med-2021](https://github.com/abachaa/VQA-Med-2021) | `vqa_med_2021` | 500 | manual download |

```python
dataset = medvlm.load_dataset("vqa_rad", split="test", question_type="closed")
```

#### Manual download (VQA-Med)

The VQA-Med datasets are hosted on GitHub and must be downloaded manually.
All three years ship the test images as a `.zip` inside the repo, so you
need to unzip after cloning. Example for VQA-Med-2019:

```bash
git clone https://github.com/abachaa/VQA-Med-2019.git data/vqa_med_2019
unzip -q data/vqa_med_2019/VQAMed2019Test/VQAMed2019_Test_Images.zip \
    -d data/vqa_med_2019/VQAMed2019Test/
```

Then point `load_dataset` at the test directory:

```python
dataset = medvlm.load_dataset(
    "vqa_med_2019",
    data_path="data/vqa_med_2019/VQAMed2019Test",
)
```

For the expected directory layout of all three VQA-Med years, see
[`data/README.md`](data/README.md) or the [Data & Models](docs/api/data-models.md) docs page.

## Pipeline Scripts

For running full-scale experiments across all models and datasets:

```
experiments/01_generate_sampling.py    # Sampling confidence (GPU)
experiments/02_generate_verbalized.py  # Verbalized confidence (GPU)
experiments/03_generate_hedge.py       # HEDGE/VASE scores (GPU)
experiments/04_postprocess.py          # LLM judge + clustering (GPU)
experiments/05_export_raw_table.py     # Export CSV (CPU)
```

## Citation

```bibtex
@article{byun2026overconfidence,
  title={Overconfidence and Calibration in Medical VQA: Empirical Findings and Hallucination-Aware Mitigation},
  author={Byun, Ji Young and Park, Young-Jin and Corbeil, Jean-Philippe and Ben Abacha, Asma},
  journal={arXiv preprint arXiv:2604.02543},
  year={2026},
  url={https://arxiv.org/abs/2604.02543},
}
```

## License

MIT
