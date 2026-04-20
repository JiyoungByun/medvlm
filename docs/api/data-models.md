# Data & Models

## `medvlm.load_dataset()`

Load a medical VQA dataset. Auto-downloads from HuggingFace when available.

```python
dataset = medvlm.load_dataset(
    name="vqa_rad",            # dataset name or HuggingFace ID
    split="test",              # "train", "val", or "test"
    question_type="closed",    # "closed", "open", or None (all)
    data_path=None,            # local path (required for VQA-Med datasets)
    subsample_size=None,       # random subsample (None = full dataset)
    seed=42,                   # random seed for subsampling
)
```

Returns a HuggingFace `Dataset` with columns: `question`, `answer`, `image`, `answer_type`.

### Available Datasets

| Name | Source | Auto-download | Questions | Types |
|------|--------|:---:|-----------|-------|
| `vqa_rad` | [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | Yes | 451 | closed, open |
| `slake` | [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) | Yes | 1,061 | closed, open |
| `vqa_med_2019` | [VQA-Med-2019](https://github.com/abachaa/VQA-Med-2019) | No | 500 | closed, open |
| `vqa_med_2020` | [VQA-Med-2020](https://github.com/abachaa/VQA-Med-2020) | No | 500 | closed, open |
| `vqa_med_2021` | [VQA-Med-2021](https://github.com/abachaa/VQA-Med-2021) | No | 500 | open |

### Manual download (VQA-Med)

The VQA-Med datasets (2019, 2020, 2021) are hosted on GitHub and must be
downloaded manually. Clone the release repo, then pass the path to the
**test set subdirectory** via `data_path`.

All three years ship the test images as a `.zip` inside the repo (and
VQA-Med-2021 also zips the outer test folder). After cloning, you must
unzip before passing `data_path` to `load_dataset`.

#### VQA-Med-2019

```bash
git clone https://github.com/abachaa/VQA-Med-2019.git data/vqa_med_2019
cd data/vqa_med_2019/VQAMed2019Test
unzip -q VQAMed2019_Test_Images.zip
```

Expected layout:

```
data/vqa_med_2019/VQAMed2019Test/
├── VQAMed2019_Test_Questions_w_Ref_Answers.txt
└── VQAMed2019_Test_Images/
    └── {image_id}.jpg
```

```python
dataset = medvlm.load_dataset(
    "vqa_med_2019",
    data_path="data/vqa_med_2019/VQAMed2019Test",
)
```

#### VQA-Med-2020

```bash
git clone https://github.com/abachaa/VQA-Med-2020.git data/vqa_med_2020
cd data/vqa_med_2020/VQA-TestSet-ReferenceAnswers-VQAMed2020-Task1
unzip -q VQAMed2020-Task1-VQAnswering-Test-Images.zip
```

Expected layout:

```
data/vqa_med_2020/VQA-TestSet-ReferenceAnswers-VQAMed2020-Task1/
├── VQAMed2020-Task1-VQAnswering-Test-Questions.txt
├── VQAMed2020-Task1-VQAnswering-Test-ReferenceAnswers.txt
└── Task1-2020-VQAnswering-Test-Images/
```

```python
dataset = medvlm.load_dataset(
    "vqa_med_2020",
    data_path="data/vqa_med_2020/VQA-TestSet-ReferenceAnswers-VQAMed2020-Task1",
)
```

#### VQA-Med-2021

VQA-Med-2021 double-zips: an outer `test.zip` containing the test folder,
and an inner image zip.

```bash
git clone https://github.com/abachaa/VQA-Med-2021.git data/vqa_med_2021
cd data/vqa_med_2021
unzip -q test.zip
cd Task1-VQA-2021-TestSet-w-GroundTruth
unzip -q Task1-VQA-2021-TestSet-Images.zip
```

Expected layout:

```
data/vqa_med_2021/Task1-VQA-2021-TestSet-w-GroundTruth/
├── Task1-VQA-2021-TestSet-Questions.txt
├── Task1-VQA-2021-TestSet-ReferenceAnswers.txt
└── images/VQA-500-Images/
```

```python
dataset = medvlm.load_dataset(
    "vqa_med_2021",
    data_path="data/vqa_med_2021/Task1-VQA-2021-TestSet-w-GroundTruth",
)
```

### Question Types

- **closed**: Yes/no questions. Confidence = P(majority answer).
- **open**: Free-form or multiple-choice answer. Correctness determined by containment matching or LLM judge.

## `medvlm.load_model()`

Load a Vision-Language Model with optional quantization.

```python
model, processor, config = medvlm.load_model(
    model_name="qwen3vl_8b",  # short key or full HuggingFace ID
    quantization=None,         # None (bf16), "4bit", or "8bit"
    adapter_path=None,         # LoRA adapter checkpoint path
    use_flash_attention=True,  # Flash Attention 2 (falls back to SDPA)
    device_map="auto",
)
```

Returns `(model, processor, ModelConfig)`.

### Supported Models

| Short Key | HuggingFace ID | Params | Family |
|-----------|---------------|--------|--------|
| `qwen3vl_2b` | `Qwen/Qwen3-VL-2B-Instruct` | 2B | Qwen-VL |
| `qwen3vl_8b` | `Qwen/Qwen3-VL-8B-Instruct` | 8B | Qwen-VL |
| `qwen3vl_32b` | `Qwen/Qwen3-VL-32B-Instruct` | 32B | Qwen-VL |
| `internvl3_2b` | `OpenGVLab/InternVL3-2B-hf` | 2B | InternVL |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B-hf` | 8B | InternVL |
| `internvl3_38b` | `OpenGVLab/InternVL3-38B-hf` | 38B | InternVL |
| `llava_next_7b` | `llava-hf/llava-v1.6-mistral-7b-hf` | 7B | LLaVA-NeXT |
| `llava_next_34b` | `llava-hf/llava-v1.6-34b-hf` | 34B | LLaVA-NeXT |

### Quantization

```python
# Full precision (bf16) — default
model, processor, config = medvlm.load_model("qwen3vl_8b")

# 8-bit quantization (BitsAndBytes)
model, processor, config = medvlm.load_model("qwen3vl_8b", quantization="8bit")

# 4-bit quantization (NF4, double quantization)
model, processor, config = medvlm.load_model("qwen3vl_8b", quantization="4bit")
```

### VRAM Requirements (approximate)

| Model | bf16 | 8-bit | 4-bit |
|-------|------|-------|-------|
| 2B | ~4 GB | ~3 GB | ~2 GB |
| 7--8B | ~16 GB | ~10 GB | ~5 GB |
| 32--38B | ~70 GB | ~35 GB | ~18 GB |

### `ModelConfig`

Returned as the third element of `load_model()`. Key fields:

```python
config.model_id          # "Qwen/Qwen3-VL-8B-Instruct"
config.model_family      # ModelFamily.QWEN_VL
config.use_4bit          # False
config.use_8bit          # False
config.use_flash_attention  # True
config.torch_dtype       # "bfloat16"
```

## `medvlm.MODEL_REGISTRY`

Dictionary mapping short keys to HuggingFace model IDs:

```python
>>> medvlm.MODEL_REGISTRY
{'qwen3vl_2b': 'Qwen/Qwen3-VL-2B-Instruct', ...}
```
