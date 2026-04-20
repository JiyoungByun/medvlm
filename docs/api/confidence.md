# Confidence Estimation

## `medvlm.compute_confidence()`

Unified entry point for computing confidence scores.

```python
results = medvlm.compute_confidence(
    model, processor, config, examples,
    method="sampling",        # "sampling" or "verbalized"
    question_type="closed",   # "closed" or "open"
    # Sampling-specific:
    num_samples=20,
    temperature=0.7,
    prompt_mode="base",       # "base" or "cot"
    samples_per_batch=25,
    # Verbalized-specific:
    variant="vanilla",
    max_new_tokens=256,
    batch_size=1,
)
```

### `examples` argument

Any iterable of dicts with keys `image`, `question`, and optionally `answer`.
Two common shapes:

```python
# (a) HuggingFace Dataset from medvlm.load_dataset — iterates as dicts
examples = medvlm.load_dataset("vqa_rad", split="test", question_type="closed")

# (b) Plain list of dicts (e.g., a single image in memory)
examples = [
    {"image": pil_image, "question": "is cardiomegaly present?", "answer": "yes"},
    {"image": pil_image, "question": "is the scan normal?"},   # no answer -> is_correct=None
]
```

### Returns

- **Sampling**: List of `SamplingResult` objects
- **Verbalized**: List of `VerbalizedResult` objects

Both have `.confidence`, `.is_correct`, `.predicted`, `.question`, `.ground_truth`.
When no `answer` is provided, `ground_truth` and `is_correct` are `None`.

## `SamplingResult`

```python
@dataclass
class SamplingResult:
    question: str
    ground_truth: str
    predicted: str           # majority answer
    confidence: float        # P(majority answer) among valid responses
    is_correct: bool
    answer_counts: dict      # {"yes": 15, "no": 5}
    unknown_count: int       # unparseable responses
    valid_response_rate: float
    raw_responses: list      # all N raw responses
```

## `VerbalizedResult`

```python
@dataclass
class VerbalizedResult:
    question: str
    ground_truth: str
    predicted: str
    confidence: float        # self-reported confidence (0-1)
    is_correct: bool
    parse_success: bool      # whether confidence was parseable
    raw_response: str        # full model response
    variant: str
    raw_stage2: str | None   # stage 2 response (two_stage variant only)
```

## `medvlm.compute_hedge_scores()`

Compute SE / RadFlag / VASE hallucination scores for each question. Wraps
[hedge-bench](https://github.com/SushantGautam/HEDGE) to run the full HEDGE
pipeline: one greedy answer plus $N$ high-temperature samples on the
original image, plus one answer per distorted image ($N$ distortions),
clustered (yes/no for closed, sentence-embedding for open) and scored.

```python
hedge = medvlm.compute_hedge_scores(
    model, processor, config, examples,
    question_type="closed",    # "closed" -> yes/no, "open" -> embedding clustering
    n_samples=10,              # high-temp samples and distortions (each)
    temperature=0.7,
    alpha=1.0,                 # VASE contrastive weight
    batch_size=5,              # distorted-image generation batch size
)
h = np.array([r.VASE for r in hedge])   # or r.SE, r.RadFlag
```

`examples` has the same contract as `compute_confidence` above.

Returns a list of `HedgeResult`:

```python
@dataclass
class HedgeResult:
    question: str
    ground_truth: str
    predicted: str               # parsed from the greedy answer
    is_correct: bool
    greedy_answer: str
    SE: float                    # semantic entropy on clean samples
    RadFlag: float               # greedy-cluster agreement rate
    VASE: float                  # contrastive entropy (clean vs. distorted)
    original_high_temp: list     # N raw samples on original image
    original_logprobs: list      # mean log-likelihood per sample
    distorted_high_temp: list    # N samples, one per distorted image
    distorted_logprobs: list
```

Use as `hallucination_scores` in HAC calibration:

```python
cal = CalibrationPipeline(method="hac_platt")
cal.fit(val_conf, val_corr, hallucination_scores=val_h)
```

Raw samples and log-likelihoods are kept so you can re-cluster or try a
different score (SE vs. RadFlag vs. VASE) later without re-generating. The
CLI wrapper `experiments/03_generate_hedge.py` runs this across a whole dataset
and saves the results to JSON.

## Prompt Templates

### Sampling

- **base**: "Provide only the answer, without any explanation. Format: Answer: [your answer]"
- **cot**: "Think step by step. Then provide your answer. Format: Reasoning: [...] Answer: [...]"

### Verbalized

All variants include the question and image, then ask for confidence in different ways.
See `medvlm.confidence.verbalized.VERBALIZED_PROMPTS` for the exact prompt text.
