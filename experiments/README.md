# Experiments

Reproduction pipeline for the MedVLM paper. **Not shipped with the pip package** —
these scripts are tied to the repo layout (`data/`, `colm_results/`) and depend
on GPUs. They are the thin CLI drivers on top of the public `medvlm` API.

## Pipeline order

1. `01_generate_sampling.py` — sampling-based confidence (N=20 samples/question).
2. `02_generate_verbalized.py` — verbalized-confidence prompts (6 variants).
3. `03_generate_hedge.py` — HEDGE / RadFlag / VASE hallucination scores.
4. `04_postprocess.py` — LLM-as-judge re-scoring + semantic clustering for open-ended answers.
5. `05_export_raw_table.py` — merge everything into a flat CSV/Parquet table.

Each driver prints its own `--help`. Results land under `./colm_results/`.

## Running from the repo root

```bash
# Install in editable mode first (one-time):
pip install -e ".[dev,hedge]"

# Then:
python experiments/01_generate_sampling.py --gpu 0 --model qwen3vl_2b --dataset vqa_rad
```

## Config split

* **Domain constants** (model registry, prompts, default N/seed) live in
  `medvlm.configs` and are part of the installable public API.
* **Repo-local paths and helpers** (`COLM_RESULTS`, `DATA_DIR`, `results_exist`)
  live in `experiments/config.py` and are research-only.
