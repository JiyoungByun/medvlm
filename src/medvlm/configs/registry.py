"""Registry of supported VLMs.

Maps short keys (used across the CLI, experiments, and examples) to HuggingFace
model IDs, plus GPU requirements for full-precision and 4-bit loading.
"""
from __future__ import annotations

from typing import Dict

MODELS: Dict[str, str] = {
    "qwen3vl_2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl_32b": "Qwen/Qwen3-VL-32B-Instruct",
    "internvl3_2b": "OpenGVLab/InternVL3-2B-hf",
    "internvl3_8b": "OpenGVLab/InternVL3-8B-hf",
    "internvl3_38b": "OpenGVLab/InternVL3-38B-hf",
    "llava_next_7b": "llava-hf/llava-v1.6-mistral-7b-hf",
    "llava_next_34b": "llava-hf/llava-v1.6-34b-hf",
}

MODEL_SHORT_KEYS: Dict[str, str] = {
    "qwen3vl_2b": "qwen2b",
    "qwen3vl_8b": "qwen",
    "qwen3vl_32b": "qwen32b",
    "internvl3_2b": "internvl2b",
    "internvl3_8b": "internvl",
    "internvl3_38b": "internvl38b",
    "llava_next_7b": "llava",
    "llava_next_34b": "llava34b",
}

MODEL_NUM_GPUS: Dict[str, int] = {
    "qwen3vl_2b": 1,
    "qwen3vl_8b": 1,
    "qwen3vl_32b": 1,
    "internvl3_2b": 1,
    "internvl3_8b": 1,
    "internvl3_38b": 2,
    "llava_next_7b": 1,
    "llava_next_34b": 1,
}

MODEL_NUM_GPUS_4BIT: Dict[str, int] = {k: 1 for k in MODEL_NUM_GPUS}
