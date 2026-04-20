"""Research-only paths and utilities for the experiment pipeline.

Domain constants (model registry, prompts, defaults) live in ``medvlm.configs``
and are part of the pip-installable public API. Everything here is tied to the
layout of this repository and is not installed.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

COLM_RESULTS = PROJECT_ROOT / "colm_results"
SAMPLING_DIR = COLM_RESULTS / "sampling"
VERBALIZED_DIR = COLM_RESULTS / "verbalized"


def results_exist(output_dir, marker_file: str = "metrics.json") -> bool:
    """Return True if a completed run already wrote ``marker_file`` under output_dir."""
    output_dir = str(output_dir)
    if not os.path.isdir(output_dir):
        return False
    if os.path.isfile(os.path.join(output_dir, marker_file)):
        return True
    for sub in ("sampling", "logits"):
        if os.path.isfile(os.path.join(output_dir, sub, marker_file)):
            return True
    return False
