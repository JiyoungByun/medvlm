"""Phase 1: clustering, splitting, registry."""
import numpy as np
import pytest
from datasets import Dataset

import medvlm
from medvlm import MODEL_REGISTRY, train_val_test_split
from medvlm.confidence import cluster_by_yesno
from medvlm.data import list_available_datasets


def test_cluster_by_yesno_basic():
    answers = ["yes", "no", "Yes", "no.", "yes!", "I don't know"]
    ids = cluster_by_yesno(answers)
    # yes / no / other -> 3 distinct clusters
    assert len(set(ids)) == 3
    # all "yes"-style get same id
    yes_ids = [ids[i] for i in (0, 2, 4)]
    assert len(set(yes_ids)) == 1
    # all "no"-style get same id
    no_ids = [ids[i] for i in (1, 3)]
    assert len(set(no_ids)) == 1


def test_cluster_by_yesno_all_same():
    ids = cluster_by_yesno(["yes", "yes", "yes"])
    assert len(set(ids)) == 1


def test_train_val_test_split_sizes():
    ds = Dataset.from_dict({"x": list(range(100))})
    val, test = train_val_test_split(ds, val_fraction=0.3, seed=42)
    assert len(val) == 30
    assert len(test) == 70


def test_train_val_test_split_no_overlap():
    ds = Dataset.from_dict({"x": list(range(100))})
    val, test = train_val_test_split(ds, val_fraction=0.2, seed=42)
    val_xs = set(val["x"])
    test_xs = set(test["x"])
    assert val_xs.isdisjoint(test_xs)
    assert val_xs | test_xs == set(range(100))


def test_train_val_test_split_reproducible():
    ds = Dataset.from_dict({"x": list(range(100))})
    v1, t1 = train_val_test_split(ds, val_fraction=0.3, seed=42)
    v2, t2 = train_val_test_split(ds, val_fraction=0.3, seed=42)
    assert v1["x"] == v2["x"]
    assert t1["x"] == t2["x"]


def test_train_val_test_split_invalid_fraction():
    ds = Dataset.from_dict({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        train_val_test_split(ds, val_fraction=0.0)
    with pytest.raises(ValueError):
        train_val_test_split(ds, val_fraction=1.0)


def test_model_registry_contents():
    """README documents 8 models — registry must match."""
    expected = {
        "qwen3vl_2b", "qwen3vl_8b", "qwen3vl_32b",
        "internvl3_2b", "internvl3_8b", "internvl3_38b",
        "llava_next_7b", "llava_next_34b",
    }
    assert set(MODEL_REGISTRY.keys()) == expected
    # values should be valid HF IDs (org/repo)
    for k, v in MODEL_REGISTRY.items():
        assert "/" in v, f"{k} -> {v} is not org/repo format"


def test_list_available_datasets():
    """README documents 5 datasets."""
    expected = {"vqa_rad", "slake", "vqa_med_2019", "vqa_med_2020", "vqa_med_2021"}
    assert set(list_available_datasets()) == expected
