"""Phase 2: dataset loading on CPU.

vqa_rad / slake auto-download from HuggingFace; vqa_med_2019/2020/2021
require local data via the symlinks in ./data.
"""
import os
import pytest
from pathlib import Path
from PIL import Image as PILImage

import medvlm

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"

# Skip VQA-Med tests if local data isn't present
HAS_2019 = (DATA / "vqa_med_2019" / "VQAMed2019Test").exists()
HAS_2020 = (DATA / "vqa_med_2020"
            / "VQA-TestSet-ReferenceAnswers-VQAMed2020-Task1").exists()
HAS_2021 = (DATA / "vqa_med_2021"
            / "Task1-VQA-2021-TestSet-w-GroundTruth").exists()


SCHEMA = {"image", "question", "answer", "answer_type",
          "question_id", "image_id", "dataset_source"}


def _check_sample(s, expected_source):
    assert SCHEMA <= set(s.keys())
    assert isinstance(s["question"], str) and s["question"]
    assert isinstance(s["answer"], str)
    assert s["answer_type"] in ("closed", "open")
    assert s["dataset_source"] == expected_source
    # image must materialize to a PIL Image
    img = s["image"]
    assert isinstance(img, PILImage.Image), f"image is {type(img)}"
    assert img.size[0] > 0 and img.size[1] > 0


# ---- VQA-RAD (HF auto-download) ----

def test_vqa_rad_load_test():
    ds = medvlm.load_dataset("vqa_rad", split="test")
    # README says 451 questions
    assert 400 <= len(ds) <= 500, f"expected ~451, got {len(ds)}"
    _check_sample(ds[0], "VQA-RAD")


def test_vqa_rad_closed_filter():
    ds = medvlm.load_dataset("vqa_rad", split="test", question_type="closed")
    assert len(ds) > 0
    types = {s["answer_type"] for s in ds}
    assert types == {"closed"}
    # closed answers should all be yes/no
    answers = {s["answer"].lower().strip() for s in ds}
    assert answers <= {"yes", "no"}


def test_vqa_rad_open_filter():
    ds = medvlm.load_dataset("vqa_rad", split="test", question_type="open")
    assert len(ds) > 0
    types = {s["answer_type"] for s in ds}
    assert types == {"open"}


def test_vqa_rad_subsample_reproducible():
    ds1 = medvlm.load_dataset("vqa_rad", split="test", subsample_size=20, seed=42)
    ds2 = medvlm.load_dataset("vqa_rad", split="test", subsample_size=20, seed=42)
    assert len(ds1) == 20 == len(ds2)
    assert ds1["question"] == ds2["question"]


def test_vqa_rad_alias():
    """README says rad_vqa is accepted as alias for vqa_rad."""
    a = medvlm.load_dataset("rad_vqa", split="test", subsample_size=10, seed=1)
    b = medvlm.load_dataset("vqa_rad", split="test", subsample_size=10, seed=1)
    assert a["question"] == b["question"]


# ---- SLAKE (HF auto-download) ----

def test_slake_load_test():
    ds = medvlm.load_dataset("slake", split="test")
    # README says 1061 (English-filtered)
    assert 900 <= len(ds) <= 1200, f"expected ~1061, got {len(ds)}"
    _check_sample(ds[0], "SLAKE")


def test_slake_closed_filter_yes_no_only():
    """SLAKE 'closed' is normalized: only yes/no answers count as closed."""
    ds = medvlm.load_dataset("slake", split="test", question_type="closed")
    assert len(ds) > 0
    answers = {s["answer"].lower().strip() for s in ds}
    assert answers <= {"yes", "no"}


# ---- VQA-Med (local) ----

@pytest.mark.skipif(not HAS_2019, reason="vqa_med_2019 not present")
def test_vqa_med_2019():
    ds = medvlm.load_dataset(
        "vqa_med_2019",
        data_path=str(DATA / "vqa_med_2019" / "VQAMed2019Test"),
    )
    assert 400 <= len(ds) <= 600
    _check_sample(ds[0], "VQA-Med-2019")


@pytest.mark.skipif(not HAS_2020, reason="vqa_med_2020 not present")
def test_vqa_med_2020():
    ds = medvlm.load_dataset(
        "vqa_med_2020",
        data_path=str(DATA / "vqa_med_2020"
                      / "VQA-TestSet-ReferenceAnswers-VQAMed2020-Task1"),
    )
    assert 400 <= len(ds) <= 600
    _check_sample(ds[0], "VQA-Med-2020")


@pytest.mark.skipif(not HAS_2021, reason="vqa_med_2021 not present")
def test_vqa_med_2021_all_open():
    """VQA-Med-2021 is all open questions per the README table."""
    ds = medvlm.load_dataset(
        "vqa_med_2021",
        data_path=str(DATA / "vqa_med_2021"
                      / "Task1-VQA-2021-TestSet-w-GroundTruth"),
    )
    assert 400 <= len(ds) <= 600
    types = {s["answer_type"] for s in ds}
    assert types == {"open"}, f"expected only open, got {types}"
    _check_sample(ds[0], "VQA-Med-2021")


# ---- error handling ----

def test_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown dataset"):
        medvlm.load_dataset("not_a_dataset")


def test_unknown_question_type():
    with pytest.raises(ValueError, match="Unknown question_type"):
        medvlm.load_dataset("vqa_rad", question_type="bogus")
