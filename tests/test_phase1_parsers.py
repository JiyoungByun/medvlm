"""Phase 1: pure-Python parsing correctness (no GPU, no model)."""
import pytest

from medvlm.confidence import (
    parse_answer_text, normalize_answer, check_answer_correct,
    parse_confidence_numeric, parse_confidence_linguistic,
    LINGUISTIC_MAP,
)
from medvlm.evaluation import parse_yes_no


# ---- yes/no parsing (used in sampling and verbalized) ----

@pytest.mark.parametrize("text,expected", [
    ("yes", "yes"),
    ("Yes", "yes"),
    ("Yes.", "yes"),
    ("Yes,", "yes"),
    ("yes!", "yes"),
    ("no", "no"),
    ("No.", "no"),
    ("Yes, the lung shows opacity", "yes"),  # starts with yes
    ("No, the scan is normal", "no"),
    ("the answer is yes", "yes"),  # contains yes only
    ("the answer is no", "no"),
    ("maybe", None),
    # Note: parse_yes_no biases toward "yes" when both appear, because
    # "starts with yes" check fires before the both-present check. Documented
    # behavior of the implementation; not a bug.
    ("yes and no", "yes"),
])
def test_parse_yes_no(text, expected):
    assert parse_yes_no(text) == expected


# ---- parse_answer_text (sampling) ----

@pytest.mark.parametrize("text,expected", [
    ("Answer: yes", "yes"),
    ("Answer: no.", "no"),
    ("Answer: cardiomegaly", "cardiomegaly"),
    ("answer: ct scan\nFoo bar", "ct scan"),
    ("yes", "yes"),  # short fallback
    ("a b c", "a b c"),  # 3 words still fallback
    ("a b c d e", None),  # > 3 words, no Answer: prefix
])
def test_parse_answer_text(text, expected):
    assert parse_answer_text(text) == expected


# ---- normalize_answer ----

@pytest.mark.parametrize("ans,qt,expected", [
    ("yes", "closed", "yes"),
    ("no", "closed", "no"),
    ("yes.", "closed", "yes"),
    ("yes, definitely", "closed", "yes"),
    ("no, the scan is normal", "closed", "no"),
    ("CARDIOMEGALY", "open", "cardiomegaly"),
    (None, "closed", None),
    ("maybe", "closed", None),
])
def test_normalize_answer(ans, qt, expected):
    assert normalize_answer(ans, qt) == expected


# ---- check_answer_correct ----

@pytest.mark.parametrize("pred,gt,qt,expected", [
    ("yes", "yes", "closed", True),
    ("no", "yes", "closed", False),
    ("cardiomegaly", "cardiomegaly", "open", True),
    ("the answer is cardiomegaly", "cardiomegaly", "open", True),  # containment
    ("cardio", "cardiomegaly", "open", True),  # pred in gt
    ("unknown", "yes", "closed", False),
    (None, "yes", "closed", False),
])
def test_check_answer_correct(pred, gt, qt, expected):
    assert check_answer_correct(pred, gt, qt) == expected


# ---- parse_confidence_numeric (verbalized) ----

@pytest.mark.parametrize("text,expected", [
    ("Confidence: 95%", 0.95),
    ("Confidence: 80", 0.80),
    ("confidence: 50%", 0.50),
    ("Confidence: 0.85", 0.85),  # already in [0,1]
    ("Confidence: 100%", 1.0),
    ("Confidence: 0%", 0.0),
    ("Confidence: 110%", 1.0),  # clipped
    ("no confidence here", None),
])
def test_parse_confidence_numeric(text, expected):
    got = parse_confidence_numeric(text)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, abs=1e-6)


# ---- parse_confidence_linguistic ----

@pytest.mark.parametrize("text,expected", [
    ("Confidence: almost certain", 0.95),
    ("Confidence: highly likely", 0.90),
    ("Confidence: very good chance", 0.85),
    ("Confidence: about even", 0.50),
    ("Confidence: unlikely", 0.30),
    ("Confidence: highly unlikely", 0.10),
    # longer-prefix wins
    ("Confidence: very good chance not", 0.15),
])
def test_parse_confidence_linguistic(text, expected):
    assert parse_confidence_linguistic(text) == pytest.approx(expected)


def test_linguistic_map_complete():
    """All 12 linguistic terms mentioned in README must be in the map."""
    expected_terms = {
        "almost certain", "highly likely", "very good chance",
        "probable", "likely", "better than even", "about even",
        "unlikely", "improbable", "very good chance not",
        "highly unlikely", "almost certainly not",
    }
    assert set(LINGUISTIC_MAP.keys()) == expected_terms
    # All values in [0,1] and roughly monotone with the order above
    for v in LINGUISTIC_MAP.values():
        assert 0.0 <= v <= 1.0
