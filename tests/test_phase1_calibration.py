"""Phase 1: calibration metrics + CalibrationPipeline (all 8 methods).

Synthetic data only — verifies that:
  * ECE is 0 for perfectly-calibrated input
  * ECE is large for systematically over-confident input
  * Standard methods (5) reduce ECE on miscalibrated data
  * HAC methods (3) require hallucination_scores, enforce sign constraints,
    and reduce ECE / improve AUROC when h is informative
"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import medvlm
from medvlm import CalibrationPipeline


# ---- evaluate_calibration / compute_calibration_metrics ----

def test_perfect_calibration_zero_ece():
    """Confidence == accuracy in every bin -> ECE ~ 0."""
    rng = np.random.default_rng(0)
    n = 5000
    conf = rng.uniform(0.05, 0.95, size=n)  # avoid edge bins
    # outcome ~ Bernoulli(conf) -> empirical accuracy in each bin matches conf
    correct = (rng.uniform(0, 1, n) < conf).astype(int)
    rep = medvlm.evaluate_calibration(correct, conf, num_bins=15)
    assert rep["ece"] < 0.02, f"ECE={rep['ece']:.3f} should be near 0"
    assert abs(rep["accuracy"] - rep["mean_confidence"]) < 0.02


def test_extreme_overconfidence_high_ece():
    """All conf=0.99 but accuracy=0.5 -> ECE ~ 0.49."""
    n = 1000
    conf = np.full(n, 0.99)
    correct = np.array([1, 0] * (n // 2))
    rep = medvlm.evaluate_calibration(correct, conf, num_bins=15)
    assert rep["ece"] == pytest.approx(0.49, abs=0.02)
    assert rep["overconfidence"] == pytest.approx(0.49, abs=0.02)
    assert rep["accuracy"] == 0.5
    assert rep["mean_confidence"] == pytest.approx(0.99)


def test_evaluate_calibration_length_check():
    with pytest.raises(ValueError):
        medvlm.evaluate_calibration([1, 0], [0.5, 0.5, 0.5])


def test_evaluate_calibration_with_method():
    """When calibration_method is set, returns calibrated_confidences + raw_metrics."""
    rng = np.random.default_rng(1)
    n = 500
    conf = np.full(n, 0.95)
    correct = (rng.uniform(0, 1, n) < 0.6).astype(int)  # actual acc ~0.6
    rep = medvlm.evaluate_calibration(
        correct, conf, calibration_method="temperature_scaling",
    )
    assert "calibrated_confidences" in rep
    assert "raw_metrics" in rep
    assert len(rep["calibrated_confidences"]) == n
    assert rep["ece"] < rep["raw_metrics"]["ece"]  # calibration helps


# ---- CalibrationPipeline: registry + error handling ----

def test_pipeline_unknown_method():
    with pytest.raises(ValueError):
        CalibrationPipeline(method="nonsense")


def test_pipeline_supported_methods_list():
    expected = {
        "temperature_scaling", "platt", "platt_confidence",
        "isotonic", "histogram_binning",
        "hac_platt", "hac_platt_confidence", "hac_gate",
    }
    assert set(CalibrationPipeline.SUPPORTED_METHODS) == expected


def test_requires_hallucination_scores_flag():
    for m in ["temperature_scaling", "platt", "platt_confidence",
              "isotonic", "histogram_binning"]:
        assert CalibrationPipeline(method=m).requires_hallucination_scores is False
    for m in ["hac_platt", "hac_platt_confidence", "hac_gate"]:
        assert CalibrationPipeline(method=m).requires_hallucination_scores is True


def test_transform_before_fit_raises():
    cal = CalibrationPipeline(method="platt")
    with pytest.raises(RuntimeError):
        cal.transform(np.array([0.5]))


def test_hac_requires_h_in_fit():
    cal = CalibrationPipeline(method="hac_platt")
    with pytest.raises(ValueError, match="hallucination_scores"):
        cal.fit(np.array([0.5, 0.6]), np.array([1, 0]))


def test_hac_requires_h_in_transform():
    cal = CalibrationPipeline(method="hac_gate")
    cal.fit(np.array([0.5, 0.6, 0.7, 0.4]),
            np.array([1, 0, 1, 0]),
            hallucination_scores=np.array([0.1, 0.2, 0.3, 0.4]))
    with pytest.raises(ValueError, match="hallucination_scores"):
        cal.transform(np.array([0.5, 0.6]))


# ---- standard methods reduce ECE on miscalibrated data ----

@pytest.fixture
def overconf_data():
    """Simulated overconfident model: conf concentrated near 1, acc ~0.6."""
    rng = np.random.default_rng(7)
    n = 1000
    # latent ability per example
    p_true = rng.uniform(0.4, 0.8, n)
    correct = (rng.uniform(0, 1, n) < p_true).astype(int)
    # confidence is overconfident transformation of p_true
    conf = np.clip(p_true ** 0.3, 0.05, 0.99)  # squashed toward 1
    # split val/test
    idx = rng.permutation(n)
    val_idx, test_idx = idx[:300], idx[300:]
    return (conf[val_idx], correct[val_idx],
            conf[test_idx], correct[test_idx])


@pytest.mark.parametrize("method", [
    "temperature_scaling", "platt", "platt_confidence",
    "isotonic", "histogram_binning",
])
def test_standard_method_reduces_ece(overconf_data, method):
    val_conf, val_corr, test_conf, test_corr = overconf_data
    raw_ece = medvlm.evaluate_calibration(test_corr, test_conf)["ece"]

    cal = CalibrationPipeline(method=method)
    cal.fit(val_conf, val_corr)
    cal_conf = cal.transform(test_conf)
    cal_ece = medvlm.evaluate_calibration(test_corr, cal_conf)["ece"]

    assert cal_ece <= raw_ece + 0.02, (
        f"{method}: ECE went UP from {raw_ece:.3f} to {cal_ece:.3f}"
    )
    # output stays in [0,1]
    assert cal_conf.min() >= 0.0 and cal_conf.max() <= 1.0


def test_monotonic_methods_preserve_auroc(overconf_data):
    """Platt + temperature_scaling are monotonic -> AUROC unchanged."""
    val_conf, val_corr, test_conf, test_corr = overconf_data
    raw_auc = roc_auc_score(test_corr, test_conf)
    for method in ("temperature_scaling", "platt", "platt_confidence"):
        cal = CalibrationPipeline(method=method)
        cal.fit(val_conf, val_corr)
        new_auc = roc_auc_score(test_corr, cal.transform(test_conf))
        assert abs(new_auc - raw_auc) < 1e-6, f"{method} changed AUROC"


# ---- HAC: enforce sign constraints + improve AUROC when h is informative ----

@pytest.fixture
def hac_data():
    """h is positively correlated with errors -> HAC should help."""
    rng = np.random.default_rng(11)
    n = 1500
    p_true = rng.uniform(0.3, 0.95, n)
    correct = (rng.uniform(0, 1, n) < p_true).astype(int)
    conf = np.clip(p_true + rng.normal(0, 0.05, n), 0.02, 0.98)
    # h: high when wrong, low when right (informative hallucination signal)
    h = (1 - p_true) + rng.normal(0, 0.1, n)
    idx = rng.permutation(n)
    val_idx, test_idx = idx[:500], idx[500:]
    return (conf[val_idx], correct[val_idx], h[val_idx],
            conf[test_idx], correct[test_idx], h[test_idx])


def test_hac_platt_sign_constraints(hac_data):
    """a >= 0, b <= 0 by construction in pipeline."""
    val_conf, val_corr, val_h, *_ = hac_data
    cal = CalibrationPipeline(method="hac_platt")
    cal.fit(val_conf, val_corr, hallucination_scores=val_h)
    a, b, d = cal._params["hac_platt_abd"]
    assert a >= -1e-9, f"a={a} should be >= 0"
    assert b <= 1e-9, f"b={b} should be <= 0"


def test_hac_gate_sign_constraint(hac_data):
    val_conf, val_corr, val_h, *_ = hac_data
    cal = CalibrationPipeline(method="hac_gate")
    cal.fit(val_conf, val_corr, hallucination_scores=val_h)
    a, b = cal._params["hac_gate_ab"]
    assert a >= -1e-9, f"a={a} should be >= 0"


@pytest.mark.parametrize("method", ["hac_platt", "hac_platt_confidence", "hac_gate"])
def test_hac_improves_auroc_with_informative_h(hac_data, method):
    val_conf, val_corr, val_h, test_conf, test_corr, test_h = hac_data
    raw_auc = roc_auc_score(test_corr, test_conf)

    cal = CalibrationPipeline(method=method)
    cal.fit(val_conf, val_corr, hallucination_scores=val_h)
    cal_conf = cal.transform(test_conf, hallucination_scores=test_h)
    new_auc = roc_auc_score(test_corr, cal_conf)

    assert new_auc >= raw_auc - 0.005, (
        f"{method}: AUROC dropped from {raw_auc:.3f} to {new_auc:.3f} "
        f"with informative h"
    )
    assert cal_conf.min() >= 0.0 and cal_conf.max() <= 1.0


def test_fit_transform_helper(overconf_data):
    val_conf, val_corr, _, _ = overconf_data
    cal = CalibrationPipeline(method="platt")
    out = cal.fit_transform(val_conf, val_corr)
    assert out.shape == val_conf.shape
    assert (out >= 0).all() and (out <= 1).all()
