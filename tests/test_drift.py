"""
Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from app.drift.multivariate import multivariate_drift_test
from app.drift.ks_test import ks_drift_test
from app.drift.monitor import DriftMonitor
from app.drift.psi import psi_categorical, psi_numeric
from app.exceptions import InsufficientDataError, SchemaMismatchError

np.random.seed(42)


def test_ks_no_drift_same_distribution():
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0, 1, 1000)
    result = ks_drift_test(ref, cur, "feat_a")
    assert result["drift_detected"] is False


def test_ks_detects_mean_shift():
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(3, 1, 1000)
    result = ks_drift_test(ref, cur, "feat_a")
    assert result["drift_detected"] is True


def test_ks_insufficient_data_raises():
    with pytest.raises(InsufficientDataError):
        ks_drift_test(np.array([1, 2, 3]), np.array([1, 2, 3, 4, 5] * 5), "feat_a")


def test_psi_numeric_stable():
    ref = np.random.normal(0, 1, 2000)
    cur = np.random.normal(0, 1, 2000)
    result = psi_numeric(ref, cur, "feat_b")
    assert result["severity"] == "none"


def test_psi_numeric_major_shift():
    ref = np.random.normal(0, 1, 2000)
    cur = np.random.normal(5, 1, 2000)
    result = psi_numeric(ref, cur, "feat_b")
    assert result["severity"] == "major"


def test_psi_categorical_unseen_category():
    ref = pd.Series(["a"] * 500 + ["b"] * 500)
    cur = pd.Series(["a"] * 400 + ["b"] * 400 + ["c"] * 200)
    result = psi_categorical(ref, cur, "feat_cat")
    assert result["unseen_category_share"] > 0


def test_monitor_full_report_and_schema_validation():
    ref_df = pd.DataFrame({
        "age": np.random.normal(35, 5, 500),
        "city": np.random.choice(["A", "B"], 500),
    })
    cur_df = pd.DataFrame({
        "age": np.random.normal(35, 5, 500),
        "city": np.random.choice(["A", "B"], 500),
    })
    monitor = DriftMonitor(numeric_features=["age"], categorical_features=["city"])
    report = monitor.run(ref_df, cur_df)
    assert report["n_features_checked"] == 2
    assert "errors" in report


def test_monitor_raises_on_missing_column():
    ref_df = pd.DataFrame({"age": [1, 2, 3]})
    cur_df = pd.DataFrame({"age": [1, 2, 3]})
    monitor = DriftMonitor(numeric_features=["age", "income"], categorical_features=[])
    with pytest.raises(SchemaMismatchError):
        monitor.run(ref_df, cur_df)

def test_multivariate_detects_correlation_flip():
    n = 200
    ref_age = np.random.uniform(20, 60, n)
    ref_income = 20000 + ref_age * 800 + np.random.normal(0, 3000, n)
    ref_df = pd.DataFrame({"age": ref_age, "income": ref_income})

    cur_age = np.random.uniform(20, 60, n)
    cur_income = 20000 + (80 - cur_age) * 800 + np.random.normal(0, 3000, n)
    cur_df = pd.DataFrame({"age": cur_age, "income": cur_income})

    result = multivariate_drift_test(ref_df, cur_df, numeric_features=["age", "income"], categorical_features=[])
    assert result["drift_detected"] is True
    assert result["auc"] > 0.65


def test_multivariate_no_drift_when_same_relationship():
    n = 200
    ref_age = np.random.uniform(20, 60, n)
    ref_income = 20000 + ref_age * 800 + np.random.normal(0, 3000, n)
    ref_df = pd.DataFrame({"age": ref_age, "income": ref_income})

    cur_age = np.random.uniform(20, 60, n)
    cur_income = 20000 + cur_age * 800 + np.random.normal(0, 3000, n)  # SAME relationship
    cur_df = pd.DataFrame({"age": cur_age, "income": cur_income})

    result = multivariate_drift_test(ref_df, cur_df, numeric_features=["age", "income"], categorical_features=[])
    assert result["drift_detected"] is False


def test_multivariate_insufficient_data_raises():
    ref_df = pd.DataFrame({"age": [1, 2, 3, 4, 5]})
    cur_df = pd.DataFrame({"age": [1, 2, 3, 4, 5]})
    with pytest.raises(InsufficientDataError):
        multivariate_drift_test(ref_df, cur_df, numeric_features=["age"], categorical_features=[])        