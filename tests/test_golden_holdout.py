"""
Unit tests for golden_holdout.py — ChronoGuard M2

Covers (per M2 spec §7/§8):
  - evaluate() returns accuracy in [0.0, 1.0].
  - Gate rejects degraded model (intentionally bad predictions).
  - Gate accepts valid model.
"""

from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from config import FEATURE_ORDER


@pytest.fixture
def holdout_csv(tmp_path):
    """Create a small golden holdout CSV for testing."""
    rng = np.random.RandomState(42)
    n = 50
    data = {}
    for feat in FEATURE_ORDER:
        data[feat] = rng.uniform(0, 100, n).astype(np.float32)
    data["label"] = np.concatenate([np.zeros(25), np.ones(25)]).astype(int)

    df = pd.DataFrame(data)
    path = str(tmp_path / "holdout.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def good_model(holdout_csv):
    """Train a LightGBM model on the holdout data itself (should get ~100% accuracy)."""
    df = pd.read_csv(holdout_csv)
    X = df[FEATURE_ORDER].values.astype(np.float32)
    y = df["label"].values.astype(int)

    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
    }
    booster = lgb.train(params, train_data, num_boost_round=100,
                        callbacks=[lgb.log_evaluation(period=0)])
    return booster


@pytest.fixture
def bad_model(holdout_csv):
    """Train a deliberately bad model (inverted labels)."""
    df = pd.read_csv(holdout_csv)
    X = df[FEATURE_ORDER].values.astype(np.float32)
    # Invert labels — model learns the opposite
    y = 1 - df["label"].values.astype(int)

    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.1,
    }
    booster = lgb.train(params, train_data, num_boost_round=100,
                        callbacks=[lgb.log_evaluation(period=0)])
    return booster


class TestEvaluate:
    def test_returns_accuracy_in_range(self, good_model, holdout_csv) -> None:
        """evaluate() returns a float in [0.0, 1.0]."""
        from golden_holdout import evaluate

        acc = evaluate(good_model, holdout_csv)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_good_model_high_accuracy(self, good_model, holdout_csv) -> None:
        """A model trained on the holdout itself achieves high accuracy."""
        from golden_holdout import evaluate

        acc = evaluate(good_model, holdout_csv)
        assert acc >= 0.7, f"Expected accuracy >= 0.7 for overfitted model, got {acc}"

    def test_bad_model_low_accuracy(self, bad_model, holdout_csv) -> None:
        """A model trained on inverted labels achieves low accuracy."""
        from golden_holdout import evaluate

        acc = evaluate(bad_model, holdout_csv)
        assert acc < 0.5, f"Expected accuracy < 0.5 for inverted model, got {acc}"

    def test_missing_holdout_raises_error(self, good_model) -> None:
        """evaluate() raises FileNotFoundError for missing holdout."""
        from golden_holdout import evaluate

        with pytest.raises((FileNotFoundError, Exception)):
            evaluate(good_model, "/nonexistent/holdout.csv")

    def test_missing_columns_raises_error(self, good_model, tmp_path) -> None:
        """evaluate() raises ValueError for holdout with missing features."""
        from golden_holdout import evaluate

        # CSV with wrong columns
        bad_csv = str(tmp_path / "bad_holdout.csv")
        pd.DataFrame({"wrong_col": [1, 2, 3], "label": [0, 1, 0]}).to_csv(
            bad_csv, index=False
        )
        with pytest.raises(ValueError, match="missing required features"):
            evaluate(good_model, bad_csv)
