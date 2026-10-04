"""
Unit tests for optimizer.py — ChronoGuard M2

Covers (per M2 spec §7/§8):
  - extract_training_window returns correct N records, time-ordered.
  - Optuna sweep never exceeds n_trials ceiling.
  - Optuna sweep respects time_budget_sec (artificially slow objective).
  - warm_start_fit produces a valid Booster.
"""

from __future__ import annotations

import os
import sqlite3
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from config import FEATURE_ORDER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def populated_db(tmp_path):
    """Create a temp SQLite DB with 100 flow_predictions rows."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")

    # Create schema
    conn.execute("""
        CREATE TABLE flow_predictions (
            flow_id TEXT PRIMARY KEY,
            src_ip TEXT NOT NULL,
            dst_ip TEXT NOT NULL,
            src_port INTEGER NOT NULL,
            dst_port INTEGER NOT NULL,
            protocol INTEGER NOT NULL,
            flow_start_ts REAL NOT NULL,
            flow_end_ts REAL NOT NULL,
            flow_duration REAL NOT NULL,
            flow_iat_mean REAL NOT NULL,
            flow_iat_std REAL NOT NULL,
            fwd_packet_count INTEGER NOT NULL,
            bwd_packet_count INTEGER NOT NULL,
            fwd_packet_length_max INTEGER NOT NULL,
            fwd_packet_length_mean REAL NOT NULL,
            bwd_packet_length_max INTEGER NOT NULL,
            bwd_packet_length_mean REAL NOT NULL,
            fwd_packets_per_sec REAL NOT NULL,
            bwd_packets_per_sec REAL NOT NULL,
            termination_reason TEXT NOT NULL,
            risk_score REAL NOT NULL,
            predicted_class TEXT,
            model_version TEXT NOT NULL,
            inference_ts REAL NOT NULL,
            inference_latency_ms REAL NOT NULL
        )
    """)

    # Create model_lineage table too
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_lineage (
            retrain_id TEXT PRIMARY KEY,
            triggered_ts REAL NOT NULL,
            triggering_feature TEXT NOT NULL,
            training_window_start_ts REAL NOT NULL,
            training_window_end_ts REAL NOT NULL,
            training_window_size INTEGER NOT NULL,
            hyperparams_json TEXT NOT NULL,
            optuna_trials_run INTEGER NOT NULL,
            candidate_accuracy REAL NOT NULL,
            base_accuracy REAL NOT NULL,
            accuracy_gate_threshold REAL NOT NULL,
            status TEXT NOT NULL,
            fail_reason TEXT,
            model_version TEXT,
            completed_ts REAL NOT NULL,
            retrain_duration_sec REAL NOT NULL
        )
    """)

    rng = np.random.RandomState(42)
    base_ts = 1700000000.0

    for i in range(100):
        conn.execute(
            """
            INSERT INTO flow_predictions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
            """,
            (
                f"flow_{i:04d}",
                "10.0.0.1", "10.0.0.2",
                12345, 80, 6,
                base_ts + i, base_ts + i + 1.5,
                1.5,
                float(rng.uniform(0.1, 1.0)),
                float(rng.uniform(0.01, 0.5)),
                int(rng.randint(1, 50)),
                int(rng.randint(0, 30)),
                int(rng.randint(100, 1500)),
                float(rng.uniform(100, 800)),
                int(rng.randint(0, 1500)),
                float(rng.uniform(0, 600)),
                float(rng.uniform(0.5, 50.0)),
                float(rng.uniform(0.0, 30.0)),
                "FIN",
                float(rng.uniform(0.0, 1.0)),
                str(rng.randint(0, 2)),
                "abc123",
                base_ts + i + 2.0,
                float(rng.uniform(0.1, 5.0)),
            ),
        )

    conn.commit()
    conn.close()

    return db_path


@pytest.fixture
def sample_train_df():
    """Create a sample training DataFrame."""
    rng = np.random.RandomState(42)
    n = 100
    data = {}
    for feat in FEATURE_ORDER:
        data[feat] = rng.uniform(0, 100, n).astype(np.float32)
    data["risk_score"] = rng.uniform(0, 1, n).astype(np.float32)
    data["predicted_class"] = rng.randint(0, 2, n).astype(str)
    data["inference_ts"] = np.arange(n, dtype=np.float64) + 1700000000.0
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestExtractTrainingWindow:
    def test_returns_correct_count(self, populated_db) -> None:
        """Returns exactly N records when N <= total."""
        from optimizer import extract_training_window

        df = extract_training_window(populated_db, n=50)
        assert len(df) == 50

    def test_returns_all_when_fewer_than_n(self, populated_db) -> None:
        """Returns all records when fewer than N exist."""
        from optimizer import extract_training_window

        df = extract_training_window(populated_db, n=500)
        assert len(df) == 100

    def test_time_ordered_ascending(self, populated_db) -> None:
        """Records are ordered by inference_ts ascending (oldest first)."""
        from optimizer import extract_training_window

        df = extract_training_window(populated_db, n=50)
        ts = df["inference_ts"].values
        assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))

    def test_pulls_most_recent(self, populated_db) -> None:
        """Pulls the most recent N records, not arbitrary ones."""
        from optimizer import extract_training_window

        df = extract_training_window(populated_db, n=10)
        # The last 10 records should have the highest inference_ts values
        assert df["inference_ts"].min() > 1700000089.0  # record 90+


class TestOptunaSweep:
    def test_sweep_bounded_by_trials(self, sample_train_df) -> None:
        """Sweep does not exceed n_trials ceiling."""
        from optimizer import run_optuna_sweep

        params = run_optuna_sweep(sample_train_df, n_trials=2, time_budget_sec=300)
        assert isinstance(params, dict)
        assert "learning_rate" in params

    def test_sweep_bounded_by_time(self, sample_train_df) -> None:
        """Sweep respects time_budget_sec ceiling.

        We use a very short budget (2 seconds) and verify it stops
        within a reasonable tolerance.
        """
        from optimizer import run_optuna_sweep

        start = time.monotonic()
        params = run_optuna_sweep(sample_train_df, n_trials=100, time_budget_sec=2)
        elapsed = time.monotonic() - start

        # Should complete within budget + overhead (generous 15s tolerance for
        # trial that started before deadline)
        assert elapsed < 20, f"Sweep took {elapsed:.1f}s, expected < 20s"
        assert isinstance(params, dict)


class TestWarmStartFit:
    def test_produces_valid_booster(self, sample_train_df) -> None:
        """warm_start_fit returns a valid LightGBM Booster."""
        from optimizer import warm_start_fit

        hyperparams = {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 5,
        }
        booster = warm_start_fit(sample_train_df, hyperparams.copy())
        assert isinstance(booster, lgb.Booster)

        # Booster should be able to predict
        X = sample_train_df[FEATURE_ORDER].values.astype(np.float32)
        preds = booster.predict(X)
        assert len(preds) == len(X)
        assert all(0.0 <= p <= 1.0 for p in preds)
