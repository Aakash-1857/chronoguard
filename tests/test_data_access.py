"""
Tests for M3 data access layer — ChronoGuard

Covers (per M3 spec §8, §9):
    - Read-only connection enforcement (§3, §7, §8 criterion #2).
    - Each data_access.py function against a fixture DB.
    - Empty-state handling (§8 criterion #8).
    - get_log_tail against a fixture log file.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import time

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def fixture_db(tmp_path):
    """Create a fixture SQLite DB with flow_predictions and model_lineage tables,
    seeded with test data."""
    db_path = str(tmp_path / "test_chronoguard.db")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")

    # Schema from db.py (verbatim).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flow_predictions (
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
        );
    """)

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
        );
    """)

    # Seed flow_predictions.
    now = time.time()
    for i in range(20):
        conn.execute(
            """INSERT INTO flow_predictions (
                flow_id, src_ip, dst_ip, src_port, dst_port, protocol,
                flow_start_ts, flow_end_ts, flow_duration,
                flow_iat_mean, flow_iat_std,
                fwd_packet_count, bwd_packet_count,
                fwd_packet_length_max, fwd_packet_length_mean,
                bwd_packet_length_max, bwd_packet_length_mean,
                fwd_packets_per_sec, bwd_packets_per_sec,
                termination_reason, risk_score, predicted_class,
                model_version, inference_ts, inference_latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"flow_{i:04d}",
                "10.0.0.1", "10.0.0.2", 12345, 80, 6,
                now - 100 + i, now - 99 + i, 1.0,
                0.3, 0.05,
                5, 3,
                1500, 750.0,
                600, 300.0,
                3.33 + i * 0.1, 2.0,
                "FIN", 0.1 + i * 0.04, "0",
                "abc123def456", now - 5 + i * 0.25, 1.5,
            ),
        )

    # Seed model_lineage with one PASS row.
    conn.execute(
        """INSERT INTO model_lineage (
            retrain_id, triggered_ts, triggering_feature,
            training_window_start_ts, training_window_end_ts, training_window_size,
            hyperparams_json, optuna_trials_run,
            candidate_accuracy, base_accuracy, accuracy_gate_threshold,
            status, fail_reason, model_version,
            completed_ts, retrain_duration_sec
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "retrain_001", now - 30, "risk_score",
            now - 2100, now - 100, 2000,
            '{"n_estimators": 100}', 3,
            0.97, 0.95, 0.95,
            "PASS", None, "new_model_v1",
            now - 25, 5.0,
        ),
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def empty_db(tmp_path):
    """Create an empty fixture DB with schema but no data."""
    db_path = str(tmp_path / "empty_chronoguard.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flow_predictions (
            flow_id TEXT PRIMARY KEY,
            src_ip TEXT NOT NULL, dst_ip TEXT NOT NULL,
            src_port INTEGER NOT NULL, dst_port INTEGER NOT NULL,
            protocol INTEGER NOT NULL,
            flow_start_ts REAL NOT NULL, flow_end_ts REAL NOT NULL,
            flow_duration REAL NOT NULL, flow_iat_mean REAL NOT NULL,
            flow_iat_std REAL NOT NULL,
            fwd_packet_count INTEGER NOT NULL, bwd_packet_count INTEGER NOT NULL,
            fwd_packet_length_max INTEGER NOT NULL, fwd_packet_length_mean REAL NOT NULL,
            bwd_packet_length_max INTEGER NOT NULL, bwd_packet_length_mean REAL NOT NULL,
            fwd_packets_per_sec REAL NOT NULL, bwd_packets_per_sec REAL NOT NULL,
            termination_reason TEXT NOT NULL, risk_score REAL NOT NULL,
            predicted_class TEXT, model_version TEXT NOT NULL,
            inference_ts REAL NOT NULL, inference_latency_ms REAL NOT NULL
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_lineage (
            retrain_id TEXT PRIMARY KEY,
            triggered_ts REAL NOT NULL, triggering_feature TEXT NOT NULL,
            training_window_start_ts REAL NOT NULL, training_window_end_ts REAL NOT NULL,
            training_window_size INTEGER NOT NULL, hyperparams_json TEXT NOT NULL,
            optuna_trials_run INTEGER NOT NULL, candidate_accuracy REAL NOT NULL,
            base_accuracy REAL NOT NULL, accuracy_gate_threshold REAL NOT NULL,
            status TEXT NOT NULL, fail_reason TEXT, model_version TEXT,
            completed_ts REAL NOT NULL, retrain_duration_sec REAL NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def fixture_log_file(tmp_path):
    """Create a fixture log file with sample lines."""
    log_path = str(tmp_path / "test.log")
    with open(log_path, "w") as f:
        for i in range(50):
            level = "INFO" if i % 5 != 0 else "ERROR"
            f.write(f"2026-08-03T10:{i:02d}:00 [{level}] test: Message {i}\n")
    return log_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestReadOnlyEnforcement:
    """§3, §7, §8 criterion #2: read-only connection must reject writes."""

    def test_readonly_connection_rejects_insert(self, fixture_db):
        """INSERT through a read-only connection must raise OperationalError."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection

        conn = get_connection(fixture_db)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute(
                    "INSERT INTO flow_predictions (flow_id, src_ip, dst_ip, "
                    "src_port, dst_port, protocol, flow_start_ts, flow_end_ts, "
                    "flow_duration, flow_iat_mean, flow_iat_std, "
                    "fwd_packet_count, bwd_packet_count, "
                    "fwd_packet_length_max, fwd_packet_length_mean, "
                    "bwd_packet_length_max, bwd_packet_length_mean, "
                    "fwd_packets_per_sec, bwd_packets_per_sec, "
                    "termination_reason, risk_score, predicted_class, "
                    "model_version, inference_ts, inference_latency_ms) "
                    "VALUES ('evil', '1.1.1.1', '2.2.2.2', 1, 2, 6, "
                    "0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'FIN', "
                    "0.5, '0', 'v1', 0, 0)"
                )
        finally:
            conn.close()

    def test_readonly_connection_rejects_update(self, fixture_db):
        """UPDATE through a read-only connection must raise OperationalError."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection

        conn = get_connection(fixture_db)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute(
                    "UPDATE flow_predictions SET risk_score = 1.0"
                )
        finally:
            conn.close()

    def test_readonly_connection_rejects_delete(self, fixture_db):
        """DELETE through a read-only connection must raise OperationalError."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection

        conn = get_connection(fixture_db)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("DELETE FROM flow_predictions")
        finally:
            conn.close()

    def test_readonly_connection_allows_select(self, fixture_db):
        """SELECT through a read-only connection must succeed."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection

        conn = get_connection(fixture_db)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM flow_predictions"
            ).fetchone()
            assert row["cnt"] == 20
        finally:
            conn.close()


class TestGetThroughput:
    """Test get_throughput against fixture DB."""

    def test_returns_float(self, fixture_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_throughput

        conn = get_connection(fixture_db)
        try:
            result = get_throughput(conn, window_sec=10)
            assert isinstance(result, float)
            assert result >= 0.0
        finally:
            conn.close()

    def test_empty_db_returns_zero(self, empty_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_throughput

        conn = get_connection(empty_db)
        try:
            result = get_throughput(conn)
            assert result == 0.0
        finally:
            conn.close()


class TestGetDriftStatus:
    """Test get_drift_status inference logic."""

    def test_returns_healthy_on_empty(self, empty_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_drift_status

        conn = get_connection(empty_db)
        try:
            result = get_drift_status(conn)
            assert result["state"] == "HEALTHY"
            assert result["triggering_feature"] is None
        finally:
            conn.close()

    def test_returns_dict_with_required_keys(self, fixture_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_drift_status

        conn = get_connection(fixture_db)
        try:
            result = get_drift_status(conn)
            assert "state" in result
            assert "since_ts" in result
            assert "triggering_feature" in result
            assert result["state"] in ("HEALTHY", "DRIFT_DETECTED", "RETRAINING")
        finally:
            conn.close()


class TestGetActiveModelVersion:
    """Test get_active_model_version."""

    def test_returns_pass_model(self, fixture_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_active_model_version

        conn = get_connection(fixture_db)
        try:
            result = get_active_model_version(conn)
            assert result["model_version"] == "new_model_v1"
        finally:
            conn.close()

    def test_empty_db_returns_none(self, empty_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_active_model_version

        conn = get_connection(empty_db)
        try:
            result = get_active_model_version(conn)
            assert result["model_version"] is None
        finally:
            conn.close()


class TestGetFeatureSeries:
    """Test get_feature_series."""

    def test_returns_dataframe(self, fixture_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_feature_series

        conn = get_connection(fixture_db)
        try:
            df = get_feature_series(conn, "risk_score", limit=10)
            assert isinstance(df, pd.DataFrame)
            assert "timestamp" in df.columns
            assert "value" in df.columns
            assert len(df) <= 10
        finally:
            conn.close()

    def test_invalid_feature_returns_empty(self, fixture_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_feature_series

        conn = get_connection(fixture_db)
        try:
            df = get_feature_series(conn, "DROP TABLE; --")
            assert df.empty
        finally:
            conn.close()

    def test_empty_db_returns_empty_df(self, empty_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_feature_series

        conn = get_connection(empty_db)
        try:
            df = get_feature_series(conn, "risk_score")
            assert df.empty
        finally:
            conn.close()


class TestGetRecentLineage:
    """Test get_recent_lineage."""

    def test_returns_dataframe(self, fixture_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_recent_lineage

        conn = get_connection(fixture_db)
        try:
            df = get_recent_lineage(conn)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert df.iloc[0]["status"] == "PASS"
        finally:
            conn.close()

    def test_empty_db_returns_empty(self, empty_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_recent_lineage

        conn = get_connection(empty_db)
        try:
            df = get_recent_lineage(conn)
            assert df.empty
        finally:
            conn.close()


class TestGetLogTail:
    """Test get_log_tail."""

    def test_reads_lines(self, fixture_log_file):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_log_tail

        lines = get_log_tail(fixture_log_file, limit=10)
        assert len(lines) == 10
        # Should be the last 10 lines.
        assert "Message 49" in lines[-1]

    def test_missing_file_returns_empty(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_log_tail

        lines = get_log_tail("/nonexistent/path/to/log.log")
        assert lines == []

    def test_reads_all_if_limit_exceeds_file(self, fixture_log_file):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_log_tail

        lines = get_log_tail(fixture_log_file, limit=1000)
        assert len(lines) == 50


class TestGetDriftEventTimestamps:
    """Test get_drift_event_timestamps."""

    def test_returns_timestamps(self, fixture_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_drift_event_timestamps

        conn = get_connection(fixture_db)
        try:
            timestamps = get_drift_event_timestamps(conn)
            assert isinstance(timestamps, list)
            assert len(timestamps) == 1
            assert isinstance(timestamps[0], float)
        finally:
            conn.close()

    def test_empty_db_returns_empty(self, empty_db):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ui.data_access import get_connection, get_drift_event_timestamps

        conn = get_connection(empty_db)
        try:
            timestamps = get_drift_event_timestamps(conn)
            assert timestamps == []
        finally:
            conn.close()
