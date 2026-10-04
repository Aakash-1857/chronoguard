"""
Integration tests for M3 — ChronoGuard

Covers (per M3 spec §8, §9):
    - Full dashboard render against a fixture DB with seeded data.
    - Drift-fire-to-chart assertion: a lineage row with drift event
      is reflected in get_drift_event_timestamps output.
    - Model version update: a new PASS lineage row updates
      get_active_model_version.
    - Read-only enforcement verified end-to-end.
    - Empty-state rendering against an empty DB.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time

import pandas as pd
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ui.data_access import (
    get_active_model_version,
    get_connection,
    get_drift_event_timestamps,
    get_drift_status,
    get_feature_series,
    get_log_tail,
    get_recent_lineage,
    get_throughput,
)


# ---------------------------------------------------------------------------
# Fixture: populated DB
# ---------------------------------------------------------------------------
@pytest.fixture()
def live_fixture_db(tmp_path):
    """Create a fixture DB simulating a live M1+M2 runtime."""
    db_path = str(tmp_path / "live_chronoguard.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")

    # Create schema.
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

    # Seed 100 flow records over the last 30 seconds.
    now = time.time()
    for i in range(100):
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
                f"live_flow_{i:04d}",
                "10.0.0.1", "10.0.0.2", 12345 + i, 80, 6,
                now - 30 + i * 0.3, now - 29.5 + i * 0.3, 0.5,
                0.2, 0.03,
                4, 2,
                1400, 700.0,
                500, 250.0,
                8.0, 4.0,
                "FIN", 0.1 + (i % 10) * 0.08, "0",
                "baseline_abc", now - 30 + i * 0.3, 1.2,
            ),
        )

    # Seed lineage: one PASS retrain, one FAIL retrain, one drift event.
    conn.execute(
        """INSERT INTO model_lineage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "retrain_pass_001", now - 60, "risk_score",
            now - 2060, now - 60, 2000,
            '{"n_estimators": 100}', 3,
            0.97, 0.95, 0.95,
            "PASS", None, "new_model_v1",
            now - 55, 5.0,
        ),
    )
    conn.execute(
        """INSERT INTO model_lineage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "retrain_fail_001", now - 20, "flow_duration",
            now - 2020, now - 20, 2000,
            '{"n_estimators": 50}', 3,
            0.80, 0.95, 0.95,
            "FAIL", "accuracy_below_gate", None,
            now - 15, 5.0,
        ),
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def fixture_log_file(tmp_path):
    """Create a populated log file."""
    log_path = str(tmp_path / "chronoguard.log")
    with open(log_path, "w") as f:
        for i in range(200):
            level = ["INFO", "DEBUG", "WARNING", "ERROR"][i % 4]
            f.write(f"2026-08-03T12:{i % 60:02d}:00 [{level}] engine: Flow processed {i}\n")
    return log_path


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------
class TestDashboardDataPipeline:
    """Full data pipeline test against a populated fixture DB."""

    def test_all_data_functions_return_valid_data(self, live_fixture_db):
        """All six data functions must return valid, non-error results."""
        conn = get_connection(live_fixture_db)
        try:
            throughput = get_throughput(conn, window_sec=60)
            assert isinstance(throughput, float)
            assert throughput > 0

            drift = get_drift_status(conn)
            assert drift["state"] in ("HEALTHY", "DRIFT_DETECTED", "RETRAINING")

            model = get_active_model_version(conn)
            assert model["model_version"] == "new_model_v1"

            series = get_feature_series(conn, "risk_score", limit=50)
            assert isinstance(series, pd.DataFrame)
            assert len(series) == 50

            lineage = get_recent_lineage(conn, limit=10)
            assert isinstance(lineage, pd.DataFrame)
            assert len(lineage) == 2  # One PASS, one FAIL.

            events = get_drift_event_timestamps(conn)
            assert len(events) == 2
        finally:
            conn.close()


class TestDriftFireToChart:
    """§8 criterion #4: drift-fire event must appear as vertical rule data."""

    def test_drift_event_timestamp_in_events(self, live_fixture_db):
        """A lineage row's triggered_ts must appear in get_drift_event_timestamps."""
        conn = get_connection(live_fixture_db)
        try:
            events = get_drift_event_timestamps(conn)
            lineage = get_recent_lineage(conn)

            # All lineage triggered_ts values should appear in events.
            lineage_triggers = set(lineage["triggered_ts"].tolist())
            event_set = set(events)
            assert lineage_triggers.issubset(event_set), (
                f"Missing drift events: {lineage_triggers - event_set}"
            )
        finally:
            conn.close()


class TestModelVersionUpdate:
    """§8 criterion #5: model hot-swap updates active version."""

    def test_new_pass_updates_version(self, live_fixture_db):
        """Inserting a new PASS lineage row must update get_active_model_version."""
        # First, verify current version.
        conn = get_connection(live_fixture_db)
        try:
            model_before = get_active_model_version(conn)
            assert model_before["model_version"] == "new_model_v1"
        finally:
            conn.close()

        # Insert a newer PASS row (via a write connection — this simulates
        # the optimizer thread).
        write_conn = sqlite3.connect(live_fixture_db)
        now = time.time()
        write_conn.execute(
            """INSERT INTO model_lineage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "retrain_pass_002", now - 5, "fwd_packets_per_sec",
                now - 2005, now - 5, 2000,
                '{"n_estimators": 150}', 3,
                0.98, 0.96, 0.95,
                "PASS", None, "new_model_v2",
                now - 2, 3.0,
            ),
        )
        write_conn.commit()
        write_conn.close()

        # Re-read via read-only connection.
        conn = get_connection(live_fixture_db)
        try:
            model_after = get_active_model_version(conn)
            assert model_after["model_version"] == "new_model_v2"
        finally:
            conn.close()


class TestReadOnlyEndToEnd:
    """§8 criterion #2: dashboard never writes to core DB."""

    def test_full_pipeline_is_readonly(self, live_fixture_db):
        """Run all data functions and verify DB is unchanged afterward."""
        # Count rows before.
        check_conn = sqlite3.connect(live_fixture_db)
        before_flows = check_conn.execute(
            "SELECT COUNT(*) FROM flow_predictions"
        ).fetchone()[0]
        before_lineage = check_conn.execute(
            "SELECT COUNT(*) FROM model_lineage"
        ).fetchone()[0]
        check_conn.close()

        # Run all dashboard data functions.
        conn = get_connection(live_fixture_db)
        try:
            get_throughput(conn)
            get_drift_status(conn)
            get_active_model_version(conn)
            get_feature_series(conn, "risk_score")
            get_recent_lineage(conn)
            get_drift_event_timestamps(conn)
        finally:
            conn.close()

        # Count rows after — must be identical.
        check_conn = sqlite3.connect(live_fixture_db)
        after_flows = check_conn.execute(
            "SELECT COUNT(*) FROM flow_predictions"
        ).fetchone()[0]
        after_lineage = check_conn.execute(
            "SELECT COUNT(*) FROM model_lineage"
        ).fetchone()[0]
        check_conn.close()

        assert before_flows == after_flows
        assert before_lineage == after_lineage


class TestLogTailIntegration:
    """Test get_log_tail in integration context."""

    def test_reads_populated_log(self, fixture_log_file):
        lines = get_log_tail(fixture_log_file, limit=100)
        assert len(lines) == 100
        # Lines should be the last 100 of 200.
        assert "Flow processed 199" in lines[-1]

    def test_log_levels_present(self, fixture_log_file):
        lines = get_log_tail(fixture_log_file, limit=200)
        text = "\n".join(lines)
        assert "[INFO]" in text
        assert "[ERROR]" in text
        assert "[WARNING]" in text


class TestEmptyStateIntegration:
    """§8 criterion #8: empty-state rendering on zero data."""

    def test_empty_db_all_functions(self, tmp_path):
        """All data functions handle empty DB gracefully."""
        db_path = str(tmp_path / "empty.db")
        conn_setup = sqlite3.connect(db_path)
        conn_setup.execute("""
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
        conn_setup.execute("""
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
        conn_setup.commit()
        conn_setup.close()

        conn = get_connection(db_path)
        try:
            assert get_throughput(conn) == 0.0
            assert get_drift_status(conn)["state"] == "HEALTHY"
            assert get_active_model_version(conn)["model_version"] is None
            assert get_feature_series(conn, "risk_score").empty
            assert get_recent_lineage(conn).empty
            assert get_drift_event_timestamps(conn) == []
        finally:
            conn.close()

        assert get_log_tail("/nonexistent/path/log.log") == []
