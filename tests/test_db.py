"""
Unit tests for db.py — ChronoGuard M1

Covers (per spec §8 deliverables):
  - SQLite schema idempotency (calling init_db twice doesn't error)
  - WAL mode is active
  - Insert + query round-trip validates all columns
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from db import close_db, init_db, insert_prediction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def db_conn(tmp_path):
    """Create an in-memory-like SQLite connection via init_db."""
    db_path = str(tmp_path / "test_chronoguard.db")
    conn = init_db(db_path)
    yield conn
    close_db(conn)


def _sample_record() -> dict:
    """A valid FlowRecord dict with all required fields."""
    return {
        "flow_id": "abc123def456",
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "src_port": 12345,
        "dst_port": 80,
        "protocol": 6,
        "flow_start_ts": 1700000000.0,
        "flow_end_ts": 1700000001.5,
        "flow_duration": 1.5,
        "flow_iat_mean": 0.5,
        "flow_iat_std": 0.1,
        "fwd_packet_count": 5,
        "bwd_packet_count": 3,
        "fwd_packet_length_max": 1500,
        "fwd_packet_length_mean": 750.0,
        "bwd_packet_length_max": 600,
        "bwd_packet_length_mean": 300.0,
        "fwd_packets_per_sec": 3.33,
        "bwd_packets_per_sec": 2.0,
        "termination_reason": "FIN",
    }


# ---------------------------------------------------------------------------
# Schema Idempotency Tests
# ---------------------------------------------------------------------------
class TestSchemaIdempotency:
    """Spec §5.4: Table creation must be idempotent (IF NOT EXISTS)."""

    def test_init_db_twice_no_error(self, tmp_path) -> None:
        db_path = str(tmp_path / "test.db")
        conn1 = init_db(db_path)
        # Second call should not raise
        conn2 = init_db(db_path)
        close_db(conn1)
        close_db(conn2)

    def test_table_exists_after_init(self, db_conn) -> None:
        cursor = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='flow_predictions'"
        )
        assert cursor.fetchone() is not None


# ---------------------------------------------------------------------------
# WAL Mode Tests
# ---------------------------------------------------------------------------
class TestWALMode:
    """Spec §5.4: PRAGMA journal_mode=WAL must be set."""

    def test_wal_mode_active(self, db_conn) -> None:
        cursor = db_conn.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"

    def test_synchronous_normal(self, db_conn) -> None:
        cursor = db_conn.execute("PRAGMA synchronous;")
        value = cursor.fetchone()[0]
        # synchronous=NORMAL is value 1
        assert value == 1


# ---------------------------------------------------------------------------
# Insert + Round-Trip Tests
# ---------------------------------------------------------------------------
class TestInsertRoundTrip:
    """Verify all columns survive an insert → query round-trip."""

    def test_insert_and_read_all_columns(self, db_conn) -> None:
        record = _sample_record()
        now = time.time()

        insert_prediction(
            conn=db_conn,
            record=record,
            risk_score=0.87,
            predicted_class="attack",
            model_version="v_test_001",
            inference_ts=now,
            inference_latency_ms=1.23,
        )

        cursor = db_conn.execute(
            "SELECT * FROM flow_predictions WHERE flow_id = ?",
            (record["flow_id"],),
        )
        row = cursor.fetchone()
        assert row is not None

        # Map column names
        col_names = [desc[0] for desc in cursor.description]
        row_dict = dict(zip(col_names, row))

        # Verify identity fields
        assert row_dict["flow_id"] == record["flow_id"]
        assert row_dict["src_ip"] == record["src_ip"]
        assert row_dict["dst_ip"] == record["dst_ip"]
        assert row_dict["src_port"] == record["src_port"]
        assert row_dict["dst_port"] == record["dst_port"]
        assert row_dict["protocol"] == record["protocol"]

        # Verify numeric features
        assert row_dict["flow_duration"] == pytest.approx(1.5)
        assert row_dict["flow_iat_mean"] == pytest.approx(0.5)
        assert row_dict["flow_iat_std"] == pytest.approx(0.1)
        assert row_dict["fwd_packet_count"] == 5
        assert row_dict["bwd_packet_count"] == 3

        # Verify prediction fields
        assert row_dict["risk_score"] == pytest.approx(0.87)
        assert row_dict["predicted_class"] == "attack"
        assert row_dict["model_version"] == "v_test_001"
        assert row_dict["inference_latency_ms"] == pytest.approx(1.23)
        assert row_dict["termination_reason"] == "FIN"

    def test_no_null_values_in_not_null_columns(self, db_conn) -> None:
        """Spec §7: all records must have non-null, schema-valid values for every column."""
        record = _sample_record()
        insert_prediction(
            conn=db_conn,
            record=record,
            risk_score=0.5,
            predicted_class=None,  # predicted_class allows NULL in schema
            model_version="v1",
            inference_ts=time.time(),
            inference_latency_ms=0.5,
        )

        cursor = db_conn.execute(
            "SELECT * FROM flow_predictions WHERE flow_id = ?",
            (record["flow_id"],),
        )
        row = cursor.fetchone()
        col_names = [desc[0] for desc in cursor.description]
        row_dict = dict(zip(col_names, row))

        # All NOT NULL columns must have values
        not_null_cols = [
            "flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
            "flow_start_ts", "flow_end_ts", "flow_duration",
            "flow_iat_mean", "flow_iat_std",
            "fwd_packet_count", "bwd_packet_count",
            "fwd_packet_length_max", "fwd_packet_length_mean",
            "bwd_packet_length_max", "bwd_packet_length_mean",
            "fwd_packets_per_sec", "bwd_packets_per_sec",
            "termination_reason", "risk_score",
            "model_version", "inference_ts", "inference_latency_ms",
        ]
        for col in not_null_cols:
            assert row_dict[col] is not None, f"NOT NULL column '{col}' is NULL"

    def test_duplicate_flow_id_replaces(self, db_conn) -> None:
        """INSERT OR REPLACE should handle duplicate flow_ids gracefully."""
        record = _sample_record()
        now = time.time()

        insert_prediction(
            conn=db_conn, record=record, risk_score=0.5,
            predicted_class="benign", model_version="v1",
            inference_ts=now, inference_latency_ms=1.0,
        )
        insert_prediction(
            conn=db_conn, record=record, risk_score=0.9,
            predicted_class="attack", model_version="v2",
            inference_ts=now + 1, inference_latency_ms=0.8,
        )

        cursor = db_conn.execute(
            "SELECT risk_score, model_version FROM flow_predictions WHERE flow_id = ?",
            (record["flow_id"],),
        )
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0] == pytest.approx(0.9)
        assert rows[0][1] == "v2"
