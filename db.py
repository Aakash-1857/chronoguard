"""
ChronoGuard v1.0 — SQLite Persistence Layer (M1 + M2)

Manages the SQLite database: schema creation, WAL configuration,
flow prediction writes, and model lineage recording.

M1 schema: flow_predictions (spec §5.4).
M2 schema: model_lineage (M2 spec §5.4) — additive, no breaking changes.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from config import DB_PATH

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema Definition (spec §5.4 — verbatim)
# ---------------------------------------------------------------------------
_CREATE_TABLE_SQL = """\
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
"""

# ---------------------------------------------------------------------------
# M2 Schema Extension — model_lineage (M2 spec §5.4)
# ---------------------------------------------------------------------------
_CREATE_MODEL_LINEAGE_SQL = """\
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
"""

_INSERT_SQL = """\
INSERT OR REPLACE INTO flow_predictions (
    flow_id, src_ip, dst_ip, src_port, dst_port, protocol,
    flow_start_ts, flow_end_ts, flow_duration,
    flow_iat_mean, flow_iat_std,
    fwd_packet_count, bwd_packet_count,
    fwd_packet_length_max, fwd_packet_length_mean,
    bwd_packet_length_max, bwd_packet_length_mean,
    fwd_packets_per_sec, bwd_packets_per_sec,
    termination_reason,
    risk_score, predicted_class, model_version,
    inference_ts, inference_latency_ms
) VALUES (
    :flow_id, :src_ip, :dst_ip, :src_port, :dst_port, :protocol,
    :flow_start_ts, :flow_end_ts, :flow_duration,
    :flow_iat_mean, :flow_iat_std,
    :fwd_packet_count, :bwd_packet_count,
    :fwd_packet_length_max, :fwd_packet_length_mean,
    :bwd_packet_length_max, :bwd_packet_length_mean,
    :fwd_packets_per_sec, :bwd_packets_per_sec,
    :termination_reason,
    :risk_score, :predicted_class, :model_version,
    :inference_ts, :inference_latency_ms
);
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Initialize the SQLite database with WAL mode and create the schema.

    PRAGMA journal_mode=WAL and PRAGMA synchronous=NORMAL are set on
    connection init as required by spec §5.4.  Table creation is
    idempotent (IF NOT EXISTS).

    Returns the open connection (caller is responsible for closing it).
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(_CREATE_TABLE_SQL)
    conn.execute(_CREATE_MODEL_LINEAGE_SQL)
    conn.commit()

    logger.info("SQLite database initialized: %s (WAL mode)", db_path)
    return conn


def insert_prediction(
    conn: sqlite3.Connection,
    record: dict,
    risk_score: float,
    predicted_class: Optional[str],
    model_version: str,
    inference_ts: float,
    inference_latency_ms: float,
) -> None:
    """Insert a flow prediction record into the database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection (from ``init_db``).
    record : dict
        A ``FlowRecord`` dict with all identity/feature fields.
    risk_score : float
        Model-predicted risk score (0.0–1.0).
    predicted_class : str | None
        Predicted class label (for multiclass), or None for binary.
    model_version : str
        Identifier for the model artifact that produced this prediction.
    inference_ts : float
        Wall-clock epoch time when inference was performed.
    inference_latency_ms : float
        Duration of the ONNX inference call in milliseconds.
    """
    params = {
        **record,
        "risk_score": risk_score,
        "predicted_class": predicted_class,
        "model_version": model_version,
        "inference_ts": inference_ts,
        "inference_latency_ms": inference_latency_ms,
    }

    try:
        conn.execute(_INSERT_SQL, params)
        conn.commit()
    except sqlite3.Error:
        # Spec §6: SQLite write failures — log error, continue loop,
        # do not crash the engine thread.
        logger.error(
            "SQLite write failed for flow_id=%s",
            record.get("flow_id", "UNKNOWN"),
            exc_info=True,
        )


def close_db(conn: sqlite3.Connection) -> None:
    """Close the database connection safely."""
    try:
        conn.close()
        logger.info("SQLite connection closed")
    except sqlite3.Error:
        logger.error("Error closing SQLite connection", exc_info=True)
