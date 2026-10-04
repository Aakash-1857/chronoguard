"""
ChronoGuard v1.0 — Dashboard Data Access Layer (M3)

The sole module permitted to open a (read-only) SQLite connection to the
core runtime's database.  No component or app.py may import sqlite3 or
open a connection directly — all data flows through this module.

Spec reference: M3 §2 (scope), §3 (read-only enforcement), §6 (data contracts).

Read-only enforcement: get_connection() uses SQLite URI mode=ro.
A test (test_data_access.py) verifies INSERT raises OperationalError.
"""

from __future__ import annotations

import collections
import os
import sqlite3
import time
from typing import Union

import pandas as pd

# Resolve project root so we can find config.LOG_FILE_PATH at import time.
# We do NOT import config at module level to avoid pulling in the full
# runtime dependency graph (scapy, etc.) into the Streamlit process.
# Instead, the log file path is passed in or resolved from env.
_DEFAULT_LOG_FILE = os.environ.get("CHRONOGUARD_LOG_FILE", "chronoguard.log")


# ---------------------------------------------------------------------------
# §6.1 — get_connection
# ---------------------------------------------------------------------------
def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a READ-ONLY connection to the core runtime's SQLite database.

    Uses the SQLite URI ``mode=ro`` flag so any write attempt raises
    ``sqlite3.OperationalError``.  This is the §3 enforcement mechanism.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# §6.1 — get_throughput
# ---------------------------------------------------------------------------
def get_throughput(conn: sqlite3.Connection, window_sec: int = 10) -> float:
    """Flows/sec over the trailing *window_sec*, from ``flow_predictions.inference_ts``."""
    cutoff = time.time() - window_sec
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM flow_predictions WHERE inference_ts > ?",
        (cutoff,),
    ).fetchone()
    count = row["cnt"] if row else 0
    return count / window_sec


# ---------------------------------------------------------------------------
# §6.1 — get_drift_status
# ---------------------------------------------------------------------------
def get_drift_status(conn: sqlite3.Connection) -> dict:
    """Infer current drift state from ``model_lineage``.

    Returns ``{'state': str, 'since_ts': float, 'triggering_feature': str|None}``.

    Inference logic (no dedicated drift-state table exists):
        1. If a recent lineage row has status that is neither 'PASS' nor 'FAIL'
           (indicating an in-progress retrain), state is 'RETRAINING'.
        2. If the most recent completed row was triggered recently (within 60s)
           and has a triggering_feature, state is 'DRIFT_DETECTED'.
        3. Otherwise, state is 'HEALTHY'.
    """
    row = conn.execute(
        "SELECT status, triggered_ts, triggering_feature, completed_ts "
        "FROM model_lineage ORDER BY triggered_ts DESC LIMIT 1"
    ).fetchone()

    if row is None:
        return {
            "state": "HEALTHY",
            "since_ts": 0.0,
            "triggering_feature": None,
        }

    status = row["status"]
    triggered_ts = row["triggered_ts"]
    triggering_feature = row["triggering_feature"]
    completed_ts = row["completed_ts"]

    # In-progress retrain: status is neither terminal outcome.
    # (The optimizer sets status to PASS or FAIL only on completion.)
    if status not in ("PASS", "FAIL"):
        return {
            "state": "RETRAINING",
            "since_ts": triggered_ts,
            "triggering_feature": triggering_feature,
        }

    # Recent drift trigger — within 60 seconds of now and completed.
    now = time.time()
    if completed_ts and (now - completed_ts) < 60:
        return {
            "state": "DRIFT_DETECTED",
            "since_ts": triggered_ts,
            "triggering_feature": triggering_feature,
        }

    return {
        "state": "HEALTHY",
        "since_ts": 0.0,
        "triggering_feature": None,
    }


# ---------------------------------------------------------------------------
# §6.1 — get_active_model_version
# ---------------------------------------------------------------------------
def get_active_model_version(conn: sqlite3.Connection) -> dict:
    """Return the active model's version and timestamp.

    Returns the latest PASS ``model_lineage`` row's ``model_version`` +
    ``completed_ts``, or falls back to the most recent ``model_version``
    in ``flow_predictions`` (the M1 baseline).
    """
    # Try lineage first — latest PASS row.
    row = conn.execute(
        "SELECT model_version, completed_ts FROM model_lineage "
        "WHERE status = 'PASS' ORDER BY completed_ts DESC LIMIT 1"
    ).fetchone()

    if row and row["model_version"]:
        return {
            "model_version": row["model_version"],
            "completed_ts": row["completed_ts"],
        }

    # Fallback: latest flow_predictions model_version (M1 baseline).
    row = conn.execute(
        "SELECT model_version, inference_ts FROM flow_predictions "
        "ORDER BY inference_ts DESC LIMIT 1"
    ).fetchone()

    if row and row["model_version"]:
        return {
            "model_version": row["model_version"],
            "completed_ts": row["inference_ts"],
        }

    return {"model_version": None, "completed_ts": None}


# ---------------------------------------------------------------------------
# §6.1 — get_feature_series
# ---------------------------------------------------------------------------
def get_feature_series(
    conn: sqlite3.Connection,
    feature: str,
    limit: int = 200,
) -> pd.DataFrame:
    """Trailing N values of a monitored feature/score plus timestamps.

    Returns a DataFrame with columns ``['timestamp', 'value']``.
    For ``risk_score``, reads directly from ``flow_predictions.risk_score``.
    For input features, reads from the corresponding column.
    """
    # Validate the feature is a real column to prevent SQL injection.
    # We check against the known schema columns.
    _ALLOWED_FEATURES = {
        "risk_score",
        "flow_duration",
        "fwd_packets_per_sec",
        "bwd_packets_per_sec",
        "flow_iat_mean",
        "flow_iat_std",
        "fwd_packet_count",
        "bwd_packet_count",
        "fwd_packet_length_max",
        "fwd_packet_length_mean",
        "bwd_packet_length_max",
        "bwd_packet_length_mean",
        "inference_latency_ms",
    }
    if feature not in _ALLOWED_FEATURES:
        return pd.DataFrame(columns=["timestamp", "value"])

    query = (
        f"SELECT inference_ts AS timestamp, {feature} AS value "
        f"FROM flow_predictions ORDER BY inference_ts DESC LIMIT ?"
    )
    df = pd.read_sql_query(query, conn, params=(limit,))

    if not df.empty:
        # Reverse so chronological order (oldest first) for charting.
        df = df.iloc[::-1].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# §6.1 — get_recent_lineage
# ---------------------------------------------------------------------------
def get_recent_lineage(
    conn: sqlite3.Connection,
    limit: int = 50,
) -> pd.DataFrame:
    """Most recent ``model_lineage`` rows, newest first."""
    query = (
        "SELECT retrain_id, triggered_ts, triggering_feature, status, "
        "candidate_accuracy, base_accuracy, model_version, completed_ts "
        "FROM model_lineage ORDER BY completed_ts DESC LIMIT ?"
    )
    return pd.read_sql_query(query, conn, params=(limit,))


# ---------------------------------------------------------------------------
# §6.1 — get_log_tail
# ---------------------------------------------------------------------------
def get_log_tail(
    conn_or_path: Union[sqlite3.Connection, str, None] = None,
    limit: int = 100,
) -> list[str]:
    """Most recent structured log lines from the runtime log file.

    Per §6.2, this reads the file's tail — it does not invent a new
    logging channel.  ``conn_or_path`` accepts either the path string
    directly or is ignored (falls back to ``_DEFAULT_LOG_FILE``).
    """
    if isinstance(conn_or_path, str):
        log_path = conn_or_path
    else:
        log_path = _DEFAULT_LOG_FILE

    if not os.path.isfile(log_path):
        return []

    try:
        # Efficient tail read: read from end of file.
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            # For moderate log files, deque is efficient enough.
            lines = collections.deque(f, maxlen=limit)
        return [line.rstrip("\n") for line in lines]
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Helper: extract drift event timestamps from lineage
# ---------------------------------------------------------------------------
def get_drift_event_timestamps(
    conn: sqlite3.Connection,
    limit: int = 50,
) -> list[float]:
    """Return ``triggered_ts`` values from model_lineage for chart vertical rules."""
    rows = conn.execute(
        "SELECT triggered_ts FROM model_lineage "
        "ORDER BY triggered_ts DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [row["triggered_ts"] for row in rows]
