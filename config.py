"""
ChronoGuard v1.0 — Centralized Configuration (M1)

All runtime-configurable parameters are defined here.
FEATURE_ORDER defines the fixed, documented feature ordering for ONNX model input
(spec §5.3). It must reference only numeric fields present in FlowRecord.
"""

import os

# ---------------------------------------------------------------------------
# Network Capture
# ---------------------------------------------------------------------------
# Loopback interface name. macOS uses "lo0"; Linux uses "lo".
# Overridable via environment variable for cross-platform flexibility.
INTERFACE: str = os.environ.get("CHRONOGUARD_INTERFACE", "lo0")

# ---------------------------------------------------------------------------
# Flow Aggregation
# ---------------------------------------------------------------------------
# Inactivity timeout in seconds — flows with no packets for this duration
# are evicted and finalized with termination_reason="TIMEOUT" (spec §1 item 4).
INACTIVITY_TIMEOUT: float = 15.0

# Interval (seconds) between periodic eviction sweeps in the sniffer thread
# (spec §6: "every 1–2 seconds").
EVICTION_INTERVAL: float = 1.0

# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------
# Maximum depth of the thread-safe queue between sniffer and engine threads.
# Provides bounded backpressure; 0 = unlimited (not recommended).
QUEUE_MAX_SIZE: int = 10_000

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
# Path to the SQLite database file.
DB_PATH: str = os.environ.get("CHRONOGUARD_DB_PATH", "chronoguard.db")

# ---------------------------------------------------------------------------
# ONNX Model
# ---------------------------------------------------------------------------
# Path to the pre-trained, pre-compiled static ONNX model artifact.
MODEL_PATH: str = os.environ.get("CHRONOGUARD_MODEL_PATH", "model.onnx")

# ---------------------------------------------------------------------------
# Feature Order (spec §5.3)
# ---------------------------------------------------------------------------
# Fixed, documented feature ordering for ONNX model input.
# Must reference ONLY numeric fields present in FlowRecord (spec §5.1).
# Excludes: flow_id, src_ip, dst_ip (strings), termination_reason (string).
# src_port and dst_port are integers but are identity fields, not model features.
# protocol is an integer and is included as a numeric feature.
FEATURE_ORDER: list[str] = [
    "flow_start_ts",
    "flow_end_ts",
    "flow_duration",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_packet_count",
    "bwd_packet_count",
    "fwd_packet_length_max",
    "fwd_packet_length_mean",
    "bwd_packet_length_max",
    "bwd_packet_length_mean",
    "fwd_packets_per_sec",
    "bwd_packets_per_sec",
    "protocol",
]

# Number of features — must match the ONNX model's input dimension (spec §5.3).
NUM_FEATURES: int = len(FEATURE_ORDER)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Default log level; overridable via environment variable.
LOG_LEVEL: str = os.environ.get("CHRONOGUARD_LOG_LEVEL", "DEBUG")

# Log file path — consumed by the M3 dashboard's log stream (§6.2).
# The FileHandler added in main.py writes structured log lines here so
# ui/data_access.get_log_tail() can tail them without inventing a new
# logging channel.
LOG_FILE_PATH: str = os.environ.get("CHRONOGUARD_LOG_FILE", "chronoguard.log")

# ---------------------------------------------------------------------------
# M2: Drift Detection (ADWIN)
# ---------------------------------------------------------------------------
# ADWIN confidence parameter — lower values make ADWIN more sensitive to
# distributional shifts but increase false-positive risk.  0.002 is River's
# commonly recommended starting point for streaming anomaly workloads.
ADWIN_DELTA: float = float(os.environ.get("CHRONOGUARD_ADWIN_DELTA", "0.002"))

# Structural input features to monitor for drift, in addition to risk_score
# (which is always monitored as the most direct behavioral-drift signal).
# Rationale: flow_duration and fwd_packets_per_sec are the two features most
# susceptible to distributional shift when traffic mix changes (e.g., a shift
# from short HTTP flows to long-lived streaming sessions, or a change in
# packet rate due to a new attack vector).  This set is an ADR candidate —
# see README_M2.md for discussion.
DRIFT_MONITORED_FEATURES: list[str] = [
    "flow_duration",
    "fwd_packets_per_sec",
]

# ---------------------------------------------------------------------------
# M2: Optimizer / Retraining
# ---------------------------------------------------------------------------
# Number of most-recent flow_predictions records to pull for each retrain
# window.  2,000 balances sufficient training signal against bounded
# sweep/fit duration.
RETRAIN_WINDOW_SIZE: int = int(
    os.environ.get("CHRONOGUARD_RETRAIN_WINDOW_SIZE", "2000")
)

# Optuna sweep bounds — the sweep terminates when EITHER ceiling is reached.
OPTUNA_N_TRIALS: int = int(
    os.environ.get("CHRONOGUARD_OPTUNA_N_TRIALS", "3")
)
OPTUNA_TIME_BUDGET_SEC: int = int(
    os.environ.get("CHRONOGUARD_OPTUNA_TIME_BUDGET_SEC", "120")
)

# Accuracy gate: candidate model must achieve at least this accuracy on the
# golden holdout AND at least 0.95 × base_accuracy to pass.
ACCURACY_GATE_THRESHOLD: float = float(
    os.environ.get("CHRONOGUARD_ACCURACY_GATE_THRESHOLD", "0.95")
)

# ---------------------------------------------------------------------------
# M2: Golden Holdout
# ---------------------------------------------------------------------------
# Path to the fixed, versioned, read-only golden holdout CSV.
GOLDEN_HOLDOUT_PATH: str = os.environ.get(
    "CHRONOGUARD_HOLDOUT_PATH", "golden_holdout.csv"
)
