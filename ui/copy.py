"""
ChronoGuard v1.0 — Centralized UI Copy (M3)

All user-facing strings (labels, empty states, status messages) are
defined here so wording changes never require touching component logic.

Spec reference: M3 §2 (ui/copy.py scope), §7 (empty/loading states).

Iteration Map: "Change wording of any label/message" → this file only.
"""

# ---------------------------------------------------------------------------
# Page / Section Titles
# ---------------------------------------------------------------------------
PAGE_TITLE: str = "ChronoGuard Control Panel"
STATUS_BAR_TITLE: str = "System Status"
CHART_TITLE: str = "Feature Tracking"
LOG_TITLE: str = "System Log"
LINEAGE_TITLE: str = "Model Lineage"

# ---------------------------------------------------------------------------
# Status Bar Labels
# ---------------------------------------------------------------------------
THROUGHPUT_LABEL: str = "Throughput"
THROUGHPUT_UNIT: str = "flows/sec"
DRIFT_LABEL: str = "Drift Status"
MODEL_VERSION_LABEL: str = "Active Model"
LIVE_INDICATOR: str = "LIVE"

# ---------------------------------------------------------------------------
# Drift State Labels
# ---------------------------------------------------------------------------
HEALTHY: str = "Healthy"
DRIFT_DETECTED: str = "Drift Detected"
RETRAINING: str = "Retraining\u2026"

# ---------------------------------------------------------------------------
# Lineage Table Headers
# ---------------------------------------------------------------------------
LINEAGE_COL_RETRAIN_ID: str = "Retrain ID"
LINEAGE_COL_TRIGGERED: str = "Triggered"
LINEAGE_COL_FEATURE: str = "Feature"
LINEAGE_COL_STATUS: str = "Status"
LINEAGE_COL_CANDIDATE_ACC: str = "Cand. Acc"
LINEAGE_COL_BASE_ACC: str = "Base Acc"
LINEAGE_COL_MODEL_VER: str = "Model Ver"
LINEAGE_COL_COMPLETED: str = "Completed"

# ---------------------------------------------------------------------------
# Empty States — plain-language, interface-voiced (§7)
# ---------------------------------------------------------------------------
NO_FLOWS_YET: str = "No flows recorded yet."
NO_DRIFT_EVENTS: str = "No drift events observed."
NO_LINEAGE: str = "No retrain attempts recorded."
NO_LOGS: str = "No log entries yet."
NO_MODEL_VERSION: str = "Baseline (no retrains)"

# ---------------------------------------------------------------------------
# Chart Labels
# ---------------------------------------------------------------------------
CHART_XAXIS: str = "Time"
CHART_YAXIS: str = "Value"
CHART_ADWIN_UPPER: str = "ADWIN Upper"
CHART_ADWIN_LOWER: str = "ADWIN Lower"
CHART_DRIFT_EVENT: str = "Drift Event"
FEATURE_SELECTOR_LABEL: str = "Monitored Feature"
