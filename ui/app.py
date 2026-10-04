"""
ChronoGuard v1.0 — Dashboard Entrypoint (M3)

Page layout assembly, polling loop, and session-state management.
Contains no styling constants and no direct SQLite calls.

Launch:  streamlit run ui/app.py [-- --db-path /path/to/chronoguard.db]

Spec reference: M3 §2 (scope), §5.3 (layout), §7 (polling model — st.fragment).

Iteration Map coupling:
    - "Widen/narrow the log panel"  → _LAYOUT_COLUMN_RATIOS below
    - "Add/remove a panel"          → modify layout + import in ui/components/
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ui/ and config imports resolve
# when launched via ``streamlit run ui/app.py`` from any directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ui import copy
from ui.components.feature_chart import render_feature_chart
from ui.components.lineage_table import render_lineage_table
from ui.components.log_stream import render_log_stream
from ui.components.status_bar import render_status_bar
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
from ui.theme import inject_css

# ---------------------------------------------------------------------------
# Layout ratios — the single lever for panel width adjustments.
# Iteration Map: "Widen/narrow the log panel" → change these ratios.
# ---------------------------------------------------------------------------
_LAYOUT_COLUMN_RATIOS: list[int] = [3, 2]

# ---------------------------------------------------------------------------
# Polling interval (seconds) — §7: 500ms via st.fragment.
# ---------------------------------------------------------------------------
_POLL_INTERVAL_SEC: float = 0.5

# ---------------------------------------------------------------------------
# Default database and log paths (from environment or config defaults).
# We avoid importing the full config module to keep the Streamlit process
# lightweight; env-var lookup mirrors config.py's own resolution.
# ---------------------------------------------------------------------------
_DB_PATH = os.environ.get("CHRONOGUARD_DB_PATH", "chronoguard.db")
_LOG_FILE_PATH = os.environ.get("CHRONOGUARD_LOG_FILE", "chronoguard.log")

# ---------------------------------------------------------------------------
# Default feature to chart.
# ---------------------------------------------------------------------------
_DEFAULT_FEATURE = "risk_score"
_AVAILABLE_FEATURES: list[str] = [
    "risk_score",
    "flow_duration",
    "fwd_packets_per_sec",
]


# =========================================================================
# Page Configuration (must be the first Streamlit call)
# =========================================================================
st.set_page_config(
    page_title=copy.PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject theme CSS (all tokens from theme.py, zero hardcoded values).
st.markdown(inject_css(), unsafe_allow_html=True)


# =========================================================================
# Polling Fragments — each zone re-renders independently at 500ms
# =========================================================================

@st.fragment(run_every=_POLL_INTERVAL_SEC)
def _status_bar_fragment() -> None:
    """Poll and render the status bar zone."""
    try:
        conn = get_connection(_DB_PATH)
        try:
            throughput = get_throughput(conn)
            drift = get_drift_status(conn)
            model = get_active_model_version(conn)
        finally:
            conn.close()
    except Exception:
        throughput = 0.0
        drift = {"state": "HEALTHY", "since_ts": 0.0, "triggering_feature": None}
        model = {"model_version": None, "completed_ts": None}

    render_status_bar(throughput, drift, model)


@st.fragment(run_every=_POLL_INTERVAL_SEC)
def _feature_chart_fragment() -> None:
    """Poll and render the feature tracking chart zone."""
    # Feature selector (stored in session state for persistence across reruns).
    selected_feature = st.selectbox(
        copy.FEATURE_SELECTOR_LABEL,
        options=_AVAILABLE_FEATURES,
        index=0,
        key="feature_selector",
        label_visibility="collapsed",
    )

    try:
        conn = get_connection(_DB_PATH)
        try:
            series_df = get_feature_series(conn, selected_feature or _DEFAULT_FEATURE)
            drift_events = get_drift_event_timestamps(conn)
        finally:
            conn.close()
    except Exception:
        import pandas as pd

        series_df = pd.DataFrame(columns=["timestamp", "value"])
        drift_events = []

    render_feature_chart(series_df, drift_events)


@st.fragment(run_every=_POLL_INTERVAL_SEC)
def _log_stream_fragment() -> None:
    """Poll and render the system log stream zone."""
    lines = get_log_tail(_LOG_FILE_PATH)
    render_log_stream(lines)


@st.fragment(run_every=_POLL_INTERVAL_SEC)
def _lineage_table_fragment() -> None:
    """Poll and render the model lineage table zone."""
    try:
        conn = get_connection(_DB_PATH)
        try:
            lineage_df = get_recent_lineage(conn)
        finally:
            conn.close()
    except Exception:
        import pandas as pd

        lineage_df = pd.DataFrame()

    render_lineage_table(lineage_df)


# =========================================================================
# Page Layout — Three-zone grid (§5.3)
# =========================================================================
def main() -> None:
    """Assemble the three-zone layout and attach polling fragments."""
    # Zone 1: Status bar (full width).
    _status_bar_fragment()

    # Zone 2: Feature chart + log stream (side by side).
    chart_col, log_col = st.columns(_LAYOUT_COLUMN_RATIOS)

    with chart_col:
        st.markdown(
            '<div class="cg-panel">',
            unsafe_allow_html=True,
        )
        _feature_chart_fragment()
        st.markdown("</div>", unsafe_allow_html=True)

    with log_col:
        _log_stream_fragment()

    # Zone 3: Lineage table (full width).
    _lineage_table_fragment()


main()
