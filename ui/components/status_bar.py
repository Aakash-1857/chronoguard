"""
ChronoGuard v1.0 — Status Bar Component (M3)

Renders the top-bar zone: Throughput, Drift Status, Active Model Version,
and the Live Beacon (Pattern A — §5.5).

Pure function of (data) → rendered Streamlit output.
All visual constants from theme.py, all strings from copy.py.

Spec reference: M3 §5.3 (layout), §5.4 (signature element), §5.5 (motion),
§6.3 (function signature).
"""

from __future__ import annotations

import datetime

import streamlit as st

from ui import copy
from ui.theme import COLORS


def render_status_bar(
    throughput: float,
    drift: dict,
    model: dict,
) -> None:
    """Render the status bar with three metric cells.

    Parameters
    ----------
    throughput : float
        Current flows/sec value.
    drift : dict
        ``{'state': str, 'since_ts': float, 'triggering_feature': str|None}``
    model : dict
        ``{'model_version': str|None, 'completed_ts': float|None}``
    """
    # Determine drift badge class and label.
    drift_state = drift.get("state", "HEALTHY")
    if drift_state == "HEALTHY":
        badge_class = "cg-badge cg-badge-ok"
        drift_label = copy.HEALTHY
    elif drift_state == "DRIFT_DETECTED":
        badge_class = "cg-badge cg-badge-warn"
        drift_label = copy.DRIFT_DETECTED
    elif drift_state == "RETRAINING":
        badge_class = "cg-badge cg-badge-signal"
        drift_label = copy.RETRAINING
    else:
        badge_class = "cg-badge cg-badge-ok"
        drift_label = copy.HEALTHY

    # Drift detail (triggering feature).
    triggering = drift.get("triggering_feature")
    drift_detail = f" &middot; {triggering}" if triggering else ""

    # Model version display.
    model_version = model.get("model_version")
    if model_version:
        model_display = model_version
    else:
        model_display = copy.NO_MODEL_VERSION

    # Format completed_ts if available.
    completed_ts = model.get("completed_ts")
    if completed_ts:
        model_ts = datetime.datetime.fromtimestamp(
            completed_ts, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        model_ts = ""

    # --- Render ---
    st.markdown(
        '<div class="cg-panel" style="margin-bottom: var(--sp-md);">',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        # Throughput metric with live beacon.
        st.markdown(
            f"""
            <div class="cg-metric">
                <div class="cg-metric-label">
                    <span class="cg-live-beacon"></span>
                    {copy.LIVE_INDICATOR} &middot; {copy.THROUGHPUT_LABEL}
                </div>
                <div class="cg-metric-value">{throughput:.1f}</div>
                <div class="cg-caption">{copy.THROUGHPUT_UNIT}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Drift status badge.
        st.markdown(
            f"""
            <div class="cg-metric">
                <div class="cg-metric-label">{copy.DRIFT_LABEL}</div>
                <div style="margin-top: var(--sp-xs);">
                    <span class="{badge_class}">{drift_label}</span>
                    <span class="cg-caption">{drift_detail}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        # Active model version.
        st.markdown(
            f"""
            <div class="cg-metric">
                <div class="cg-metric-label">{copy.MODEL_VERSION_LABEL}</div>
                <div class="cg-mono" style="margin-top: var(--sp-xs);">
                    {model_display}
                </div>
                <div class="cg-caption">{model_ts}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
