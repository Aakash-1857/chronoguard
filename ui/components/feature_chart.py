"""
ChronoGuard v1.0 — Feature Tracking Chart Component (M3)

The signature element (§5.4): live metric line in status.signal, ADWIN
bounds as a dashed corridor with 40% opacity fill, drift-fire vertical
rules in status.warn with monospace timestamp labels.

Pure function of (data) → rendered Streamlit output.
All colors from theme.py, all strings from copy.py.

Spec reference: M3 §5.4 (signature element), §5.5 Pattern B (chart transitions),
§6.3 (function signature), §7 (chart rendering — Plotly).
"""

from __future__ import annotations

import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import copy
from ui.theme import COLORS, MOTION, TYPOGRAPHY, hex_to_rgba


def render_feature_chart(
    series_df: pd.DataFrame,
    drift_events: list[float],
) -> None:
    """Render the feature tracking chart with ADWIN bounds and drift markers.

    Parameters
    ----------
    series_df : pd.DataFrame
        Columns: ``['timestamp', 'value']``.  May also include
        ``'adwin_upper'`` and ``'adwin_lower'`` if ADWIN bounds are available.
    drift_events : list[float]
        Epoch timestamps of drift-fire events (vertical rules).
    """
    st.markdown(
        f'<div class="cg-section-title">{copy.CHART_TITLE}</div>',
        unsafe_allow_html=True,
    )

    # Empty state.
    if series_df.empty:
        st.markdown(
            f'<div class="cg-empty-state">{copy.NO_FLOWS_YET}</div>',
            unsafe_allow_html=True,
        )
        return

    fig = go.Figure()

    # Convert epoch timestamps to datetime for readable x-axis.
    timestamps = pd.to_datetime(series_df["timestamp"], unit="s", utc=True)
    values = series_df["value"]

    # --- Main metric line (status.signal) ---
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=values,
            mode="lines",
            name=copy.CHART_YAXIS,
            line=dict(
                color=COLORS["status"]["signal"],
                width=2,
            ),
            hovertemplate="%{y:.4f}<extra></extra>",
        )
    )

    # --- ADWIN bounds corridor (dashed, text.muted, 40% opacity fill) ---
    if "adwin_upper" in series_df.columns and "adwin_lower" in series_df.columns:
        upper = series_df["adwin_upper"]
        lower = series_df["adwin_lower"]

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=upper,
                mode="lines",
                name=copy.CHART_ADWIN_UPPER,
                line=dict(
                    color=COLORS["text"]["muted"],
                    width=1,
                    dash="dash",
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=lower,
                mode="lines",
                name=copy.CHART_ADWIN_LOWER,
                line=dict(
                    color=COLORS["text"]["muted"],
                    width=1,
                    dash="dash",
                ),
                fill="tonexty",
                fillcolor=hex_to_rgba(COLORS["text"]["muted"], 0.4),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # --- Drift-fire vertical rules (status.warn + monospace timestamp) ---
    for ts in drift_events:
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        label = dt.strftime("%H:%M:%S")

        fig.add_vline(
            x=dt,
            line=dict(
                color=COLORS["status"]["warn"],
                width=2,
            ),
            annotation=dict(
                text=label,
                font=dict(
                    family=TYPOGRAPHY["data"]["family"],
                    size=11,
                    color=COLORS["status"]["warn"],
                ),
                yanchor="bottom",
                y=0,
                showarrow=False,
            ),
        )

    # --- Layout styling (all tokens from theme.py) ---
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family=TYPOGRAPHY["body"]["family"],
            color=COLORS["text"]["muted"],
            size=13,
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            zeroline=False,
            tickfont=dict(
                family=TYPOGRAPHY["data"]["family"],
                color=COLORS["text"]["muted"],
                size=11,
            ),
        ),
        yaxis=dict(
            title=None,
            showgrid=True,
            gridcolor=COLORS["surface"]["border"],
            gridwidth=1,
            zeroline=False,
            tickfont=dict(
                family=TYPOGRAPHY["data"]["family"],
                color=COLORS["text"]["muted"],
                size=11,
            ),
        ),
        margin=dict(l=40, r=20, t=10, b=30),
        height=340,
        showlegend=False,
        # Pattern B: chart transitions (§5.5).
        transition=dict(
            duration=int(MOTION["chart_transition_ms"]),
            easing="cubic-in-out",
        ),
    )

    st.plotly_chart(fig, use_container_width=True, key="feature_chart")
