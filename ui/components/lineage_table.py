"""
ChronoGuard v1.0 — Model Lineage Table Component (M3)

Renders the retrain history table in a fixed-height scrollable container.
PASS rows styled with status.ok, FAIL rows with status.critical.
New rows receive flash-to-rest glow (Pattern C — §5.5).

Pure function of (data) → rendered Streamlit output.
All visual constants from theme.py, all strings from copy.py.

Spec reference: M3 §5.3 (layout), §5.5 (motion Pattern C, layout stability),
§6.3 (function signature).
"""

from __future__ import annotations

import datetime

import pandas as pd
import streamlit as st

from ui import copy
from ui.theme import COLORS


def render_lineage_table(lineage_df: pd.DataFrame) -> None:
    """Render the model lineage table.

    Parameters
    ----------
    lineage_df : pd.DataFrame
        Columns: ``retrain_id, triggered_ts, triggering_feature, status,
        candidate_accuracy, base_accuracy, model_version, completed_ts``.
        Newest first.
    """
    st.markdown(
        f'<div class="cg-section-title">{copy.LINEAGE_TITLE}</div>',
        unsafe_allow_html=True,
    )

    # Empty state.
    if lineage_df.empty:
        st.markdown(
            f'<div class="cg-panel cg-scroll-container-short">'
            f'<div class="cg-empty-state">{copy.NO_LINEAGE}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    # Header row.
    header_html = (
        '<div class="cg-lineage-row cg-lineage-header">'
        f"<div>{copy.LINEAGE_COL_RETRAIN_ID}</div>"
        f"<div>{copy.LINEAGE_COL_TRIGGERED}</div>"
        f"<div>{copy.LINEAGE_COL_FEATURE}</div>"
        f"<div>{copy.LINEAGE_COL_STATUS}</div>"
        f"<div>{copy.LINEAGE_COL_CANDIDATE_ACC}</div>"
        f"<div>{copy.LINEAGE_COL_BASE_ACC}</div>"
        f"<div>{copy.LINEAGE_COL_MODEL_VER}</div>"
        f"<div>{copy.LINEAGE_COL_COMPLETED}</div>"
        "</div>"
    )

    # Data rows.
    rows_html_parts: list[str] = []
    total = len(lineage_df)
    flash_threshold = max(0, total - 2)  # Newest 2 rows get flash.

    for i, (_, row) in enumerate(lineage_df.iterrows()):
        status = row.get("status", "")

        # Status badge.
        if status == "PASS":
            badge = f'<span class="cg-badge cg-badge-ok">{status}</span>'
            flash_class = "cg-flash-ok" if i >= flash_threshold else ""
        elif status == "FAIL":
            badge = f'<span class="cg-badge cg-badge-critical">{status}</span>'
            flash_class = "cg-flash-critical" if i >= flash_threshold else ""
        else:
            badge = f'<span class="cg-badge cg-badge-signal">{status}</span>'
            flash_class = "cg-flash-warn" if i >= flash_threshold else ""

        # Format timestamps.
        triggered_ts = row.get("triggered_ts", 0)
        completed_ts = row.get("completed_ts", 0)
        triggered_str = _format_ts(triggered_ts)
        completed_str = _format_ts(completed_ts)

        # Format accuracy values.
        cand_acc = row.get("candidate_accuracy", 0)
        base_acc = row.get("base_accuracy", 0)
        cand_str = f"{cand_acc:.4f}" if cand_acc else "—"
        base_str = f"{base_acc:.4f}" if base_acc else "—"

        # Retrain ID (truncated for display).
        retrain_id = str(row.get("retrain_id", ""))
        retrain_display = retrain_id[:8] if len(retrain_id) > 8 else retrain_id

        # Feature and model version.
        feature = row.get("triggering_feature", "—") or "—"
        model_ver_raw = row.get("model_version")
        # pandas stores SQL NULL as float NaN — guard against that.
        if model_ver_raw is None or (isinstance(model_ver_raw, float) and model_ver_raw != model_ver_raw):
            model_ver = "—"
        else:
            model_ver = str(model_ver_raw)
        model_ver_display = model_ver[:12] if len(model_ver) > 12 else model_ver

        rows_html_parts.append(
            f'<div class="cg-lineage-row {flash_class}">'
            f'<div title="{retrain_id}">{retrain_display}</div>'
            f"<div>{triggered_str}</div>"
            f"<div>{feature}</div>"
            f"<div>{badge}</div>"
            f"<div>{cand_str}</div>"
            f"<div>{base_str}</div>"
            f'<div title="{model_ver}">{model_ver_display}</div>'
            f"<div>{completed_str}</div>"
            f"</div>"
        )

    rows_html = "\n".join(rows_html_parts)

    # Fixed-height container with internal scroll (§5.5 layout stability).
    st.markdown(
        f'<div class="cg-panel cg-scroll-container-short">'
        f"{header_html}{rows_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _format_ts(ts: float) -> str:
    """Format an epoch timestamp as a compact UTC string."""
    if not ts:
        return "—"
    try:
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        return dt.strftime("%H:%M:%S")
    except (OSError, ValueError):
        return "—"
