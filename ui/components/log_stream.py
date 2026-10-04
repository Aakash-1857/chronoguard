"""
ChronoGuard v1.0 — System Log Stream Component (M3)

Live-tailing, monospace-rendered log stream in a fixed-height scrollable
container.  New rows receive a flash-to-rest glow (Pattern C — §5.5).

Pure function of (data) → rendered Streamlit output.
All visual constants from theme.py, all strings from copy.py.

Spec reference: M3 §5.3 (layout), §5.5 (motion Pattern C, layout stability),
§6.3 (function signature).
"""

from __future__ import annotations

import html

import streamlit as st

from ui import copy


def render_log_stream(lines: list[str]) -> None:
    """Render the live-tailing system log stream.

    Parameters
    ----------
    lines : list[str]
        Most recent structured log lines (newest last).
    """
    st.markdown(
        f'<div class="cg-section-title">{copy.LOG_TITLE}</div>',
        unsafe_allow_html=True,
    )

    # Empty state.
    if not lines:
        st.markdown(
            f'<div class="cg-panel cg-scroll-container">'
            f'<div class="cg-empty-state">{copy.NO_LOGS}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    # Build log HTML.
    # Reverse so newest is at the bottom (natural tailing order).
    # The last few lines (newest) get flash-to-rest animation.
    log_html_parts: list[str] = []
    total = len(lines)
    # Only the newest 3 lines get the flash animation (they're
    # the ones most likely to have just appeared).
    flash_threshold = max(0, total - 3)

    for i, line in enumerate(lines):
        escaped = html.escape(line)

        # Determine semantic coloring based on log level in the line.
        extra_class = ""
        flash_class = ""

        if "[ERROR]" in line or "[CRITICAL]" in line:
            extra_class = "cg-log-line-error"
            if i >= flash_threshold:
                flash_class = "cg-flash-critical"
        elif "[WARNING]" in line:
            extra_class = "cg-log-line-warning"
            if i >= flash_threshold:
                flash_class = "cg-flash-warn"
        else:
            if i >= flash_threshold:
                flash_class = "cg-flash-ok"

        log_html_parts.append(
            f'<div class="cg-log-line {extra_class} {flash_class}">'
            f"{escaped}</div>"
        )

    log_html = "\n".join(log_html_parts)

    # Fixed-height container with internal scroll (§5.5 layout stability).
    st.markdown(
        f'<div class="cg-panel cg-scroll-container">{log_html}</div>',
        unsafe_allow_html=True,
    )

    # Auto-scroll to bottom via a small JS snippet.
    st.markdown(
        """
        <script>
        (function() {
            const containers = document.querySelectorAll('.cg-scroll-container');
            containers.forEach(c => { c.scrollTop = c.scrollHeight; });
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )
