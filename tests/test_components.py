"""
Tests for M3 UI components — ChronoGuard

Uses unittest.mock to patch Streamlit calls. Verifies:
    - Each component handles empty-state data correctly.
    - Components call st.markdown/st.plotly_chart with expected structures.
    - No exceptions on valid data inputs.
"""

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# We need to pre-create a mock 'streamlit' module before importing components,
# because streamlit may not be installed in the test environment or we want to
# avoid its heavy import.  Instead, we import the modules directly and patch
# the 'st' object they use.

@pytest.fixture(autouse=True)
def _ensure_streamlit_mock():
    """Ensure streamlit is importable (even if only as a mock) for component imports."""
    if "streamlit" not in sys.modules:
        mock_st = MagicMock()
        sys.modules["streamlit"] = mock_st
    yield


class TestStatusBar:
    """Test status_bar.render_status_bar."""

    def test_renders_with_data(self):
        import ui.components.status_bar as mod
        mod.st = MagicMock()
        mod.st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

        mod.render_status_bar(
            throughput=42.5,
            drift={
                "state": "HEALTHY",
                "since_ts": 0.0,
                "triggering_feature": None,
            },
            model={
                "model_version": "abc123def456",
                "completed_ts": 1700000000.0,
            },
        )

        assert mod.st.markdown.called
        assert mod.st.columns.called

    def test_renders_drift_detected(self):
        import ui.components.status_bar as mod
        mod.st = MagicMock()
        mod.st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

        mod.render_status_bar(
            throughput=100.0,
            drift={
                "state": "DRIFT_DETECTED",
                "since_ts": 1700000000.0,
                "triggering_feature": "risk_score",
            },
            model={"model_version": None, "completed_ts": None},
        )

        assert mod.st.markdown.called
        # Verify drift badge is warn class.
        all_calls = str(mod.st.markdown.call_args_list)
        assert "cg-badge-warn" in all_calls

    def test_renders_zero_throughput_empty_state(self):
        import ui.components.status_bar as mod
        from ui import copy

        mod.st = MagicMock()
        mod.st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

        mod.render_status_bar(
            throughput=0.0,
            drift={"state": "HEALTHY", "since_ts": 0.0, "triggering_feature": None},
            model={"model_version": None, "completed_ts": None},
        )

        all_calls = str(mod.st.markdown.call_args_list)
        assert copy.NO_MODEL_VERSION in all_calls

    def test_renders_retraining_state(self):
        import ui.components.status_bar as mod
        from ui import copy

        mod.st = MagicMock()
        mod.st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

        mod.render_status_bar(
            throughput=50.0,
            drift={
                "state": "RETRAINING",
                "since_ts": 1700000000.0,
                "triggering_feature": "flow_duration",
            },
            model={"model_version": "v1hash", "completed_ts": 1700000000.0},
        )

        all_calls = str(mod.st.markdown.call_args_list)
        assert "cg-badge-signal" in all_calls


class TestFeatureChart:
    """Test feature_chart.render_feature_chart."""

    def test_renders_empty_state(self):
        import ui.components.feature_chart as mod
        from ui import copy

        mod.st = MagicMock()

        mod.render_feature_chart(
            series_df=pd.DataFrame(columns=["timestamp", "value"]),
            drift_events=[],
        )

        all_calls = str(mod.st.markdown.call_args_list)
        assert copy.NO_FLOWS_YET in all_calls
        # Should NOT call plotly_chart on empty data.
        assert not mod.st.plotly_chart.called

    def test_renders_with_data(self):
        import ui.components.feature_chart as mod
        import time

        mod.st = MagicMock()
        now = time.time()
        df = pd.DataFrame({
            "timestamp": [now - 10 + i for i in range(20)],
            "value": [0.1 * i for i in range(20)],
        })

        mod.render_feature_chart(series_df=df, drift_events=[now - 5])

        assert mod.st.plotly_chart.called

    def test_renders_with_adwin_bounds(self):
        import ui.components.feature_chart as mod
        import time

        mod.st = MagicMock()
        now = time.time()
        df = pd.DataFrame({
            "timestamp": [now - 10 + i for i in range(20)],
            "value": [0.5 for _ in range(20)],
            "adwin_upper": [0.7 for _ in range(20)],
            "adwin_lower": [0.3 for _ in range(20)],
        })

        mod.render_feature_chart(series_df=df, drift_events=[])
        assert mod.st.plotly_chart.called


class TestLogStream:
    """Test log_stream.render_log_stream."""

    def test_renders_empty_state(self):
        import ui.components.log_stream as mod
        from ui import copy

        mod.st = MagicMock()

        mod.render_log_stream(lines=[])

        all_calls = str(mod.st.markdown.call_args_list)
        assert copy.NO_LOGS in all_calls

    def test_renders_with_lines(self):
        import ui.components.log_stream as mod

        mod.st = MagicMock()

        lines = [
            "2026-08-03T10:00:00 [INFO] test: Normal message",
            "2026-08-03T10:00:01 [ERROR] test: Error message",
            "2026-08-03T10:00:02 [WARNING] test: Warning message",
        ]

        mod.render_log_stream(lines=lines)

        all_calls = str(mod.st.markdown.call_args_list)
        assert "cg-scroll-container" in all_calls
        assert "cg-log-line" in all_calls

    def test_escapes_html_in_lines(self):
        import ui.components.log_stream as mod

        mod.st = MagicMock()

        lines = ["<script>alert('xss')</script>"]
        mod.render_log_stream(lines=lines)

        # Check the log-content call (second markdown call), not the
        # auto-scroll script call (which is legitimate internal code).
        # The log content is in the call that contains 'cg-scroll-container'.
        log_content_calls = [
            str(call) for call in mod.st.markdown.call_args_list
            if "cg-log-line" in str(call)
        ]
        assert len(log_content_calls) > 0
        log_html = log_content_calls[0]
        # User-provided <script> must be escaped.
        assert "&lt;script&gt;" in log_html

    def test_semantic_coloring(self):
        import ui.components.log_stream as mod

        mod.st = MagicMock()

        lines = [
            "2026-08-03T10:00:01 [ERROR] test: Error message",
            "2026-08-03T10:00:02 [WARNING] test: Warning message",
        ]

        mod.render_log_stream(lines=lines)

        all_calls = str(mod.st.markdown.call_args_list)
        assert "cg-log-line-error" in all_calls
        assert "cg-log-line-warning" in all_calls


class TestLineageTable:
    """Test lineage_table.render_lineage_table."""

    def test_renders_empty_state(self):
        import ui.components.lineage_table as mod
        from ui import copy

        mod.st = MagicMock()

        mod.render_lineage_table(lineage_df=pd.DataFrame())

        all_calls = str(mod.st.markdown.call_args_list)
        assert copy.NO_LINEAGE in all_calls

    def test_renders_with_data(self):
        import ui.components.lineage_table as mod
        import time

        mod.st = MagicMock()
        now = time.time()
        df = pd.DataFrame([{
            "retrain_id": "retrain_001",
            "triggered_ts": now - 30,
            "triggering_feature": "risk_score",
            "status": "PASS",
            "candidate_accuracy": 0.97,
            "base_accuracy": 0.95,
            "model_version": "new_model_v1",
            "completed_ts": now - 25,
        }])

        mod.render_lineage_table(lineage_df=df)

        all_calls = str(mod.st.markdown.call_args_list)
        assert "cg-lineage-row" in all_calls
        assert "cg-badge-ok" in all_calls

    def test_renders_fail_row(self):
        import ui.components.lineage_table as mod
        import time

        mod.st = MagicMock()
        now = time.time()
        df = pd.DataFrame([{
            "retrain_id": "retrain_002",
            "triggered_ts": now - 30,
            "triggering_feature": "flow_duration",
            "status": "FAIL",
            "candidate_accuracy": 0.80,
            "base_accuracy": 0.95,
            "model_version": None,
            "completed_ts": now - 25,
        }])

        mod.render_lineage_table(lineage_df=df)

        all_calls = str(mod.st.markdown.call_args_list)
        assert "cg-badge-critical" in all_calls

    def test_renders_multiple_rows(self):
        import ui.components.lineage_table as mod
        import time

        mod.st = MagicMock()
        now = time.time()
        df = pd.DataFrame([
            {
                "retrain_id": "retrain_001", "triggered_ts": now - 60,
                "triggering_feature": "risk_score", "status": "PASS",
                "candidate_accuracy": 0.97, "base_accuracy": 0.95,
                "model_version": "v1", "completed_ts": now - 55,
            },
            {
                "retrain_id": "retrain_002", "triggered_ts": now - 30,
                "triggering_feature": "flow_duration", "status": "FAIL",
                "candidate_accuracy": 0.80, "base_accuracy": 0.95,
                "model_version": None, "completed_ts": now - 25,
            },
        ])

        mod.render_lineage_table(lineage_df=df)

        all_calls = str(mod.st.markdown.call_args_list)
        assert "cg-badge-ok" in all_calls
        assert "cg-badge-critical" in all_calls
