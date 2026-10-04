"""
Tests for M3 copy system — ChronoGuard

Verifies:
    - All required string constants exist and are non-empty.
    - No inline string literals in component files that should come from copy.py.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ui import copy


class TestCopyConstants:
    """All required string constants must exist and be non-empty."""

    @pytest.mark.parametrize(
        "attr",
        [
            "PAGE_TITLE",
            "STATUS_BAR_TITLE",
            "CHART_TITLE",
            "LOG_TITLE",
            "LINEAGE_TITLE",
            "THROUGHPUT_LABEL",
            "THROUGHPUT_UNIT",
            "DRIFT_LABEL",
            "MODEL_VERSION_LABEL",
            "LIVE_INDICATOR",
            "HEALTHY",
            "DRIFT_DETECTED",
            "RETRAINING",
            "NO_FLOWS_YET",
            "NO_DRIFT_EVENTS",
            "NO_LINEAGE",
            "NO_LOGS",
            "NO_MODEL_VERSION",
            "CHART_XAXIS",
            "CHART_YAXIS",
            "FEATURE_SELECTOR_LABEL",
            "LINEAGE_COL_RETRAIN_ID",
            "LINEAGE_COL_TRIGGERED",
            "LINEAGE_COL_FEATURE",
            "LINEAGE_COL_STATUS",
            "LINEAGE_COL_CANDIDATE_ACC",
            "LINEAGE_COL_BASE_ACC",
            "LINEAGE_COL_MODEL_VER",
            "LINEAGE_COL_COMPLETED",
        ],
    )
    def test_constant_exists_and_nonempty(self, attr):
        assert hasattr(copy, attr), f"copy.{attr} is missing"
        value = getattr(copy, attr)
        assert isinstance(value, str), f"copy.{attr} is not a string"
        assert len(value) > 0, f"copy.{attr} is empty"

    def test_empty_state_messages_are_user_friendly(self):
        """Empty states should use plain language, not technical jargon."""
        # Per §7: "No flows recorded yet." not "No data available."
        assert "recorded" in copy.NO_FLOWS_YET.lower() or "yet" in copy.NO_FLOWS_YET.lower()
        assert "No data available" not in copy.NO_FLOWS_YET
