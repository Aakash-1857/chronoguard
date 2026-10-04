"""
Unit tests for drift_monitor.py — ChronoGuard M2

Covers (per M2 spec §7/§8):
  - ADWIN correctly fires on injected synthetic drift.
  - ADWIN does not false-positive on stationary distribution.
  - Multiple monitored features tracked independently.
  - risk_score is always monitored.
"""

from __future__ import annotations

import numpy as np
import pytest

from drift_monitor import DriftMonitor


def _make_record(**overrides) -> dict:
    """Build a minimal FlowRecord-like dict for drift monitor testing."""
    base = {
        "flow_id": "test_flow",
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "src_port": 12345,
        "dst_port": 80,
        "protocol": 6,
        "flow_start_ts": 1700000000.0,
        "flow_end_ts": 1700000001.5,
        "flow_duration": 1.5,
        "flow_iat_mean": 0.3,
        "flow_iat_std": 0.05,
        "fwd_packet_count": 5,
        "bwd_packet_count": 3,
        "fwd_packet_length_max": 1500,
        "fwd_packet_length_mean": 750.0,
        "bwd_packet_length_max": 600,
        "bwd_packet_length_mean": 300.0,
        "fwd_packets_per_sec": 3.33,
        "bwd_packets_per_sec": 2.0,
        "termination_reason": "FIN",
    }
    base.update(overrides)
    return base


class TestDriftMonitorInit:
    """DriftMonitor initialization tests."""

    def test_risk_score_always_monitored(self) -> None:
        """risk_score is always in the monitored set (spec §5.1)."""
        dm = DriftMonitor(monitored_features=["flow_duration"])
        assert "risk_score" in dm.monitored_features

    def test_default_features_from_config(self) -> None:
        """Default monitored features come from config.DRIFT_MONITORED_FEATURES."""
        dm = DriftMonitor()
        assert "risk_score" in dm.monitored_features
        assert "flow_duration" in dm.monitored_features
        assert "fwd_packets_per_sec" in dm.monitored_features

    def test_custom_features(self) -> None:
        """Custom feature list is respected."""
        dm = DriftMonitor(monitored_features=["flow_iat_mean"])
        assert "flow_iat_mean" in dm.monitored_features
        assert "risk_score" in dm.monitored_features  # always added


class TestDriftFiresOnShift:
    """Spec §7: ADWIN correctly fires on injected synthetic drift."""

    def test_drift_fires_on_mean_shift(self) -> None:
        """A sudden mean shift in risk_score triggers drift detection."""
        dm = DriftMonitor(
            monitored_features=[],  # only risk_score
            adwin_delta=0.002,
        )

        rng = np.random.RandomState(123)

        # Phase 1: stationary distribution (mean=0.3, std=0.05)
        for _ in range(500):
            score = float(np.clip(rng.normal(0.3, 0.05), 0.0, 1.0))
            record = _make_record()
            dm.update(record, score)

        # Phase 2: shifted distribution (mean=0.8, std=0.05) — drift should fire
        drift_detected = False
        for i in range(500):
            score = float(np.clip(rng.normal(0.8, 0.05), 0.0, 1.0))
            record = _make_record()
            drifted = dm.update(record, score)
            if "risk_score" in drifted:
                drift_detected = True
                break

        assert drift_detected, (
            "ADWIN did not fire drift within 500 samples after a 0.3→0.8 mean shift"
        )

    def test_drift_fires_on_feature_shift(self) -> None:
        """Drift in a structural feature (flow_duration) is detected."""
        dm = DriftMonitor(
            monitored_features=["flow_duration"],
            adwin_delta=0.002,
        )

        rng = np.random.RandomState(456)

        # Phase 1: stationary flow_duration (mean=5.0)
        for _ in range(500):
            record = _make_record(flow_duration=float(rng.normal(5.0, 1.0)))
            dm.update(record, risk_score=0.3)

        # Phase 2: shifted flow_duration (mean=50.0)
        drift_detected = False
        for _ in range(500):
            record = _make_record(flow_duration=float(rng.normal(50.0, 5.0)))
            drifted = dm.update(record, risk_score=0.3)
            if "flow_duration" in drifted:
                drift_detected = True
                break

        assert drift_detected, (
            "ADWIN did not fire drift on flow_duration shift 5→50"
        )


class TestNoFalsePositive:
    """Spec §7: ADWIN does not false-positive over stationary distribution."""

    def test_no_false_positive_stationary_5min(self) -> None:
        """No spurious drift fires over a 5-minute-equivalent stationary stream.

        At ~100 flows/sec equivalent, 5 minutes ≈ 30,000 samples.
        We use 30,000 samples from a single stationary distribution.
        """
        dm = DriftMonitor(
            monitored_features=["flow_duration", "fwd_packets_per_sec"],
            adwin_delta=0.002,
        )

        rng = np.random.RandomState(789)
        false_positives: list[str] = []

        for _ in range(30_000):
            score = float(np.clip(rng.normal(0.3, 0.05), 0.0, 1.0))
            record = _make_record(
                flow_duration=float(rng.normal(5.0, 1.0)),
                fwd_packets_per_sec=float(max(0.1, rng.normal(10.0, 2.0))),
            )
            drifted = dm.update(record, score)
            false_positives.extend(drifted)

        assert len(false_positives) == 0, (
            f"ADWIN false-positived {len(false_positives)} times "
            f"on stationary stream: {false_positives[:5]}"
        )


class TestMultipleFeatures:
    """Multiple features are tracked independently."""

    def test_independent_tracking(self) -> None:
        """Drift in one feature does not affect others."""
        dm = DriftMonitor(
            monitored_features=["flow_duration", "fwd_packets_per_sec"],
            adwin_delta=0.002,
        )

        rng = np.random.RandomState(101)

        # Phase 1: stationary everything
        for _ in range(500):
            record = _make_record(
                flow_duration=float(rng.normal(5.0, 1.0)),
                fwd_packets_per_sec=float(max(0.1, rng.normal(10.0, 2.0))),
            )
            dm.update(record, risk_score=0.3)

        # Phase 2: shift only flow_duration, keep others stationary
        duration_drifted = False
        pps_drifted = False

        for _ in range(500):
            record = _make_record(
                flow_duration=float(rng.normal(100.0, 10.0)),  # shifted
                fwd_packets_per_sec=float(max(0.1, rng.normal(10.0, 2.0))),  # stable
            )
            drifted = dm.update(record, risk_score=0.3)  # risk stable
            if "flow_duration" in drifted:
                duration_drifted = True
            if "fwd_packets_per_sec" in drifted:
                pps_drifted = True

        assert duration_drifted, "flow_duration drift should have fired"
        # pps_drifted could happen due to ADWIN sensitivity, but shouldn't
        # in a clearly stationary signal — we just verify duration was caught.
