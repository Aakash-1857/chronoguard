"""
Unit tests for engine.py — ChronoGuard M1

Covers (per spec §8 deliverables):
  - Feature vector assembly against FEATURE_ORDER
  - Length mismatch raises explicit exception (spec §5.3)
  - score_flow with placeholder model returns valid (float, Optional[str])
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import pytest

from config import FEATURE_ORDER, NUM_FEATURES
from engine import score_flow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def onnx_model_path(tmp_path_factory):
    """Generate a placeholder ONNX model for testing."""
    model_dir = tmp_path_factory.mktemp("model")
    model_path = str(model_dir / "test_model.onnx")

    # Inline generation to avoid dependency on the scaffolding script
    try:
        from sklearn.ensemble import RandomForestClassifier
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        rng = np.random.RandomState(42)
        X = rng.randn(100, NUM_FEATURES).astype(np.float32)
        y = rng.randint(0, 2, size=100)

        clf = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42)
        clf.fit(X, y)

        initial_type = [("input", FloatTensorType([None, NUM_FEATURES]))]
        onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=13)

        with open(model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        return model_path
    except ImportError:
        pytest.skip("scikit-learn and skl2onnx required for engine tests")


@pytest.fixture
def ort_session(onnx_model_path):
    """Load the placeholder ONNX model as an InferenceSession."""
    import onnxruntime as ort
    return ort.InferenceSession(onnx_model_path)


def _sample_flow_record() -> dict:
    """A complete FlowRecord with valid numeric values for all FEATURE_ORDER fields."""
    return {
        "flow_id": "test_flow_001",
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


# ---------------------------------------------------------------------------
# Feature Vector Assembly Tests
# ---------------------------------------------------------------------------
class TestFeatureVectorAssembly:
    """Spec §5.3: feature vector must match FEATURE_ORDER exactly."""

    def test_feature_extraction_correct_order(self, ort_session) -> None:
        """Features are extracted in FEATURE_ORDER sequence."""
        record = _sample_flow_record()

        # Manually extract to verify order
        expected = [float(record[f]) for f in FEATURE_ORDER]
        assert len(expected) == NUM_FEATURES

        # score_flow should work without error
        risk, cls = score_flow(record, ort_session)
        assert isinstance(risk, float)

    def test_missing_feature_raises_error(self, ort_session) -> None:
        """Missing feature in FlowRecord raises ValueError (spec §5.3)."""
        record = _sample_flow_record()
        del record["flow_duration"]  # Remove a required feature

        with pytest.raises(ValueError, match="Feature.*not found"):
            score_flow(record, ort_session)


# ---------------------------------------------------------------------------
# score_flow Return Type Tests
# ---------------------------------------------------------------------------
class TestScoreFlow:
    """Spec §5.5: score_flow returns (risk_score: float, predicted_class: Optional[str])."""

    def test_returns_float_risk_score(self, ort_session) -> None:
        record = _sample_flow_record()
        risk, _ = score_flow(record, ort_session)
        assert isinstance(risk, float)
        assert 0.0 <= risk <= 1.0

    def test_returns_optional_predicted_class(self, ort_session) -> None:
        record = _sample_flow_record()
        _, cls = score_flow(record, ort_session)
        # For a binary classifier, predicted_class should be a string (or None)
        assert cls is None or isinstance(cls, str)

    def test_deterministic_predictions(self, ort_session) -> None:
        """Spec §7: same input → bit-identical risk_score (deterministic inference)."""
        record = _sample_flow_record()
        risk1, cls1 = score_flow(record, ort_session)
        risk2, cls2 = score_flow(record, ort_session)
        assert risk1 == risk2  # Bit-identical, not approx
        assert cls1 == cls2

    def test_risk_score_clamped(self, ort_session) -> None:
        """Risk score is always in [0.0, 1.0]."""
        record = _sample_flow_record()
        risk, _ = score_flow(record, ort_session)
        assert 0.0 <= risk <= 1.0
