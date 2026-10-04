"""
ChronoGuard v1.0 — Drift Monitor (M2)

Streaming drift evaluator using River's ADWIN algorithm.  Attached to
the Engine thread's inference path — one ADWIN instance per monitored
feature/score.

Spec references: M2 §5.1 (DriftMonitor interface), §6 (pipeline ordering).

Monitored feature rationale (documented per §5.1 directive):
    - risk_score: the model's own output — the most direct signal of
      behavioral drift.  If the risk-score distribution shifts, the model's
      decision surface is encountering data it was not trained on.
    - flow_duration: structural input feature prone to shift when the
      traffic mix changes (e.g., short HTTP request flows → long-lived
      streaming/tunneling sessions).
    - fwd_packets_per_sec: structural input feature that shifts under
      volumetric attacks or protocol-mix changes (e.g., bursty scanning
      vs. steady application traffic).

    This set is explicitly configurable via config.DRIFT_MONITORED_FEATURES
    and is an ADR candidate for future review.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from river.drift import ADWIN

from config import ADWIN_DELTA, DRIFT_MONITORED_FEATURES

if TYPE_CHECKING:
    from flow_state import FlowRecord

logger = logging.getLogger(__name__)


class DriftMonitor:
    """ADWIN-based streaming drift detector.

    Maintains one ADWIN instance per monitored feature/score.
    The Engine thread calls update() synchronously after each inference —
    ADWIN.update() is O(1)-amortized per call, so this does not add
    meaningful latency to the inference path.

    Spec §5.1 contract:
        __init__(monitored_features, adwin_delta)
        update(record, risk_score) -> list[str]
    """

    def __init__(
        self,
        monitored_features: list[str] | None = None,
        adwin_delta: float = ADWIN_DELTA,
    ) -> None:
        """Initialize one ADWIN instance per monitored feature/score.

        Parameters
        ----------
        monitored_features : list[str] | None
            Feature names from FlowRecord to monitor.  Defaults to
            config.DRIFT_MONITORED_FEATURES.  ``risk_score`` is always
            added implicitly (it is the most direct behavioral-drift signal).
        adwin_delta : float
            River ADWIN confidence parameter.  Sourced from config.ADWIN_DELTA
            at runtime (the default parameter here is documentation only).
        """
        if monitored_features is None:
            monitored_features = list(DRIFT_MONITORED_FEATURES)

        # Ensure risk_score is always monitored (spec §5.1 requirement)
        all_features = list(monitored_features)
        if "risk_score" not in all_features:
            all_features.insert(0, "risk_score")

        self._feature_names: list[str] = all_features
        self._adwin_delta = adwin_delta

        # One ADWIN instance per monitored feature
        self._detectors: dict[str, ADWIN] = {
            name: ADWIN(delta=adwin_delta) for name in self._feature_names
        }

        logger.info(
            "DriftMonitor initialized: features=%s, adwin_delta=%.4f",
            self._feature_names,
            adwin_delta,
        )

    @property
    def monitored_features(self) -> list[str]:
        """Return the list of monitored feature names."""
        return list(self._feature_names)

    def update(self, record: "FlowRecord", risk_score: float) -> list[str]:
        """Feed each monitored value into its ADWIN instance.

        Returns a list of feature names for which drift fired on this update
        (empty list if none).

        Parameters
        ----------
        record : FlowRecord
            The finalized flow record (dict) from the engine.
        risk_score : float
            The model's risk score for this flow.

        Returns
        -------
        list[str]
            Feature names where ADWIN detected a drift event.
        """
        drifted: list[str] = []

        for feat_name in self._feature_names:
            # Get the value to feed into ADWIN
            if feat_name == "risk_score":
                value = risk_score
            else:
                value = record.get(feat_name)
                if value is None:
                    # Feature not present in record — skip silently.
                    # This should not happen with correctly configured
                    # DRIFT_MONITORED_FEATURES but is not a crash-worthy error.
                    logger.debug(
                        "Monitored feature '%s' not found in FlowRecord, skipping",
                        feat_name,
                    )
                    continue
                value = float(value)

            detector = self._detectors[feat_name]
            # ADWIN.update returns the detector itself; drift is detected
            # by checking the .drift_detected attribute after update.
            detector.update(value)
            if detector.drift_detected:
                drifted.append(feat_name)

        return drifted
