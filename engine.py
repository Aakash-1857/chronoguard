"""
ChronoGuard v1.0 — Engine Thread (M1 + M2)

Consumes finalized FlowRecords from the thread-safe queue, scores each
flow via a pre-trained static ONNX model, and persists the flow record
plus prediction to the SQLite database.

M2 additions:
    - Resolves the active ONNX model via ModelRegistry on each inference call
      (hot-swap aware — reloads session when model path changes).
    - Feeds drift_monitor.update() after each inference.
    - Signals the optimizer thread on drift detection (non-blocking).

Spec references: M1 §5.3, §5.5, §6; M2 §4, §5.1, §5.2, §6.
"""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
import onnxruntime as ort

import db
from config import FEATURE_ORDER, MODEL_PATH, NUM_FEATURES
from flow_state import FlowRecord

if TYPE_CHECKING:
    from drift_monitor import DriftMonitor
    from model_registry import ModelRegistry

logger = logging.getLogger(__name__)


def _compute_model_version(model_path: str) -> str:
    """Derive a model version string from the file's SHA-256 hash (first 12 chars)."""
    try:
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except OSError:
        return "unknown"


def score_flow(
    record: FlowRecord,
    session: ort.InferenceSession,
) -> tuple[float, Optional[str]]:
    """Score a single flow record using the ONNX model.

    Returns (risk_score, predicted_class).

    Spec §5.3:
    - Input: 2D float32 tensor of shape (1, N) where N = len(FEATURE_ORDER).
    - Output: scalar risk score (float, 0.0–1.0) and optional predicted class.
    - Any mismatch between FlowRecord numeric fields and FEATURE_ORDER length
      at runtime must raise an explicit exception (not silently truncate/pad).

    Signature matches spec §5.5.
    """
    # Assemble feature vector in FEATURE_ORDER
    features: list[float] = []
    for feat_name in FEATURE_ORDER:
        if feat_name not in record:
            raise ValueError(
                f"Feature '{feat_name}' from FEATURE_ORDER not found in FlowRecord. "
                f"Available keys: {list(record.keys())}"
            )
        features.append(float(record[feat_name]))

    # Validate dimension match (spec §5.3)
    if len(features) != NUM_FEATURES:
        raise ValueError(
            f"Feature vector length ({len(features)}) does not match "
            f"NUM_FEATURES ({NUM_FEATURES}). This indicates a FEATURE_ORDER "
            f"configuration error."
        )

    # Build input tensor: shape (1, N), dtype float32
    input_array = np.array([features], dtype=np.float32)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_array})

    # Extract risk score and optional class label from model output.
    # Convention for ONNX classifiers:
    #   outputs[0] = predicted labels (shape [1])
    #   outputs[1] = probability map (list of dicts or 2D array)
    # For regression or simple models:
    #   outputs[0] = scores/probabilities
    predicted_class: Optional[str] = None
    risk_score: float

    if len(outputs) >= 2:
        # Classifier with label + probability outputs
        raw_label = outputs[0]
        if hasattr(raw_label, '__len__') and len(raw_label) > 0:
            predicted_class = str(raw_label[0])

        probas = outputs[1]
        if isinstance(probas, list) and len(probas) > 0:
            # skl2onnx outputs list of dicts [{class: prob, ...}]
            prob_dict = probas[0]
            if isinstance(prob_dict, dict):
                # Risk = max probability across non-benign classes,
                # or probability of positive class for binary.
                # For a simple binary placeholder, just take max prob.
                risk_score = float(max(prob_dict.values()))
            else:
                risk_score = float(prob_dict)
        elif isinstance(probas, np.ndarray):
            # 2D array of shape (1, num_classes)
            if probas.ndim == 2:
                risk_score = float(np.max(probas[0]))
            else:
                risk_score = float(np.max(probas))
        else:
            risk_score = 0.0
    else:
        # Single output — treat as risk score directly
        raw = outputs[0]
        if isinstance(raw, np.ndarray):
            risk_score = float(raw.flat[0])
        else:
            risk_score = float(raw)

    # Clamp to [0.0, 1.0]
    risk_score = max(0.0, min(1.0, risk_score))

    return risk_score, predicted_class


def run_engine(
    in_queue: queue.Queue[FlowRecord],
    stop_event: threading.Event,
    model_path: str,
    db_path: str,
    # --- M2 optional parameters (backward-compatible with M1 callers) ---
    registry: Optional["ModelRegistry"] = None,
    drift_monitor: Optional["DriftMonitor"] = None,
    drift_signal: Optional[threading.Event] = None,
    trigger_meta_queue: Optional[queue.Queue] = None,
) -> None:
    """Engine loop — runs in a dedicated thread.

    Blocks on the input queue, scores each flow via ONNX, and persists
    results to SQLite.

    M2 extensions (all optional for backward compatibility):
        - registry: if provided, the engine resolves the active model path
          via registry.current_model_path() on each inference call and
          reloads the ONNX session when it changes.
        - drift_monitor: if provided, update() is called after each inference.
        - drift_signal / trigger_meta_queue: if provided, used to notify the
          optimizer thread when drift is detected.

    Parameters
    ----------
    in_queue : queue.Queue[FlowRecord]
        Thread-safe queue to consume finalized flow records from.
    stop_event : threading.Event
        Cooperative shutdown signal.
    model_path : str
        Filesystem path to the ONNX model artifact.
    db_path : str
        Filesystem path to the SQLite database.
    registry : ModelRegistry | None
        M2 model registry for hot-swap awareness.
    drift_monitor : DriftMonitor | None
        M2 drift monitor instance.
    drift_signal : threading.Event | None
        M2 drift-trigger signal for the optimizer thread.
    trigger_meta_queue : queue.Queue | None
        M2 trigger metadata queue for the optimizer thread.
    """
    # --- Load ONNX model ---
    # M2: resolve initial model path via registry if available.
    active_model_path = registry.current_model_path() if registry else model_path

    try:
        session = ort.InferenceSession(active_model_path)
        model_version = _compute_model_version(active_model_path)
        logger.info(
            "ONNX model loaded: %s (version: %s, input: %s)",
            active_model_path,
            model_version,
            session.get_inputs()[0].shape,
        )
    except Exception:
        logger.error(
            "FATAL: Failed to load ONNX model from '%s'", active_model_path,
            exc_info=True,
        )
        stop_event.set()
        return

    # --- Initialize SQLite ---
    try:
        conn = db.init_db(db_path)
    except Exception:
        logger.error(
            "FATAL: Failed to initialize SQLite at '%s'", db_path,
            exc_info=True,
        )
        stop_event.set()
        return

    logger.info("Engine thread started — consuming from queue")

    # Validate model input dimension against FEATURE_ORDER
    model_input_shape = session.get_inputs()[0].shape
    if model_input_shape is not None and len(model_input_shape) == 2:
        expected_features = model_input_shape[1]
        if expected_features is not None and expected_features != NUM_FEATURES:
            logger.error(
                "FATAL: ONNX model expects %d features but FEATURE_ORDER has %d. "
                "This is a configuration error.",
                expected_features,
                NUM_FEATURES,
            )
            stop_event.set()
            db.close_db(conn)
            return

    # --- Helper: check for model hot-swap (M2) ---
    def _check_model_swap() -> None:
        """Re-resolve current model from the registry; reload session if changed.

        Design choice: per-inference-call check.  registry.current_model_path()
        is a string comparison under a lightweight lock — essentially free.
        This guarantees zero-inference staleness after a swap.
        """
        nonlocal session, model_version, active_model_path

        if registry is None:
            return

        new_path = registry.current_model_path()
        if new_path == active_model_path:
            return

        # Model path changed — reload
        try:
            new_session = ort.InferenceSession(new_path)
            new_version = registry.current_model_version()
            logger.info(
                "Engine hot-reload: %s → %s (path=%s)",
                model_version,
                new_version,
                new_path,
            )
            session = new_session
            model_version = new_version
            active_model_path = new_path
        except Exception:
            logger.error(
                "Failed to reload ONNX model after swap — retaining previous",
                exc_info=True,
            )

    # --- Helper: update drift monitor (M2) ---
    def _update_drift(record: FlowRecord, risk_score: float) -> None:
        """Feed drift monitor and signal optimizer on drift detection.

        Spec M2 §5.1, §5.2: drift check is synchronous, optimizer signal
        is non-blocking.
        """
        if drift_monitor is None:
            return

        try:
            drifted_features = drift_monitor.update(record, risk_score)
            if drifted_features:
                logger.info(
                    "Drift detected: features=%s at flow_id=%s",
                    drifted_features,
                    record.get("flow_id", "UNKNOWN"),
                )

                if drift_signal is not None and trigger_meta_queue is not None:
                    # Non-blocking signal — if optimizer is already running,
                    # the signal coalesces (drift_signal stays set, queue is
                    # maxsize=1 so put_nowait may silently drop, which is
                    # correct coalescing behavior).
                    if not drift_signal.is_set():
                        try:
                            trigger_meta_queue.put_nowait({
                                "features": drifted_features,
                                "timestamp": time.time(),
                            })
                        except queue.Full:
                            # Coalesced — optimizer will pick up existing metadata.
                            logger.debug(
                                "Trigger meta queue full — drift signal coalesced"
                            )
                        drift_signal.set()
                    else:
                        logger.debug(
                            "Drift signal already set — coalescing "
                            "(optimizer is running)"
                        )
        except Exception:
            # Drift monitor errors must never crash the engine thread.
            logger.error(
                "Drift monitor update failed", exc_info=True
            )

    # --- Main loop ---
    try:
        while not stop_event.is_set():
            # Block with timeout so we periodically check stop_event (spec §6)
            try:
                record = in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # --- M2: check for model hot-swap ---
            _check_model_swap()

            # --- Score the flow ---
            inference_ts = time.time()
            try:
                t0 = time.perf_counter()
                risk_score, predicted_class = score_flow(record, session)
                inference_latency_ms = (time.perf_counter() - t0) * 1000.0

                logger.debug(
                    "Inference: flow_id=%s risk=%.4f class=%s latency=%.2fms",
                    record["flow_id"],
                    risk_score,
                    predicted_class,
                    inference_latency_ms,
                )
            except Exception:
                # Spec §6: ONNX inference exceptions — log error with flow_id,
                # skip persistence for that record, continue loop.
                # Must not crash the engine thread.
                logger.error(
                    "Inference failed for flow_id=%s — skipping",
                    record.get("flow_id", "UNKNOWN"),
                    exc_info=True,
                )
                continue

            # --- M2: update drift monitor ---
            _update_drift(record, risk_score)

            # --- Persist to SQLite ---
            # Spec §6: SQLite write exceptions — log error, continue loop.
            db.insert_prediction(
                conn=conn,
                record=record,
                risk_score=risk_score,
                predicted_class=predicted_class,
                model_version=model_version,
                inference_ts=inference_ts,
                inference_latency_ms=inference_latency_ms,
            )

    except Exception:
        # Spec §6: uncaught exceptions must be logged before thread exit,
        # and must set stop_event for clean shutdown.
        logger.error(
            "Engine thread encountered an unhandled exception",
            exc_info=True,
        )
        stop_event.set()
    finally:
        # --- Drain remaining queue items ---
        drained = 0
        while True:
            try:
                record = in_queue.get_nowait()
                inference_ts = time.time()
                try:
                    t0 = time.perf_counter()
                    risk_score, predicted_class = score_flow(record, session)
                    inference_latency_ms = (time.perf_counter() - t0) * 1000.0
                    db.insert_prediction(
                        conn=conn,
                        record=record,
                        risk_score=risk_score,
                        predicted_class=predicted_class,
                        model_version=model_version,
                        inference_ts=inference_ts,
                        inference_latency_ms=inference_latency_ms,
                    )
                    drained += 1
                except Exception:
                    logger.error(
                        "Inference/write failed during drain for flow_id=%s",
                        record.get("flow_id", "UNKNOWN"),
                        exc_info=True,
                    )
            except queue.Empty:
                break

        if drained > 0:
            logger.info("Drained %d remaining records from queue", drained)
        else:
            logger.info("Queue empty at shutdown — nothing to drain")

        db.close_db(conn)
        logger.info("Engine thread stopped")
