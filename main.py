"""
ChronoGuard v1.0 — Application Entrypoint (M1 + M2)

Wires together the sniffer thread, engine thread, and optimizer thread
with shared thread-safe resources and cooperative shutdown via threading.Event.

M2 additions:
    - ModelRegistry for hot-swap model management.
    - DriftMonitor for streaming ADWIN drift detection.
    - Optimizer thread (3rd thread) for async drift-triggered retraining.

Handles SIGINT/SIGTERM for clean shutdown (spec §7: all threads exit
within 5 seconds, queue drained or drain-abandonment logged, DB closed
without corruption).
"""

from __future__ import annotations

import logging
import os
import queue
import signal
import sys
import threading
import time

from config import (
    ADWIN_DELTA,
    DB_PATH,
    DRIFT_MONITORED_FEATURES,
    GOLDEN_HOLDOUT_PATH,
    INTERFACE,
    LOG_FILE_PATH,
    LOG_LEVEL,
    MODEL_PATH,
    QUEUE_MAX_SIZE,
)
from drift_monitor import DriftMonitor
from engine import run_engine
from flow_state import FlowRecord
from model_registry import ModelRegistry
from optimizer import run_optimizer
from sniffer import run_sniffer

# ---------------------------------------------------------------------------
# Logging Setup (spec §6: use Python's logging module, not print)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.DEBUG),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# M3 §6.2: Add FileHandler so ui/data_access.get_log_tail() can read
# structured log lines from a file.  This is the one narrow permitted
# exception to §3's file-modification restriction — additive only.
_file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
_file_handler.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.DEBUG))
_file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
)
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("chronoguard.main")

# Suppress overly verbose scapy warnings at runtime
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)


def main() -> int:
    """Application entrypoint.

    Returns
    -------
    int
        Exit code (0 for clean shutdown, non-zero for errors).
    """
    # --- Startup config summary (spec §6: info level) ---
    logger.info("=" * 60)
    logger.info("ChronoGuard v1.0 — M2 Drift Detection & Adaptive Retraining")
    logger.info("  Interface   : %s", INTERFACE)
    logger.info("  Model path  : %s", MODEL_PATH)
    logger.info("  DB path     : %s", DB_PATH)
    logger.info("  Queue size  : %d", QUEUE_MAX_SIZE)
    logger.info("  Holdout path: %s", GOLDEN_HOLDOUT_PATH)
    logger.info("  ADWIN delta : %.4f", ADWIN_DELTA)
    logger.info("  Log level   : %s", LOG_LEVEL)
    logger.info("=" * 60)

    # --- Validate prerequisites ---
    if not os.path.isfile(MODEL_PATH):
        logger.error(
            "FATAL: ONNX model not found at '%s'. "
            "Generate a placeholder with: python scripts/_TEMP_generate_placeholder_model.py",
            MODEL_PATH,
        )
        return 1

    if not os.path.isfile(GOLDEN_HOLDOUT_PATH):
        logger.warning(
            "Golden holdout not found at '%s'. "
            "Optimizer will fail on retrain attempts until holdout is provided. "
            "Generate with: python scripts/generate_golden_holdout.py",
            GOLDEN_HOLDOUT_PATH,
        )

    # --- Shared resources ---
    flow_queue: queue.Queue[FlowRecord] = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    stop_event = threading.Event()

    # M2: Model registry, drift monitor, and optimizer coordination
    model_registry = ModelRegistry(MODEL_PATH)
    drift_monitor_inst = DriftMonitor(
        monitored_features=list(DRIFT_MONITORED_FEATURES),
        adwin_delta=ADWIN_DELTA,
    )
    drift_signal = threading.Event()
    trigger_meta_queue: queue.Queue = queue.Queue(maxsize=1)

    # --- Signal handling (spec §7: SIGINT/SIGTERM → cooperative shutdown) ---
    def _signal_handler(signum: int, frame) -> None:  # noqa: ANN001
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating shutdown", sig_name)
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # --- Start threads ---
    # M1 threads (unchanged interfaces)
    sniffer_thread = threading.Thread(
        target=run_sniffer,
        args=(INTERFACE, flow_queue, stop_event),
        name="sniffer-thread",
        daemon=False,  # spec §6: no daemon-thread force-kill reliance
    )

    engine_thread = threading.Thread(
        target=run_engine,
        args=(flow_queue, stop_event, MODEL_PATH, DB_PATH),
        kwargs={
            "registry": model_registry,
            "drift_monitor": drift_monitor_inst,
            "drift_signal": drift_signal,
            "trigger_meta_queue": trigger_meta_queue,
        },
        name="engine-thread",
        daemon=False,
    )

    # M2: Optimizer thread (3rd thread — dormant until drift signal)
    optimizer_thread = threading.Thread(
        target=run_optimizer,
        args=(
            drift_signal,
            trigger_meta_queue,
            stop_event,
            DB_PATH,
            GOLDEN_HOLDOUT_PATH,
            model_registry,
        ),
        name="optimizer-thread",
        daemon=False,
    )

    startup_ts = time.monotonic()

    engine_thread.start()
    sniffer_thread.start()
    optimizer_thread.start()

    startup_elapsed = time.monotonic() - startup_ts
    logger.info("All threads started in %.2fs (sniffer, engine, optimizer)", startup_elapsed)

    # --- Wait for shutdown ---
    try:
        # Block main thread until stop_event is set (by signal handler or
        # by a thread encountering a fatal error).
        while not stop_event.is_set():
            # Check thread health periodically
            if not sniffer_thread.is_alive() and not stop_event.is_set():
                logger.warning("Sniffer thread died unexpectedly — shutting down")
                stop_event.set()
                break
            if not engine_thread.is_alive() and not stop_event.is_set():
                logger.warning("Engine thread died unexpectedly — shutting down")
                stop_event.set()
                break
            if not optimizer_thread.is_alive() and not stop_event.is_set():
                logger.warning("Optimizer thread died unexpectedly — shutting down")
                stop_event.set()
                break
            stop_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — initiating shutdown")
        stop_event.set()

    # --- Join threads (spec §7: exit within 5 seconds) ---
    # Also set drift_signal so optimizer thread unblocks from wait()
    drift_signal.set()

    join_timeout = 5.0
    shutdown_ts = time.monotonic()

    sniffer_thread.join(timeout=join_timeout)
    remaining = join_timeout - (time.monotonic() - shutdown_ts)

    if remaining > 0:
        engine_thread.join(timeout=remaining)
    remaining = join_timeout - (time.monotonic() - shutdown_ts)

    if remaining > 0:
        optimizer_thread.join(timeout=remaining)

    # Report shutdown status
    threads_alive = []
    if sniffer_thread.is_alive():
        threads_alive.append("sniffer")
    if engine_thread.is_alive():
        threads_alive.append("engine")
    if optimizer_thread.is_alive():
        threads_alive.append("optimizer")

    if threads_alive:
        logger.warning(
            "Shutdown incomplete — threads still alive after %.1fs: %s",
            time.monotonic() - shutdown_ts,
            ", ".join(threads_alive),
        )
        return 1

    logger.info(
        "Clean shutdown completed in %.2fs",
        time.monotonic() - shutdown_ts,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
