"""
ChronoGuard v1.0 — Unified Orchestration Entrypoint (M4)

Starts the core runtime and the Streamlit dashboard together with correct
startup ordering and coordinated clean shutdown on SIGINT/SIGTERM.

Architecture (M4 §4):
    run.py
      ├─► spawns: core runtime (main.py) — may require elevated privilege
      ├─► spawns: streamlit dashboard (ui/app.py) — runs unprivileged
      └─► on SIGINT/SIGTERM: stops dashboard first, then core runtime

Usage:
    # macOS (sudo for raw-socket capture):
    sudo python run.py

    # Linux with CAP_NET_RAW pre-granted:
    python run.py

    # With environment overrides:
    CHRONOGUARD_INTERFACE=lo CHRONOGUARD_LOG_LEVEL=INFO sudo -E python run.py

    # Skip dashboard (core runtime only):
    python run.py --no-dashboard

    # Custom Streamlit port:
    python run.py --dashboard-port 8502
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Logging — simple console output for the orchestrator itself.
# The core runtime and dashboard have their own logging configurations.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] run.py: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("chronoguard.run")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_MAIN_PY = os.path.join(_PROJECT_ROOT, "main.py")
_APP_PY = os.path.join(_PROJECT_ROOT, "ui", "app.py")
_DB_PATH_DEFAULT = os.environ.get("CHRONOGUARD_DB_PATH", "chronoguard.db")

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------
_STARTUP_READINESS_TIMEOUT = 15.0   # seconds to wait for core runtime readiness
_STARTUP_READINESS_POLL = 0.5       # poll interval during readiness check
_STARTUP_SETTLE_DELAY = 1.0         # settle delay after readiness detected
_DASHBOARD_SHUTDOWN_TIMEOUT = 3.0   # seconds to wait for dashboard to exit
_RUNTIME_SHUTDOWN_TIMEOUT = 8.0     # seconds to wait for core runtime to exit


def _wait_for_runtime_ready(db_path: str, timeout: float) -> bool:
    """Wait for the core runtime to signal readiness.

    Readiness signal: the SQLite database file exists.  The engine creates
    this file on startup via ``db.init_db()``.

    Returns True if ready within timeout, False otherwise.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.isfile(db_path):
            return True
        time.sleep(_STARTUP_READINESS_POLL)
    return False


def _terminate_and_wait(
    proc: subprocess.Popen,
    label: str,
    timeout: float,
    sig: int = signal.SIGTERM,
) -> bool:
    """Send a signal to a process and wait for it to exit.

    Returns True if the process exited within timeout.
    """
    if proc.poll() is not None:
        logger.info("%s already exited (code %d)", label, proc.returncode)
        return True

    try:
        proc.send_signal(sig)
        logger.info("Sent %s to %s (PID %d)", signal.Signals(sig).name, label, proc.pid)
    except ProcessLookupError:
        logger.info("%s already exited", label)
        return True

    try:
        proc.wait(timeout=timeout)
        logger.info("%s exited (code %d)", label, proc.returncode)
        return True
    except subprocess.TimeoutExpired:
        logger.warning(
            "%s did not exit within %.1fs — sending SIGKILL", label, timeout
        )
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass
        return False


def main() -> int:
    """Orchestrate the full ChronoGuard system."""
    parser = argparse.ArgumentParser(
        description="ChronoGuard v1.0 — Unified Orchestration Entrypoint",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Start the core runtime only (skip the Streamlit dashboard).",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Port for the Streamlit dashboard (default: 8501).",
    )
    args = parser.parse_args()

    python = sys.executable
    db_path = os.path.abspath(
        os.environ.get("CHRONOGUARD_DB_PATH", _DB_PATH_DEFAULT)
    )

    logger.info("=" * 60)
    logger.info("ChronoGuard v1.0 — Orchestration Entrypoint (M4)")
    logger.info("  Python       : %s", python)
    logger.info("  Core runtime : %s", _MAIN_PY)
    logger.info("  Dashboard    : %s (port %d)", _APP_PY, args.dashboard_port)
    logger.info("  DB path      : %s", db_path)
    logger.info("  Dashboard    : %s", "disabled" if args.no_dashboard else "enabled")
    logger.info("=" * 60)

    # --- Validate prerequisites ---
    if not os.path.isfile(_MAIN_PY):
        logger.error("FATAL: main.py not found at %s", _MAIN_PY)
        return 1
    if not args.no_dashboard and not os.path.isfile(_APP_PY):
        logger.error("FATAL: ui/app.py not found at %s", _APP_PY)
        return 1

    # --- Track child processes ---
    runtime_proc: subprocess.Popen | None = None
    dashboard_proc: subprocess.Popen | None = None
    shutdown_initiated = False

    def _shutdown_handler(signum: int, frame) -> None:  # noqa: ANN001
        nonlocal shutdown_initiated
        if shutdown_initiated:
            return  # Prevent re-entrant shutdown
        shutdown_initiated = True
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating coordinated shutdown", sig_name)

    # Install signal handlers
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    exit_code = 0

    try:
        # ---- Phase 1: Start core runtime ----
        logger.info("Starting core runtime...")
        runtime_proc = subprocess.Popen(
            [python, _MAIN_PY],
            cwd=_PROJECT_ROOT,
            # Inherit environment so CHRONOGUARD_* env vars propagate.
            env=os.environ.copy(),
        )
        logger.info("Core runtime started (PID %d)", runtime_proc.pid)

        # Wait for readiness
        logger.info(
            "Waiting for core runtime readiness (DB file: %s)...", db_path
        )
        if not _wait_for_runtime_ready(db_path, _STARTUP_READINESS_TIMEOUT):
            logger.warning(
                "Core runtime readiness check timed out after %.1fs — "
                "proceeding anyway (DB may not exist yet)",
                _STARTUP_READINESS_TIMEOUT,
            )
        else:
            logger.info("Core runtime ready (DB file detected)")

        # Brief settle delay for threads to fully initialize
        time.sleep(_STARTUP_SETTLE_DELAY)

        # Check the runtime didn't die during startup
        if runtime_proc.poll() is not None:
            logger.error(
                "Core runtime exited during startup (code %d)",
                runtime_proc.returncode,
            )
            return runtime_proc.returncode or 1

        # ---- Phase 2: Start dashboard (if enabled) ----
        if not args.no_dashboard:
            logger.info("Starting Streamlit dashboard on port %d...", args.dashboard_port)
            dashboard_proc = subprocess.Popen(
                [
                    python, "-m", "streamlit", "run", _APP_PY,
                    "--server.port", str(args.dashboard_port),
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false",
                ],
                cwd=_PROJECT_ROOT,
                env=os.environ.copy(),
            )
            logger.info("Dashboard started (PID %d)", dashboard_proc.pid)

        # ---- Phase 3: Wait for shutdown signal or child exit ----
        logger.info("System running. Press Ctrl+C to shut down.")

        while not shutdown_initiated:
            # Check child health
            if runtime_proc.poll() is not None:
                logger.warning(
                    "Core runtime exited unexpectedly (code %d)",
                    runtime_proc.returncode,
                )
                shutdown_initiated = True
                exit_code = runtime_proc.returncode or 1
                break
            if dashboard_proc and dashboard_proc.poll() is not None:
                logger.warning(
                    "Dashboard exited unexpectedly (code %d)",
                    dashboard_proc.returncode,
                )
                # Dashboard exit is not fatal — keep running core
            time.sleep(1.0)

    except KeyboardInterrupt:
        if not shutdown_initiated:
            shutdown_initiated = True
            logger.info("KeyboardInterrupt — initiating coordinated shutdown")

    # ---- Phase 4: Coordinated shutdown ----
    logger.info("-" * 40)
    logger.info("Shutdown sequence starting")
    shutdown_start = time.monotonic()

    # Step 1: Stop dashboard FIRST (it's the read-only consumer)
    if dashboard_proc and dashboard_proc.poll() is None:
        _terminate_and_wait(
            dashboard_proc,
            "Dashboard",
            _DASHBOARD_SHUTDOWN_TIMEOUT,
            signal.SIGTERM,
        )

    # Step 2: Stop core runtime (SIGINT to trigger its cooperative shutdown)
    if runtime_proc and runtime_proc.poll() is None:
        _terminate_and_wait(
            runtime_proc,
            "Core runtime",
            _RUNTIME_SHUTDOWN_TIMEOUT,
            signal.SIGINT,
        )

    shutdown_elapsed = time.monotonic() - shutdown_start

    # ---- Phase 5: Report final state ----
    orphans = []
    if runtime_proc and runtime_proc.poll() is None:
        orphans.append(f"core-runtime (PID {runtime_proc.pid})")
    if dashboard_proc and dashboard_proc.poll() is None:
        orphans.append(f"dashboard (PID {dashboard_proc.pid})")

    if orphans:
        logger.warning(
            "Shutdown incomplete — orphaned processes: %s",
            ", ".join(orphans),
        )
        exit_code = 1
    else:
        logger.info(
            "Clean shutdown completed in %.2fs — no orphaned processes",
            shutdown_elapsed,
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
