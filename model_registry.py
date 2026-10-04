"""
ChronoGuard v1.0 — Model Registry (M2)

Provides atomic ONNX model load/swap logic.  The Engine thread's single
source of truth for "which model file is currently active."

Spec references: M2 §5.3 (ModelRegistry interface), §6 (hot-swap atomicity).

Design choice (documented per §6 directive):
    The Engine thread re-resolves current_model_path() on every inference call.
    Since the registry stores an atomically-updated path string, this check is
    a single string comparison under a lightweight lock — essentially free.
    This guarantees the engine never holds a stale reference for more than
    zero inferences after a swap.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import threading

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


class ModelRegistry:
    """Thread-safe registry for the active ONNX model path.

    The Engine thread reads current_model_path() on every inference call.
    The Optimizer thread calls atomic_swap() after a successful retrain.
    Both methods are serialized via a lightweight lock — reads are not
    blocked during the filesystem operations of a swap, only during the
    path-string update (nanoseconds).

    Spec §5.3 contract:
        current_model_path() -> str
        atomic_swap(new_onnx_path, model_version) -> None
    """

    def __init__(self, initial_model_path: str) -> None:
        self._lock = threading.Lock()
        self._current_path: str = os.path.abspath(initial_model_path)
        self._current_version: str = _compute_model_version(self._current_path)
        logger.info(
            "ModelRegistry initialized: path=%s version=%s",
            self._current_path,
            self._current_version,
        )

    def current_model_path(self) -> str:
        """Return the absolute path to the currently active ONNX model.

        Thread-safe — called by the Engine thread on every inference call.
        """
        with self._lock:
            return self._current_path

    def current_model_version(self) -> str:
        """Return the version hash of the currently active model."""
        with self._lock:
            return self._current_version

    def atomic_swap(self, new_onnx_path: str, model_version: str) -> None:
        """Atomically swap the production ONNX model.

        Uses a write-temp-file + os.rename pattern (POSIX-atomic) so the
        Engine thread never observes a partially-written ONNX file.

        Steps:
            1. Copy new_onnx_path to a temp file in the same directory as
               the current model (same filesystem — required for atomic rename).
            2. os.rename(temp_file, target_path) — atomic on POSIX.
            3. Update the internal path reference under lock.

        If any step fails, the previous model file is left untouched and
        an exception is raised (caller logs a FAIL lineage row).

        Spec §5.3, §6 (hot-swap atomicity).
        """
        current = self.current_model_path()
        target_dir = os.path.dirname(current)

        # Generate the target path for the new model version.
        # We overwrite the existing model path (same filename) so the engine
        # doesn't need to track changing filenames — it just re-reads the file
        # when the version changes.
        target_path = current

        old_version = self.current_model_version()

        # Step 1: Copy new model to a temp file in the same directory
        fd, tmp_path = tempfile.mkstemp(
            suffix=".onnx.tmp", dir=target_dir
        )
        try:
            os.close(fd)
            shutil.copy2(new_onnx_path, tmp_path)

            # Step 2: Atomic rename (POSIX guarantees this is atomic on the
            # same filesystem)
            os.replace(tmp_path, target_path)

            # Step 3: Update internal state under lock
            with self._lock:
                self._current_path = os.path.abspath(target_path)
                self._current_version = model_version

            logger.info(
                "Model hot-swap complete: %s → %s (path=%s)",
                old_version,
                model_version,
                target_path,
            )
        except Exception:
            # Clean up temp file if it still exists
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            logger.error(
                "atomic_swap failed — previous model retained",
                exc_info=True,
            )
            raise
