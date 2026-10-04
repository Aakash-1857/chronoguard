"""
Unit tests for model_registry.py — ChronoGuard M2

Covers (per M2 spec §7/§8):
  - atomic_swap uses rename pattern (file existence).
  - Concurrent current_model_path() reads during swap are safe.
  - Swap with invalid source leaves previous model untouched.
"""

from __future__ import annotations

import os
import shutil
import threading
import time

import pytest

from model_registry import ModelRegistry, _compute_model_version


@pytest.fixture
def model_dir(tmp_path):
    """Create a temp directory with a dummy ONNX model."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"ONNX_MODEL_V1_PLACEHOLDER")
    return tmp_path


@pytest.fixture
def registry(model_dir):
    """Create a ModelRegistry pointing to the dummy model."""
    return ModelRegistry(str(model_dir / "model.onnx"))


class TestModelRegistryInit:
    def test_initial_path(self, registry, model_dir) -> None:
        """Registry returns the initial model path."""
        assert registry.current_model_path() == str(
            (model_dir / "model.onnx").resolve()
        )

    def test_initial_version(self, registry) -> None:
        """Registry computes a version hash on init."""
        version = registry.current_model_version()
        assert isinstance(version, str)
        assert len(version) == 12


class TestAtomicSwap:
    def test_swap_replaces_model(self, registry, model_dir) -> None:
        """atomic_swap replaces the model file and updates version."""
        # Create a new model file
        new_model = model_dir / "candidate.onnx"
        new_model.write_bytes(b"ONNX_MODEL_V2_CANDIDATE")
        new_version = _compute_model_version(str(new_model))

        old_version = registry.current_model_version()
        registry.atomic_swap(str(new_model), new_version)

        # Version should change
        assert registry.current_model_version() == new_version
        assert registry.current_model_version() != old_version

        # The model file should be readable
        path = registry.current_model_path()
        assert os.path.isfile(path)

    def test_swap_atomic_file_exists(self, registry, model_dir) -> None:
        """After swap, the target model file exists and is not a partial write."""
        new_model = model_dir / "candidate2.onnx"
        content = b"ONNX_MODEL_V3_FULL_CONTENT"
        new_model.write_bytes(content)
        new_version = _compute_model_version(str(new_model))

        registry.atomic_swap(str(new_model), new_version)

        # Read back the swapped file — it should be the full content
        with open(registry.current_model_path(), "rb") as f:
            actual = f.read()
        assert actual == content

    def test_swap_invalid_source_preserves_previous(self, registry) -> None:
        """Swap with a non-existent source file leaves previous model intact."""
        old_path = registry.current_model_path()
        old_version = registry.current_model_version()

        with pytest.raises(Exception):
            registry.atomic_swap("/nonexistent/model.onnx", "bad_version")

        # Previous model should be untouched
        assert registry.current_model_path() == old_path
        assert registry.current_model_version() == old_version
        assert os.path.isfile(old_path)


class TestConcurrentAccess:
    def test_concurrent_reads_during_swap(self, registry, model_dir) -> None:
        """Multiple threads reading current_model_path() during swap don't crash."""
        new_model = model_dir / "concurrent_candidate.onnx"
        new_model.write_bytes(b"ONNX_MODEL_CONCURRENT")
        new_version = _compute_model_version(str(new_model))

        errors: list[str] = []
        read_count = 0
        lock = threading.Lock()

        def reader():
            nonlocal read_count
            for _ in range(100):
                try:
                    path = registry.current_model_path()
                    assert isinstance(path, str)
                    assert len(path) > 0
                    with lock:
                        read_count += 1
                except Exception as e:
                    errors.append(str(e))

        # Start reader threads
        readers = [threading.Thread(target=reader) for _ in range(5)]
        for t in readers:
            t.start()

        # Perform swap concurrently
        time.sleep(0.01)
        registry.atomic_swap(str(new_model), new_version)

        for t in readers:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"Concurrent read errors: {errors}"
        assert read_count == 500  # 5 threads × 100 reads
