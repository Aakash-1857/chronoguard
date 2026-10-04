"""
ChronoGuard v1.0 — Consolidated Smoke Test Runner (M4)

Runs the existing M1, M2, and M3 acceptance test suites in sequence via
pytest and reports one consolidated pass/fail summary.

This script orchestrates existing tests — it does NOT define new test logic
(M4 §3: out of scope).

Usage:
    python scripts/run_smoke_test.py

Exit code:
    0 if all milestones pass, non-zero if any fails.
"""

from __future__ import annotations

import os
import sys
import time

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so imports resolve correctly.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import pytest — the only test infrastructure dependency.
try:
    import pytest
except ImportError:
    print("ERROR: pytest is required.  pip install pytest", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Test suite definitions — grouped by milestone.
# Each group references ONLY existing test files; no new test logic.
# ---------------------------------------------------------------------------
_TESTS_DIR = os.path.join(_PROJECT_ROOT, "tests")

MILESTONES: list[dict] = [
    {
        "name": "M1",
        "description": "Live Ingestion & Static ONNX Scoring",
        "files": [
            os.path.join(_TESTS_DIR, "test_flow_state.py"),
            os.path.join(_TESTS_DIR, "test_sniffer.py"),
            os.path.join(_TESTS_DIR, "test_engine.py"),
            os.path.join(_TESTS_DIR, "test_db.py"),
        ],
    },
    {
        "name": "M2",
        "description": "Drift Detection & Adaptive Retraining",
        "files": [
            os.path.join(_TESTS_DIR, "test_drift_monitor.py"),
            os.path.join(_TESTS_DIR, "test_optimizer.py"),
            os.path.join(_TESTS_DIR, "test_model_registry.py"),
            os.path.join(_TESTS_DIR, "test_golden_holdout.py"),
            os.path.join(_TESTS_DIR, "test_integration_m2.py"),
        ],
    },
    {
        "name": "M3",
        "description": "Control Panel (Streamlit Dashboard)",
        "files": [
            os.path.join(_TESTS_DIR, "test_data_access.py"),
            os.path.join(_TESTS_DIR, "test_theme.py"),
            os.path.join(_TESTS_DIR, "test_copy.py"),
            os.path.join(_TESTS_DIR, "test_components.py"),
            os.path.join(_TESTS_DIR, "test_integration_m3.py"),
        ],
    },
]


class _TestResult:
    """Result of running one milestone's test suite."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.exit_code: int = -1
        self.tests_run: int = 0
        self.passed: int = 0
        self.failed: int = 0
        self.errors: int = 0
        self.duration_sec: float = 0.0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class _ResultCollector:
    """Pytest plugin that collects test counts without adding new logic."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.total = 0

    def pytest_runtest_logreport(self, report) -> None:  # noqa: ANN001
        if report.when == "call":
            self.total += 1
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
        elif report.when in ("setup", "teardown") and report.failed:
            self.errors += 1
            self.total += 1


def _run_milestone(milestone: dict) -> _TestResult:
    """Run a single milestone's test suite and collect results."""
    result = _TestResult(milestone["name"], milestone["description"])

    # Verify all test files exist
    missing = [f for f in milestone["files"] if not os.path.isfile(f)]
    if missing:
        print(f"  WARNING: Missing test files: {missing}", file=sys.stderr)

    existing_files = [f for f in milestone["files"] if os.path.isfile(f)]
    if not existing_files:
        print(f"  SKIP: No test files found for {milestone['name']}")
        return result

    collector = _ResultCollector()

    start = time.monotonic()
    result.exit_code = pytest.main(
        ["-v", "--tb=short", "-q"] + existing_files,
        plugins=[collector],
    )
    result.duration_sec = time.monotonic() - start

    result.tests_run = collector.total
    result.passed = collector.passed
    result.failed = collector.failed
    result.errors = collector.errors

    return result


def main() -> int:
    """Run all milestone test suites and print a consolidated summary."""
    print("=" * 72)
    print("ChronoGuard v1.0 — Consolidated Smoke Test (M4)")
    print("=" * 72)
    print()

    results: list[_TestResult] = []

    for milestone in MILESTONES:
        print(f"--- {milestone['name']}: {milestone['description']} ---")
        result = _run_milestone(milestone)
        results.append(result)
        status = "PASS" if result.ok else "FAIL"
        print(
            f"  {status}: {result.passed}/{result.tests_run} passed "
            f"in {result.duration_sec:.1f}s"
        )
        print()

    # --- Summary table ---
    print("=" * 72)
    print("CONSOLIDATED SUMMARY")
    print("=" * 72)
    header = f"{'Milestone':<12} {'Tests Run':>10} {'Passed':>8} {'Failed':>8} {'Duration':>10} {'Status':>8}"
    print(header)
    print("-" * len(header))

    total_run = 0
    total_passed = 0
    total_failed = 0
    total_duration = 0.0
    any_failed = False

    for r in results:
        status = "PASS" if r.ok else "FAIL"
        if not r.ok:
            any_failed = True
        print(
            f"{r.name:<12} {r.tests_run:>10} {r.passed:>8} "
            f"{r.failed + r.errors:>8} {r.duration_sec:>9.1f}s {status:>8}"
        )
        total_run += r.tests_run
        total_passed += r.passed
        total_failed += r.failed + r.errors
        total_duration += r.duration_sec

    print("-" * len(header))
    overall = "PASS" if not any_failed else "FAIL"
    print(
        f"{'TOTAL':<12} {total_run:>10} {total_passed:>8} "
        f"{total_failed:>8} {total_duration:>9.1f}s {overall:>8}"
    )
    print()

    if any_failed:
        print("RESULT: FAIL — one or more milestone suites failed.")
        return 1

    print("RESULT: PASS — all milestone suites passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
