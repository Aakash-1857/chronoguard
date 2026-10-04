# ChronoGuard v1.0 — Test Plan

**Status:** Living Document
**Version:** 0.1.0

This document defines the testing strategy across all milestones. Milestone-specific acceptance criteria live in their respective `specs/M*.md` files; this document defines the **categories and general methodology** each milestone must satisfy where applicable.

---

## 1. Unit Tests

Scope: individual functions/classes in isolation (5-tuple extraction, flow eviction logic, ONNX pre/post-processing, ADWIN wrapper behavior, Optuna objective function).

Standard: deterministic, no network/filesystem dependency where avoidable (mock capture sources, in-memory SQLite for isolated tests).

---

## 2. Integration Tests

Scope: multi-component interaction — Sniffer → Queue → Engine → SQLite; Engine → Drift Evaluator → Optimizer trigger; Optimizer → Model Registry → Engine hot-swap.

Standard: run against a small, fixed replay PCAP fixture with known expected flow counts and classifications.

---

## 3. Streaming Tests

Scope: sustained ingestion under `tcpreplay` at defined packet rates (e.g., 1,000 pps). Validates queue backpressure handling, no unbounded growth, correct flow finalization timing.

---

## 4. Drift Detection Tests

Scope: synthetic distribution-shift injection (e.g., replaying a PCAP with altered packet-size distribution mid-stream) to confirm ADWIN fires within expected latency and does not false-positive on stationary streams.

---

## 5. Retraining Tests

Scope: validate the full drift → Optuna tune → warm-start fit → golden holdout gate → hot-swap/reject path. Includes explicit test of the **reject** path (inject a poisoned/degraded batch, confirm champion model is retained and exception is logged).

---

## 6. Failure Recovery Tests

Scope: capture interface failure mid-run, malformed packet injection, SQLite write contention, ONNX runtime load failure, optimizer thread crash. Validate the system degrades gracefully (logs, does not crash core threads) per each milestone's exception-handling strategy.

---

## 7. Stress Tests

Scope: sustained high-throughput replay (rate escalation) to identify the ingestion ceiling and confirm behavior at/beyond capacity (graceful degradation vs. crash).

---

## 8. Performance Benchmarks

Scope: end-to-end latency (packet arrival → SQLite commit), inference latency (ONNX call only), throughput ceiling (flows/sec sustained with zero loss).

---

## 9. Memory Profiling

Scope: long-duration soak run confirming bounded memory in the flow hash map and no leak across repeated hot-swap cycles.

---

## 10. Regression Tests

Scope: fixed replay fixture + fixed model artifact reproduces identical classification outputs across code changes unless an intentional model/logic change is recorded in `DECISIONS.md`.

---

## Methodology Notes

- Each milestone spec's **Acceptance Criteria** section defines the specific pass/fail thresholds for that milestone; this document defines *what kinds* of tests are expected, not the numeric targets themselves.
- Test artifacts (fixtures, synthetic drift PCAPs, golden holdout sets) should be versioned and referenced explicitly in the relevant milestone spec's Deliverables section.
