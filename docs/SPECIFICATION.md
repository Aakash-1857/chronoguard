# ChronoGuard v1.0 — System Specification

**Status:** Living Document
**Owner:** Meta-Orchestrator / Principal ML Architect
**Version:** 0.1.0 (Initialized — Pre-M1)

---

## 1. Purpose

ChronoGuard is a self-adapting, streaming Network Intrusion Detection System (NIDS) that:

1. Ingests real network traffic (via `tcpreplay`-driven loopback replay of CSE-CIC-IDS2018/2019 PCAPs) rather than static CSVs.
2. Aggregates raw packets into 5-tuple flow records in real time.
3. Scores flows sub-millisecond using a compiled ONNX inference artifact derived from a LightGBM/CatBoost GBDT model.
4. Monitors feature and prediction distributions using River's ADWIN streaming drift detector.
5. Triggers isolated, async, non-blocking retraining (Optuna-tuned, warm-started) when drift is detected.
6. Validates retrained models against a protected golden holdout before hot-swapping — never degrading below 95% of baseline accuracy.
7. Persists all inference and lineage data locally in SQLite (WAL mode).
8. Exposes system state via a polling Streamlit dashboard.

This document is the **single source of truth** for system-level architecture. Milestone-level implementation detail lives in `specs/M*.md`. Trade-off rationale lives in `docs/DECISIONS.md`. Testing strategy lives in `docs/TEST_PLAN.md`.

---

## 2. System Boundaries

**In Scope**
- Local, single-workstation, multi-threaded Python runtime.
- Passive traffic replay ingestion (no live production network integration in v1.0).
- Binary/multiclass flow classification (benign vs. attack category).
- Local persistence and local visualization only.

**Out of Scope (v1.0)**
- Distributed/multi-node deployment (Kafka, Flink, etc.).
- Cloud infrastructure or remote model registries.
- Real production network taps (loopback replay only).
- Authentication/authorization on the Streamlit UI.
- Multi-tenant support.

---

## 3. Component Inventory

| Component | Module | Responsibility |
|---|---|---|
| Sniffer Engine | `sniffer.py` | Packet capture, 5-tuple flow aggregation, queue emission |
| Core Runtime | `engine.py` | Dequeue, ONNX inference, ADWIN update, SQLite commit |
| Async Optimizer | `optimizer.py` | Drift-triggered Optuna tuning, warm-start retrain, golden holdout validation, hot-swap |
| Persistence Layer | SQLite (WAL) | Flow records, prediction logs, model lineage |
| Presentation Layer | Streamlit | Live dashboard, drift visualization, system logs |

---

## 4. Cross-Cutting Architectural Invariants

These invariants apply across **all** milestones and must not be violated by any specification:

1. **Thread Isolation:** The packet-capture thread must never block on inference, drift computation, or retraining. Compute-heavy work is isolated to dedicated threads/processes.
2. **GIL Awareness:** Any CPU-bound retraining logic (Optuna sweeps, tree fitting) must be explicitly justified as thread- vs. process-isolated in the relevant spec, with reasoning recorded in `DECISIONS.md`.
3. **Non-Blocking Hot Reload:** Model swaps must not drop or stall in-flight inference requests.
4. **Deterministic Inference:** Given identical input feature vectors and a fixed model artifact, ONNX output must be bit-for-bit reproducible across runs.
5. **Bounded Memory:** The in-memory flow table must enforce eviction (FIN/RST teardown or 15s inactivity timeout) to prevent unbounded growth.
6. **No Silent Failures:** Every exception path (capture failure, inference failure, drift computation failure, retrain failure) must have an explicit logging and recovery strategy defined in its milestone spec.
7. **Golden Holdout Gate:** No retrained model may enter production without passing `Accuracy_new ≥ 0.95 × Accuracy_base` against the protected holdout set.

---

## 5. Data Flow Summary

```
PCAP → tcpreplay → lo interface → Sniffer (scapy/pypcap)
     → 5-tuple aggregation → queue.Queue → Core Engine
     → ONNX scoring → SQLite (predictions) + ADWIN update
     → [drift fires] → Optimizer thread → Optuna tune → warm-start fit
     → golden holdout validate → pass: hot-swap ONNX / fail: log + retain champion
     → Streamlit polls SQLite every 500ms → dashboard render
```

---

## 6. Milestone Map (Summary — detail in ROADMAP.md)

| Milestone | Name | Status |
|---|---|---|
| M1 | Live Ingestion & Static ONNX Scoring | **Next** |
| M2 | Drift Detection & Adaptive Retraining | Pending M1 |
| M3 | Control Panel (Streamlit Dashboard) | Pending M2 |
| M4 | Packaging & Deployment Hardening | Pending M3 |

---

## 7. Document Control

This file is updated whenever:
- A cross-cutting invariant changes.
- A component's core responsibility changes.
- A milestone completes and system boundaries shift.

All changes should be reflected simultaneously in `ROADMAP.md` and, where relevant, `DECISIONS.md`.
