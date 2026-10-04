# ChronoGuard v1.0 — Architecture Decision Records

**Status:** Living Document
**Version:** 0.1.0

Each entry records a finalized architectural trade-off: the decision, alternatives considered, trade-offs, rationale, and consequences for later milestones.

---

## ADR-001: Packet Capture Library — scapy (over pypcap/libpcap direct binding)

**Decision:** Use `scapy` for packet capture and parsing in `sniffer.py`.

**Alternatives Considered:** `pypcap`/direct `libpcap` bindings; `pyshark` (tshark wrapper).

**Trade-offs:** `scapy`'s pure-Python packet parsing carries more per-packet overhead than a thin `libpcap` binding, and was the primary throughput risk flagged going into M1. `pyshark` was rejected outright — it shells out to `tshark` per packet, which is far too slow for a 1,000 pps sustained target.

**Justification:** M1's accepted implementation sustained 1,000 pps with zero packet loss using `scapy`, meeting the Acceptance Criteria without requiring the lower-level `pypcap` binding. Given this, the added implementation/maintenance complexity of a `libpcap` binding is not justified at current throughput targets.

**Consequences:** If a future milestone (M4 packaging/hardening, or a post-v1.0 scale-up) raises the target throughput materially above 1,000 pps, this decision should be revisited — `scapy` is the more likely ceiling versus the SQLite write path or ONNX inference.

---

## ADR-002: Concurrency Model — Threads (over separate processes) for Sniffer/Engine

**Decision:** Sniffer and Engine run as two threads within a single process, communicating via `queue.Queue`, per M1 spec §6.

**Alternatives Considered:** Separate OS processes communicating via `multiprocessing.Queue` or a lightweight IPC/socket mechanism.

**Trade-offs:** Threads share the GIL; CPU-bound work on either thread can stall the other. Processes would isolate CPU-bound retraining fully but add serialization overhead and startup/lifecycle complexity.

**Justification:** Packet capture is I/O-bound, and ONNX Runtime releases the GIL during inference calls, so the two M1 threads do not meaningfully contend for the GIL. This was validated empirically: M1's accepted implementation met all throughput, latency, and clean-shutdown criteria under the threaded model.

**Consequences:** This decision is **scoped to M1 only**. M2 introduces genuinely CPU-bound work (Optuna hyperparameter sweeps, LightGBM/CatBoost warm-start fitting) — see ADR-003 below for the M2-specific extension of this decision.

---

## ADR-003: Async Optimizer Isolation — Dedicated Thread, Deferred to Process Isolation if Needed

**Decision:** For M2, the retraining/optimization workflow (`optimizer.py`) runs in its own dedicated thread, separate from the Sniffer and Engine threads, coordinated via a drift-trigger signal (not a shared queue).

**Alternatives Considered:** Running retraining as a separate OS process from the start; running it inline within the Engine thread (rejected immediately — would block live inference).

**Trade-offs:** A dedicated thread is simpler to implement and share state with (e.g., direct access to the SQLite connection pool, direct reference-swap of the loaded ONNX session) than a separate process, but inherits GIL contention risk during CPU-bound Optuna/LightGBM/CatBoost fitting — this is the one place in the system where M1's "threads are sufficient" finding does *not* automatically carry over, because retraining is genuinely CPU-bound rather than I/O-bound.

**Justification:** Starting with thread isolation keeps M2 scoped and testable; the retraining workload is infrequent (drift-triggered, not continuous) and bounded (fixed N=2,000 record window, 3-iteration Optuna sweep), so brief GIL contention during a retrain cycle is an acceptable trade-off versus the added complexity of process isolation, IPC, and cross-process model handoff — provided M2's Acceptance Criteria confirm live inference latency does not degrade unacceptably during an active retrain cycle.

**Empirical Validation (M2 acceptance):** The initial latency measurement submitted during M2 review covered ONNX inference only and omitted the SQLite write, producing implausibly low numbers (0.0045ms median / 0.0133ms p95) that were rejected during review. Corrected measurement against the full engine hot-loop (real ONNX inference + real SQLite WAL-mode write) produced: baseline median 0.0190ms / p95 0.0505ms; under concurrent retrain load, median 0.0212ms / p95 0.0340ms — a p95 ratio of 0.67×, well under the 1.5× threshold. Thread isolation is confirmed sufficient for M2's workload profile.

**Consequences:** M2's spec must include an explicit acceptance criterion measuring inference latency *during* an active retrain cycle. If that criterion cannot be met with thread isolation, this ADR must be revisited and upgraded to process isolation (`multiprocessing`) before M2 can be accepted.

---

## ADR-004: Model Library — LightGBM (over CatBoost) for M2 Retraining

**Decision:** Use LightGBM as the sole GBDT library for the warm-start retraining pipeline (`optimizer.py`), with `init_model`-based incremental fitting.

**Alternatives Considered:** CatBoost — both were named as viable options in the ChronoGuard technical specification (§4) with no prior binding preference in `DECISIONS.md`.

**Trade-offs:** CatBoost has stronger native handling of high-cardinality categorical features (e.g., destination port) without manual encoding, which is directly relevant to network flow data. LightGBM has a simpler, more battle-tested `init_model` warm-start path and lighter dependency footprint, and integrates more directly with the `skl2onnx`/ONNX export path already used in M1.

**Justification:** LightGBM's warm-start mechanics were the more direct fit for M2's specific requirement (incremental fitting on a fixed 2,000-record drift-triggered window, not full categorical-heavy retraining from scratch), and this is reflected in the implementation. This ADR formally closes the open model-library choice flagged as pending in `specs/M2_DRIFT_RETRAIN.md` §9.

**Consequences:** If a future milestone reveals meaningful accuracy loss from LightGBM's categorical handling on `dst_port`-like high-cardinality fields, this ADR should be revisited in favor of CatBoost. This is a candidate check for M2's post-acceptance monitoring, not a blocking condition today.

**Status:** Final — M2 accepted; LightGBM's `init_model` warm-start path and ONNX export (via `onnxmltools.convert_lightgbm`, corrected during M2 acceptance review after an initial LGBMClassifier-wrapper export bug) are confirmed working end-to-end.

---

## ADR-005: LightGBM → ONNX Export — `onnxmltools.convert_lightgbm` (over sklearn-wrapper conversion)

**Decision:** Export candidate LightGBM Boosters to ONNX using `onnxmltools.convert_lightgbm()` directly on the raw `Booster` object.

**Alternatives Considered:** Wrapping the Booster in an `LGBMClassifier` (scikit-learn API) and converting via `skl2onnx`, as originally implemented.

**Trade-offs:** The `LGBMClassifier`-wrapper approach failed on LightGBM 4.7.0 because `n_classes_` became a read-only property, causing an `AttributeError` during wrapper construction. This exception was being silently absorbed by the optimizer's generic exception handler and misreported as a golden-holdout `FAIL` (with a valid `candidate_accuracy=1.0`) — a serialization bug masquerading as a model-quality rejection. Direct Booster conversion avoids the wrapper entirely and has no such dependency on sklearn-API property semantics.

**Justification:** Discovered and fixed during M2 acceptance review when Criterion 6 (golden holdout gate correctly accepts a valid candidate) could not be deterministically proven. The fix was verified with a passing hot-swap, a loadable ONNX artifact, and a real forward-pass check.

**Consequences:** Any future LightGBM version bump should re-verify this export path, since it depends on `onnxmltools`'s Booster-conversion support tracking LightGBM's Booster API rather than its sklearn-wrapper API. This is a good candidate for a regression test pinned to library versions in M4 hardening.

---

## ADR-006: Sniffer Captures All `lo0` Traffic Indiscriminately — Noted Risk, Deferred to M4

**Context:** During M3's concurrent throughput acceptance test (dashboard running alongside core runtime), sniffer-observed packet volume rose 3.4× (7,083 → 24,068 over ~16s) versus the no-dashboard baseline, even though tcpreplay-layer throughput and loss were unaffected (0 loss, 1000.16 pps, identical in both runs — M3 §8 Criterion #3 PASSES on its literal metric). The excess volume was Streamlit's own HTTP/WebSocket traffic on `lo0`, logged at ~100/sec as discarded/skipped packets.

**Decision:** Not fixed now. The core acceptance criterion (no throughput degradation) holds, so this does not block M3 acceptance. However, this is a real operational risk under sustained real-world use — high-frequency discard logging has disk I/O and log-growth implications that weren't part of any milestone's original scope.

**Consequences:** M4 (Packaging & Deployment Hardening) should evaluate whether the sniffer needs an explicit BPF capture filter (e.g., excluding the dashboard's own port) or whether discard-path logging should be rate-limited/sampled rather than per-packet. Flagging here so it isn't rediscovered as a surprise later.

---

## Change Log
- **v0.1.0** — ADR-001 and ADR-002 recorded following M1 acceptance. ADR-003 recorded as a forward-looking decision for M2, pending validation during M2 implementation.
- **v0.2.0** — ADR-004 recorded (LightGBM model library choice), resolving the §3/§9 tension flagged in the M2 state audit. Marked provisional pending final M2 acceptance.
- **v0.3.0** — M2 accepted. ADR-003 updated with corrected empirical latency evidence (full ONNX+SQLite path). ADR-004 marked Final. ADR-005 recorded (ONNX export fix via `onnxmltools.convert_lightgbm`).
- **v0.4.0** — ADR-006 recorded: sniffer captures all `lo0` traffic indiscriminately, causing 3.4× log volume when the M3 dashboard runs concurrently. Throughput unaffected; flagged for M4 evaluation.
