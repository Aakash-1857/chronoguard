# M1 — Live Ingestion & Static ONNX Scoring

**Status:** Ready for Implementation
**Owner:** Antigravity (implementer) — reviewed by Meta-Orchestrator
**Depends on:** None (foundational milestone)
**Blocks:** M2 (Drift Detection & Adaptive Retraining)

---

## 1. Objective

Implement a stable, multi-threaded runtime that:

1. Replays a PCAP file over the loopback interface (`lo`) via `tcpreplay` (external, operator-invoked — not launched by the application).
2. Captures live packets off `lo` in a dedicated sniffer thread.
3. Aggregates packets into stateful 5-tuple flow records using defined stateless and stateful feature rules.
4. Finalizes flows on teardown (FIN/RST) or inactivity timeout (15s) and pushes completed flow feature vectors onto a thread-safe queue.
5. Consumes the queue in a dedicated engine thread, scores each flow via a pre-trained, pre-compiled static ONNX model, and persists the flow record + prediction to a local SQLite database (WAL mode).

No drift detection, no retraining, no UI. The model is static and provided as a pre-existing artifact (training/compilation of this artifact is **not** part of M1 — see Out of Scope).

---

## 2. Scope

Antigravity must build:

- `sniffer.py` — packet capture + 5-tuple flow aggregation + queue emission.
- `engine.py` — queue consumption + ONNX inference + SQLite persistence.
- `flow_state.py` — the flow table data structure, eviction logic, feature computation (shared by `sniffer.py`).
- `db.py` — SQLite schema definition, WAL configuration, connection/write helpers.
- `config.py` — centralized configuration (interface name, timeout thresholds, DB path, model path, queue size limits).
- `main.py` — process entrypoint that wires sniffer thread + engine thread together with clean startup/shutdown.
- A minimal `requirements.txt` covering M1 dependencies only (`scapy` or `pypcap`, `onnxruntime`, `numpy`; note choice + rationale must align with `docs/DECISIONS.md` — see Section 9 below, Antigravity should flag if a decision is not yet recorded).
- Structured logging across all modules per Section 6.

---

## 3. Out of Scope

Antigravity must **not**:

- Train, retrain, or compile the ONNX model. M1 assumes a pre-existing, valid `model.onnx` artifact is provided at a configured path. If none exists, Antigravity must produce a trivial placeholder-training script (clearly marked as **temporary scaffolding**, not production code) sufficient only to generate a syntactically valid ONNX artifact for pipeline testing — this must be flagged explicitly in the PR/deliverable notes.
- Implement any ADWIN / drift detection logic.
- Implement any Optuna / retraining logic.
- Implement any Streamlit / UI code.
- Modify `docs/SPECIFICATION.md`, `docs/ROADMAP.md`, `docs/TEST_PLAN.md`, or `docs/DECISIONS.md`. These are owned by the Meta-Orchestrator.
- Launch or manage `tcpreplay` itself — it is operator-invoked externally, outside the application process.
- Implement authentication, remote access, or multi-host support.

---

## 4. Architecture Context

M1 implements the top two boxes of the system data flow:

```
PCAP → tcpreplay (external, operator-run) → lo interface
     → [M1] Sniffer Thread (5-tuple aggregation)
     → [M1] queue.Queue
     → [M1] Engine Thread (ONNX scoring)
     → [M1] SQLite (WAL) persistence
     → (M2: ADWIN hook attaches here — not implemented yet)
     → (M3: Streamlit polls SQLite — not implemented yet)
```

The queue is the sole interface boundary between the sniffer and engine threads. This boundary must remain stable, since M2 will attach the drift evaluator to the engine-side consumption path without modifying the sniffer.

---

## 5. Data Contracts

### 5.1 Flow Feature Record (Queue Payload)

Each completed flow pushed onto the queue must be a `dict` (or equivalent typed structure — `TypedDict` or `dataclass` is acceptable and encouraged) with **exactly** these fields, in this logical grouping:

```python
FlowRecord = {
    # Identity
    "flow_id": str,              # deterministic hash of 5-tuple + start timestamp
    "src_ip": str,
    "dst_ip": str,
    "src_port": int,
    "dst_port": int,
    "protocol": int,             # IANA protocol number (e.g. 6=TCP, 17=UDP)

    # Temporal (stateful)
    "flow_start_ts": float,      # epoch seconds, first packet in flow
    "flow_end_ts": float,        # epoch seconds, last packet observed (finalization trigger time)
    "flow_duration": float,      # flow_end_ts - flow_start_ts
    "flow_iat_mean": float,      # mean inter-arrival time across all packets in flow, seconds
    "flow_iat_std": float,       # std dev of inter-arrival times (0.0 if <2 packets)

    # Directional byte/packet distribution (stateless aggregates)
    "fwd_packet_count": int,
    "bwd_packet_count": int,
    "fwd_packet_length_max": int,
    "fwd_packet_length_mean": float,
    "bwd_packet_length_max": int,
    "bwd_packet_length_mean": float,
    "fwd_packets_per_sec": float,
    "bwd_packets_per_sec": float,

    # Termination
    "termination_reason": str,   # one of: "FIN", "RST", "TIMEOUT"
}
```

**Timestamp Authority:** Use the capture-time wall-clock timestamp provided by the packet capture library at the moment the application processes each packet (i.e., `scapy`'s packet `.time` field, which reflects kernel capture time on `lo`). Do **not** attempt to parse or trust any timestamp encoded inside the replayed payload itself. This must be documented as a code comment at the point of timestamp extraction.

**Directionality:** "Forward" = packets where `src` matches the flow's initiating endpoint (first packet of the 5-tuple observed defines the initiator). "Backward" = the reverse direction. A flow's 5-tuple key must be canonicalized (see 5.2) but directionality tracking must be preserved per-packet relative to the *original* initiator, not the canonical key order.

### 5.2 5-Tuple Canonicalization

Because A→B and B→A packets belong to the same logical flow, the flow table key must be canonicalized so both directions map to one entry:

```python
def canonical_key(src_ip, dst_ip, src_port, dst_port, protocol) -> tuple:
    # Sort the two (ip, port) endpoints to produce a direction-independent key.
    # The ORIGINAL (non-canonical) src/dst must still be tracked separately
    # per-packet to preserve forward/backward directionality (see 5.1).
    ...
```

### 5.3 ONNX Model Interface Contract

- Input: a single 2D float32 tensor of shape `(1, N)` where `N` is the number of numeric features the model expects, in a **fixed, documented feature order** stored in `config.py` as `FEATURE_ORDER: list[str]`.
- `FEATURE_ORDER` must reference only fields present in `FlowRecord` (excluding identity/termination string fields).
- Output: a probability/score tensor; engine must extract a scalar risk score (float, 0.0–1.0) and, if the model is multiclass, also the predicted class label (int/str).
- Any mismatch between `FlowRecord` numeric fields and `FEATURE_ORDER` length at runtime must raise an explicit exception (not silently truncate/pad).

### 5.4 SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS flow_predictions (
    flow_id TEXT PRIMARY KEY,
    src_ip TEXT NOT NULL,
    dst_ip TEXT NOT NULL,
    src_port INTEGER NOT NULL,
    dst_port INTEGER NOT NULL,
    protocol INTEGER NOT NULL,
    flow_start_ts REAL NOT NULL,
    flow_end_ts REAL NOT NULL,
    flow_duration REAL NOT NULL,
    flow_iat_mean REAL NOT NULL,
    flow_iat_std REAL NOT NULL,
    fwd_packet_count INTEGER NOT NULL,
    bwd_packet_count INTEGER NOT NULL,
    fwd_packet_length_max INTEGER NOT NULL,
    fwd_packet_length_mean REAL NOT NULL,
    bwd_packet_length_max INTEGER NOT NULL,
    bwd_packet_length_mean REAL NOT NULL,
    fwd_packets_per_sec REAL NOT NULL,
    bwd_packets_per_sec REAL NOT NULL,
    termination_reason TEXT NOT NULL,
    risk_score REAL NOT NULL,
    predicted_class TEXT,
    model_version TEXT NOT NULL,
    inference_ts REAL NOT NULL,
    inference_latency_ms REAL NOT NULL
);
```

`PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=NORMAL;` must be set on connection init. Table creation must be idempotent (`IF NOT EXISTS`).

### 5.5 Function Signatures (Minimum Required)

```python
# flow_state.py
class FlowTable:
    def ingest_packet(self, pkt: ScapyPacket) -> Optional[FlowRecord]:
        """Update or create flow entry. Returns a finalized FlowRecord
        if this packet triggered finalization (FIN/RST), else None."""

    def evict_stale(self, now: float) -> list[FlowRecord]:
        """Called periodically; returns and removes all flows whose
        last-seen timestamp exceeds the 15s inactivity threshold."""

# sniffer.py
def run_sniffer(interface: str, out_queue: "queue.Queue[FlowRecord]", stop_event: threading.Event) -> None: ...

# engine.py
def run_engine(in_queue: "queue.Queue[FlowRecord]", stop_event: threading.Event,
               model_path: str, db_path: str) -> None: ...

def score_flow(record: FlowRecord, session: "onnxruntime.InferenceSession") -> tuple[float, Optional[str]]:
    """Returns (risk_score, predicted_class)."""
```

---

## 6. Implementation Directives

**Concurrency Model:** Two long-lived threads (not processes) for M1:
- Sniffer thread: runs the packet capture loop, owns the `FlowTable`, pushes finalized `FlowRecord`s to `out_queue`. Must also run a periodic eviction sweep (e.g., every 1–2 seconds, either in the same loop or a lightweight timer) to catch inactivity-timeout flows even when no new packets arrive.
- Engine thread: blocks on `in_queue.get(timeout=...)`, runs inference, writes to SQLite.
- Both threads must accept a shared `threading.Event` for cooperative shutdown; no daemon-thread force-kill reliance.
- Justify in your PR notes why threads (not processes) are sufficient for M1: I/O-bound capture + short-lived ONNX Runtime calls (which release the GIL internally) should not require process isolation at this stage. Flag if profiling suggests otherwise.

**Pipeline Ordering:** Capture → canonicalize key → update/create flow entry → (if terminal packet) finalize + enqueue → dequeue → feature vector assembly in `FEATURE_ORDER` → ONNX inference → SQLite write. Each stage must be independently unit-testable (see `docs/TEST_PLAN.md` §1).

**Performance Expectations:** Sustain the M1 Acceptance Criteria throughput (Section 7) with the queue acting as the sole buffer — no additional batching layer required in M1.

**Logging Requirements:** Use Python's `logging` module (not `print`). Minimum required log events:
- Sniffer start/stop, interface bind success/failure.
- Flow finalization (debug level): flow_id + termination_reason.
- Engine: each inference with latency (debug level); any inference exception (error level, with flow_id, non-fatal — continue loop).
- SQLite write failures (error level, non-fatal — continue loop, but count and log a warning if failures exceed a small threshold in a rolling window).
- Startup config summary and clean shutdown confirmation (info level).

**Exception Handling Strategy:**
- Malformed/non-IP packets: skip silently at debug-log level (expected noise on `lo`).
- ONNX inference exceptions: log error with flow_id, skip persistence for that record, continue loop — must not crash the engine thread.
- SQLite write exceptions: log error, continue loop; do not crash the engine thread.
- Sniffer interface bind failure at startup: fatal — log error and exit with non-zero status (this is a startup precondition, not a runtime fault).
- Uncaught exceptions in either thread must be logged before the thread exits, and must set `stop_event` so the other thread and `main.py` can shut down cleanly rather than hang.

---

## 7. Acceptance Criteria

- [ ] Sustain **1,000 packets/sec** replayed via `tcpreplay --pps=1000` on loopback with **zero packet loss** at the capture layer (verify via `tcpreplay` sent-packet count vs. sniffer-observed packet count).
- [ ] Flow finalization correctness: for a fixture PCAP with known flow count (documented in test fixtures), the number of `FlowRecord`s written to SQLite exactly matches the expected flow count.
- [ ] Deterministic ONNX predictions: replaying the same fixture PCAP twice produces bit-identical `risk_score` values for corresponding flows.
- [ ] Memory remains bounded: flow table size does not grow monotonically over a 5-minute sustained replay (validated via periodic size sampling in a soak test).
- [ ] Startup completes (interface bound, model loaded, DB initialized) in under **3 seconds**.
- [ ] Clean shutdown: on `SIGINT`/stop signal, both threads exit within **5 seconds**, queue is drained or drain-abandonment is explicitly logged, DB connection closed without corruption (verify via `PRAGMA integrity_check`).
- [ ] No unhandled exception ever terminates the process outside of the documented fatal startup-failure path.
- [ ] All flow records in SQLite have non-null, schema-valid values for every column.

---

## 8. Deliverables

- `sniffer.py`
- `engine.py`
- `flow_state.py`
- `db.py`
- `config.py`
- `main.py`
- `requirements.txt`
- Temporary placeholder ONNX training/compilation script, **if** no artifact is supplied (clearly marked as scaffolding, e.g. `scripts/_TEMP_generate_placeholder_model.py`)
- Unit tests covering: 5-tuple canonicalization, flow eviction timing, feature vector assembly against `FEATURE_ORDER`, SQLite schema idempotency.
- A short `README_M1.md` describing how to run `tcpreplay` + `main.py` together, including required OS permissions (e.g., raw socket capture typically requires `sudo` or `CAP_NET_RAW`).

---

## 9. Validation Checklist

Before this milestone is marked complete:

- [ ] All Acceptance Criteria (Section 7) pass and are demonstrated with logged evidence (not just claimed).
- [ ] All Deliverables (Section 8) exist and match the Data Contracts (Section 5) exactly — no field drift.
- [ ] Unit tests pass and cover the four areas listed in Section 8.
- [ ] No code touches anything listed in Out of Scope (Section 3).
- [ ] Logging output reviewed for completeness against Section 6 minimums.
- [ ] `docs/DECISIONS.md` has been updated (by the Meta-Orchestrator, post-review) with the finalized scapy-vs-pypcap decision and threads-vs-processes decision, based on Antigravity's implementation notes and any profiling data produced.
- [ ] Explicit sign-off recorded before M2 spec is generated.

---

*End of M1 specification. Hand this document to Antigravity as-is. Do not proceed to M2 until this milestone is implemented, tested, reviewed, and accepted.*
