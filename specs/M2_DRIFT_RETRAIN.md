# M2 — Drift Detection & Adaptive Retraining

**Status:** Ready for Implementation
**Owner:** Antigravity (implementer) — reviewed by Meta-Orchestrator
**Depends on:** M1 (Live Ingestion & Static ONNX Scoring) — **Accepted**
**Blocks:** M3 (Control Panel / Streamlit Dashboard)

---

## 1. Objective

Extend the M1 runtime with:

1. A streaming drift evaluator (River `ADWIN`) attached to the Engine thread's inference path, monitoring a defined set of numeric features and/or prediction scores per flow.
2. A drift-triggered, thread-isolated async optimizer (`optimizer.py`) that: extracts the last N=2,000 SQLite records, runs a bounded 3-iteration Optuna hyperparameter sweep, performs a warm-start incremental fit (`init_model`) on LightGBM/CatBoost, validates the candidate against a protected golden holdout set, and — on pass — atomically hot-swaps the production ONNX model with zero inference interruption; on fail, retains the current champion and logs the rejection.
3. Model lineage persistence: every retrain attempt (pass or fail) is recorded to SQLite for later consumption by M3's dashboard.

---

## 2. Scope

Antigravity must build:

- `drift_monitor.py` — ADWIN wrapper(s), feature selection for monitoring, drift-event signaling.
- `optimizer.py` — the full retrain lifecycle described in Objective #2, running in its own dedicated thread.
- `model_registry.py` — atomic ONNX model load/swap logic; the Engine thread's single source of truth for "which model file is currently active."
- `golden_holdout.py` — golden holdout set loading, protection (read-only access pattern), and evaluation harness (accuracy computation against the holdout).
- Extensions to `engine.py` (from M1): after each inference, feed the relevant feature(s)/score into `drift_monitor.py`; on drift signal, notify `optimizer.py` (non-blocking).
- Extensions to `db.py` / schema: new `model_lineage` table (see §5.4).
- Extensions to `config.py`: ADWIN parameters, Optuna sweep bounds, retrain window size (N=2000), golden holdout path, accuracy-gate threshold (0.95).
- Updated `requirements.txt`: add `river`, `optuna`, `lightgbm` and/or `catboost` (see §9 — Antigravity must confirm actual model library choice against `docs/DECISIONS.md`; if undecided, default to LightGBM per the technical spec's stated preference and flag CatBoost as a documented alternative for a future ADR).
- Unit + integration tests per §8/§9 and `docs/TEST_PLAN.md` §4–5.

---

## 3. Out of Scope

Antigravity must **not**:

- Modify `sniffer.py` or the 5-tuple flow aggregation logic from M1 — the sniffer/queue boundary is stable and untouched.
- Implement any Streamlit/UI code (M3).
- Change the M1 `flow_predictions` schema in a breaking way — only additive schema changes (new tables/columns) are permitted.
- Implement multi-model ensembling, online/incremental-per-sample learning (only batch warm-start retraining as specified), or any retraining trigger mechanism other than ADWIN drift signals (no fixed-interval retraining in v1.0).
- Modify `docs/SPECIFICATION.md`, `docs/ROADMAP.md`, `docs/TEST_PLAN.md`, or `docs/DECISIONS.md`.
- Upgrade the concurrency model to multiprocessing preemptively — thread isolation (ADR-003) is the starting point; only escalate if M2's own Acceptance Criteria (§7) prove it insufficient, and flag this explicitly rather than silently switching.

---

## 4. Architecture Context

M2 attaches to the M1 engine's consumption path and adds a third thread:

```
[M1] Engine Thread: dequeue → ONNX inference → SQLite write
                          │
                          ├──► [M2] drift_monitor.update(features/score)
                          │         │
                          │         ▼ (if drift detected)
                          │    signal optimizer (non-blocking, e.g. threading.Event or Queue)
                          │
[M2] Optimizer Thread (dormant until signaled):
    SQLite: pull last N=2000 records
       → Optuna sweep (3 iterations, bounded time budget)
       → warm-start fit (init_model)
       → golden_holdout.evaluate(candidate)
       → PASS: model_registry.atomic_swap(new_onnx_path)
              → log to model_lineage (status=PASS)
       → FAIL: log to model_lineage (status=FAIL, reason)
              → champion model untouched
```

`model_registry.py` is the only component permitted to change which ONNX file the Engine thread loads for inference. The Engine thread must always resolve "current model" through this registry, never a hardcoded path, so swaps are atomic from its perspective.

---

## 5. Data Contracts

### 5.1 Drift Monitor Interface

```python
# drift_monitor.py
class DriftMonitor:
    def __init__(self, monitored_features: list[str], adwin_delta: float = 0.002):
        """One ADWIN instance per monitored feature/score. adwin_delta is the
        River ADWIN confidence parameter — must be sourced from config.py,
        not hardcoded."""

    def update(self, record: "FlowRecord", risk_score: float) -> list[str]:
        """Feeds each monitored value into its ADWIN instance.
        Returns a list of feature names for which drift fired on this update
        (empty list if none)."""
```

**Monitored set (must be explicit, not implicit):** at minimum, `risk_score` (the model's own output) must be monitored — this is the most direct signal of behavioral drift. Additionally monitor at least two structural input features prone to distributional shift (e.g., `flow_duration`, `fwd_packets_per_sec`) as specified in `config.py: DRIFT_MONITORED_FEATURES`. Document the choice rationale in code comments — this becomes an ADR candidate.

### 5.2 Drift Signal → Optimizer Handoff

Use a `threading.Event` (`drift_signal: threading.Event`) plus a small metadata payload (which feature(s) triggered, timestamp) passed via a single-slot structure (e.g., a `queue.Queue(maxsize=1)` holding the trigger metadata) — not the full flow record. The Engine thread must never block waiting for the Optimizer to consume this signal. If a drift signal arrives while the Optimizer is already running a retrain cycle, it must be coalesced/ignored (no concurrent retrain runs) and logged at debug level.

### 5.3 Optimizer Lifecycle Function Signatures

```python
# optimizer.py
def run_optimizer(drift_signal: threading.Event, trigger_meta_queue: "queue.Queue",
                   stop_event: threading.Event, db_path: str,
                   holdout_path: str, registry: "ModelRegistry") -> None:
    """Long-lived thread entrypoint. Blocks on drift_signal.wait(), then
    executes one full retrain cycle per trigger (coalescing concurrent
    triggers as described in §5.2)."""

def extract_training_window(db_path: str, n: int = 2000) -> "pandas.DataFrame":
    """Pulls the most recent N flow_predictions records, strictly time-ordered
    by inference_ts, excluding any rows flagged as part of the golden holdout."""

def run_optuna_sweep(train_df: "pandas.DataFrame", n_trials: int = 3,
                      time_budget_sec: int = 120) -> dict:
    """Returns best hyperparameter dict found within the bounded sweep.
    Must enforce BOTH n_trials and time_budget_sec as hard ceilings —
    whichever is reached first stops the sweep."""

def warm_start_fit(train_df: "pandas.DataFrame", hyperparams: dict,
                    base_model_path: str) -> "Booster":
    """Loads base_model_path as init_model and fits incrementally on train_df."""

# golden_holdout.py
def evaluate(model: "Booster", holdout_path: str) -> float:
    """Returns accuracy (0.0-1.0) of `model` against the protected,
    read-only golden holdout set."""

# model_registry.py
class ModelRegistry:
    def current_model_path(self) -> str: ...
    def atomic_swap(self, new_onnx_path: str, model_version: str) -> None:
        """Must be safe to call while the Engine thread is concurrently
        calling current_model_path()/loading a session — e.g. write-new-file
        + rename (atomic on POSIX) rather than in-place overwrite."""
```

### 5.4 SQLite Schema Extension — `model_lineage`

```sql
CREATE TABLE IF NOT EXISTS model_lineage (
    retrain_id TEXT PRIMARY KEY,
    triggered_ts REAL NOT NULL,
    triggering_feature TEXT NOT NULL,
    training_window_start_ts REAL NOT NULL,
    training_window_end_ts REAL NOT NULL,
    training_window_size INTEGER NOT NULL,
    hyperparams_json TEXT NOT NULL,
    optuna_trials_run INTEGER NOT NULL,
    candidate_accuracy REAL NOT NULL,
    base_accuracy REAL NOT NULL,
    accuracy_gate_threshold REAL NOT NULL,
    status TEXT NOT NULL,              -- 'PASS' or 'FAIL'
    fail_reason TEXT,                  -- NULL if status='PASS'
    model_version TEXT,                -- NULL if status='FAIL'
    completed_ts REAL NOT NULL,
    retrain_duration_sec REAL NOT NULL
);
```

### 5.5 Golden Holdout Set

A fixed, versioned, read-only dataset (file path in `config.py: GOLDEN_HOLDOUT_PATH`), disjoint from any data ever pulled into a training window. `golden_holdout.py` must never write to this path. Composition (size, class balance, source) must be documented in `README_M2.md` (see Deliverables) since it directly determines gate reliability.

---

## 6. Implementation Directives

**Concurrency Model:** Three threads total (Sniffer, Engine — both unchanged from M1 — plus Optimizer). Optimizer is dormant (blocked on `drift_signal.wait()`) except during an active retrain cycle. Per ADR-003, thread isolation is the starting point; Acceptance Criteria (§7) include an explicit test of inference latency during an active retrain to validate this empirically.

**Pipeline Ordering:** Drift check happens synchronously within the Engine thread's per-flow loop (it must be O(1)-ish per update — ADWIN is designed for this) but must never block on the Optimizer. Retrain execution is fully asynchronous to the Engine/Sniffer threads.

**Hot-Swap Atomicity:** `ModelRegistry.atomic_swap` must use a write-temp-file + `os.rename` pattern (POSIX-atomic) so the Engine thread never observes a partially-written ONNX file. The Engine thread should re-resolve the current model reference on a defined cadence (e.g., check-and-reload-if-changed on each inference call, or on a lightweight polling interval — Antigravity to choose and document, with the constraint that the Engine thread must never hold a stale reference for more than a small bounded number of seconds after a swap).

**Optuna Sweep Bounding:** Hard-enforce both `n_trials=3` and a wall-clock `time_budget_sec` ceiling (value in `config.py`). If the budget is exceeded mid-sweep, stop and use the best trial found so far — never let a sweep run unbounded.

**Logging Requirements (in addition to M1's):**
- Drift fire event (info level): which feature(s), timestamp.
- Coalesced/ignored drift signal while retrain in progress (debug level).
- Retrain lifecycle stages (info level): window extraction size, sweep start/end + trial count, fit duration, holdout accuracy vs. base accuracy vs. threshold, PASS/FAIL decision.
- Hot-swap execution (info level): old model version → new model version.
- Any exception during the retrain cycle (error level) — must not crash the Optimizer thread; the cycle aborts, champion is retained, and the thread returns to `drift_signal.wait()`.

**Exception Handling Strategy:**
- Any failure in the retrain pipeline (data extraction, sweep, fit, holdout eval) results in an aborted cycle, a `FAIL`-status `model_lineage` row (with `fail_reason` populated), and the champion model remains active — never a crash, never a partial/corrupt swap.
- `model_registry.atomic_swap` failures (e.g., disk write error) must leave the previous model file untouched and be logged as a `FAIL` lineage row even if holdout evaluation passed.

---

## 7. Acceptance Criteria

- [ ] ADWIN correctly fires on injected synthetic drift (a fixture PCAP with a deliberately shifted packet-size/timing distribution partway through) within a bounded number of flows/time after the shift begins — exact threshold to be empirically determined and documented, not assumed.
- [ ] ADWIN does **not** false-positive over a 5-minute stationary-distribution replay (zero spurious drift fires).
- [ ] On a drift fire, a retrain cycle completes end-to-end (sweep → fit → holdout eval → decision) within a documented maximum wall-clock time.
- [ ] **Latency-under-load criterion (validates ADR-003):** median and p95 inference latency measured *during* an active retrain cycle does not exceed [1.5×] the baseline (non-retraining) p95 latency established in M1. If this fails, escalate per ADR-003's stated consequence (revisit thread vs. process isolation) rather than silently relaxing the criterion.
- [ ] Golden holdout gate correctly rejects a deliberately degraded candidate model (inject a poisoned training window in a test) — champion model remains active and a `FAIL` row is logged.
- [ ] Golden holdout gate correctly accepts a valid improved/comparable candidate — hot-swap occurs, Engine thread's next inferences use the new model, and a `PASS` row is logged.
- [ ] Hot-swap never causes a dropped, duplicated, or errored inference on the Engine thread (verify via continuous flow processing across a swap event in an integration test).
- [ ] Every retrain attempt (PASS or FAIL) produces exactly one `model_lineage` row with all fields populated per schema.
- [ ] Optuna sweep never exceeds configured `n_trials` or `time_budget_sec` ceilings (verify via test with an artificially slow objective function).
- [ ] Concurrent drift signals during an active retrain are coalesced — no more than one retrain cycle runs at a time.

---

## 8. Deliverables

- `drift_monitor.py`
- `optimizer.py`
- `model_registry.py`
- `golden_holdout.py`
- Updated `engine.py`, `db.py`, `config.py`, `requirements.txt`
- Golden holdout dataset + generation/versioning notes
- Unit tests: ADWIN wrapper behavior, Optuna sweep bounding, atomic swap safety, training-window extraction correctness (time-ordering, holdout exclusion).
- Integration tests: full drift→retrain→swap cycle (pass path), full drift→retrain→reject cycle (fail path), latency-under-load measurement.
- `README_M2.md`: golden holdout composition/provenance, how to trigger a manual test drift event, how to interpret `model_lineage` rows.

---

## 9. Validation Checklist

- [ ] All Acceptance Criteria (§7) pass with logged evidence.
- [ ] All Deliverables (§8) exist and match Data Contracts (§5) exactly.
- [ ] No code touches anything in Out of Scope (§3), especially `sniffer.py`.
- [ ] `docs/DECISIONS.md` ADR-003 is revisited by the Meta-Orchestrator post-review: confirm thread isolation held under the latency-under-load criterion, or escalate to process isolation and record a superseding ADR if not.
- [ ] Model library choice (LightGBM vs. CatBoost) actually used is confirmed and recorded as a new ADR if not already covered.
- [ ] Explicit sign-off recorded before M3 spec is generated.

---

*End of M2 specification. Hand this document to Antigravity as-is. Do not proceed to M3 until this milestone is implemented, tested, reviewed, and accepted.*
