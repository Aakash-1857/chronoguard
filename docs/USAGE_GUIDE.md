# ChronoGuard v1.0 — Usage Guide

## 1. Prerequisites & Installation

### System Requirements

| Requirement | Detail |
|---|---|
| **Python** | 3.10 or later |
| **OS** | macOS or Linux |
| **Capture privilege** | Raw socket access required for packet capture only (see below) |

### Install Dependencies

```bash
pip install -r requirements.txt
```

All dependencies are version-pinned in `requirements.txt` to the exact versions validated against the full test suite.

> **Verified by:** `scripts/run_smoke_test.py` — installs into a clean environment and passes all 167 tests across M1+M2+M3.

### Capture Permissions

Raw socket packet capture requires elevated privileges. The core runtime (sniffer) requires this access.

**macOS:**
```bash
sudo python run.py
```

**Linux — Option A (sudo):**
```bash
sudo python run.py
```

**Linux — Option B (recommended — non-root via capabilities):**
```bash
# Grant CAP_NET_RAW to your Python binary (one-time setup):
sudo setcap cap_net_raw+ep $(which python3)

# Then run without sudo:
python run.py
```

> **Verified by:** `tests/test_sniffer.py::TestPacketProcessing` — tests confirm the sniffer processes packets correctly when capture privilege is available and gracefully halts when denied. (Note: The dashboard running entirely unprivileged is an architectural design of `run.py` and is not explicitly verified by an automated test).

---

## 2. Quick Start

### Start the Full System

```bash
# macOS:
sudo python run.py

# Linux with CAP_NET_RAW:
python run.py
```

`run.py` starts the **core runtime** first (sniffer + engine + optimizer threads), waits for it to signal readiness (SQLite database file created), then starts the **Streamlit dashboard** on port 8501.

### Generate & Replay Test Traffic

ChronoGuard captures packets from the loopback interface. To generate test traffic:

```bash
# Step 1: Generate a synthetic PCAP fixture (~6,000 packets, 60+ flows):
python scripts/generate_throughput_fixture.py
# Output: tests/fixtures/throughput_fixture.pcap

# Step 2: Replay on loopback (in a separate terminal):
sudo tcpreplay --intf1=lo0 --pps=1000 tests/fixtures/throughput_fixture.pcap
```

> **macOS:** loopback interface is `lo0`. **Linux:** use `--intf1=lo` and set `CHRONOGUARD_INTERFACE=lo`.

> **Verified by:** `tests/test_flow_state.py` (flow aggregation), `tests/test_engine.py` (feature vector assembly + ONNX scoring), `tests/test_db.py` (SQLite persistence).

### Core Runtime Only (No Dashboard)

```bash
sudo python run.py --no-dashboard
```

### Custom Dashboard Port

```bash
sudo python run.py --dashboard-port 8502
```

---

## 3. Feature Walkthrough

### 3.1 Live Ingestion & Scoring (M1)

**What it does:** Captures raw packets from the loopback interface, aggregates them into 5-tuple flow records (by source/destination IP, port, protocol), extracts 14 statistical features (duration, inter-arrival time, packet sizes, rates), scores each flow via an ONNX model, and persists results to SQLite.

**How to observe it working:**

1. Start ChronoGuard and replay a PCAP (see Quick Start above).
2. Query the database:
   ```bash
   sqlite3 chronoguard.db "SELECT flow_id, risk_score, predicted_class, inference_latency_ms FROM flow_predictions ORDER BY timestamp DESC LIMIT 5;"
   ```
3. Each row represents a completed flow with its risk score (0.0–1.0), predicted class (0=benign, 1=attack), and inference latency.

**Key behaviors:**
- Flows are finalized on TCP FIN/RST or after 15 seconds of inactivity (configurable via `config.INACTIVITY_TIMEOUT`).
- The 14-feature vector follows a fixed ordering defined in `config.FEATURE_ORDER`.
- ONNX inference releases the GIL, so it does not block the capture thread.

> **Verified by:** `tests/test_flow_state.py` (5-tuple canonicalization, FIN/RST finalization, timeout eviction), `tests/test_engine.py` (feature vector assembly against FEATURE_ORDER), `tests/test_sniffer.py` (mock packet processing, queue integration), `tests/test_db.py` (schema creation, WAL mode, upsert).

### 3.2 Drift Detection (M2)

**What it does:** Monitors three feature distributions in real time using River's ADWIN streaming drift detector. When a statistically significant distributional shift is detected, a drift signal fires to trigger retraining.

**Monitored features:**
- `risk_score` — the model's own output (most direct behavioral drift signal)
- `flow_duration` — sensitive to traffic-mix changes (short HTTP vs. long streaming sessions)
- `fwd_packets_per_sec` — sensitive to volumetric attacks and protocol-mix changes

**How to trigger and observe a drift event:**

1. Replay two different PCAPs in sequence — one with normal traffic, then one with a significantly different distribution *(Note: these paths are illustrative pseudocode)*:
   ```bash
   # Terminal 1: start ChronoGuard
   sudo python run.py
   
   # Terminal 2: replay normal traffic
   sudo tcpreplay --intf1=lo0 --pps=500 /path/to/normal_illustrative.pcap
   
   # Then replay shifted traffic
   sudo tcpreplay --intf1=lo0 --pps=500 /path/to/shifted_illustrative.pcap
   ```

2. Watch for drift events in the logs:
   ```
   [INFO] chronoguard.drift_monitor: ADWIN drift detected on feature 'flow_duration'
   ```

3. Or lower ADWIN sensitivity to trigger drift on minimal data:
   ```bash
   CHRONOGUARD_ADWIN_DELTA=0.5 sudo -E python run.py
   ```

> **Verified by:** `tests/test_drift_monitor.py` (ADWIN correctly fires on injected synthetic drift, reset after fire), `tests/test_integration_m2.py::TestRetrainCycle::test_pass_path` (full drift→retrain→swap pipeline).

### 3.3 Adaptive Retraining (M2)

**What it does:** When drift is detected, the optimizer thread pulls the most recent 2,000 flow predictions from SQLite, runs a bounded Optuna hyperparameter sweep (3 trials, 120s budget), fits a warm-start LightGBM model, validates against a golden holdout, and either hot-swaps the new model into production (PASS) or retains the champion (FAIL).

**How to observe a retrain cycle:**

1. After drift triggers (see §3.2), watch the logs for:
   ```
   [INFO] chronoguard.optimizer: Drift-triggered retrain starting...
   [INFO] chronoguard.optimizer: Optuna sweep: 3 trials completed
   [INFO] chronoguard.optimizer: Golden holdout: candidate_accuracy=0.97, base_accuracy=0.95
   [INFO] chronoguard.optimizer: PASS — hot-swapping model (version: a1b2c3d4e5f6)
   ```

2. Query the `model_lineage` table:
   ```bash
   sqlite3 chronoguard.db "SELECT retrain_id, triggering_feature, status, candidate_accuracy, base_accuracy, model_version FROM model_lineage ORDER BY triggered_ts;"
   ```

**Interpreting `model_lineage` rows:**

| Column | Meaning |
|---|---|
| `retrain_id` | Unique 8-char UUID per retrain attempt |
| `triggering_feature` | Which feature(s) triggered drift |
| `status` | `PASS` (model swapped) or `FAIL` (champion retained) |
| `candidate_accuracy` | Candidate accuracy on golden holdout |
| `base_accuracy` | Baseline accuracy threshold |
| `model_version` | SHA-256 hash (first 12 chars) of new model if PASS |
| `fail_reason` | NULL if PASS; descriptive reason if FAIL |

**Status values:**
- **PASS**: Candidate met the accuracy gate (≥0.95 × baseline) and was hot-swapped.
- **FAIL (accuracy_gate_failed)**: Candidate below threshold. Champion retained.
- **FAIL (atomic_swap_failed)**: Candidate passed holdout but disk write failed.
- **FAIL (exception: ...)**: Error during the pipeline.

> **Verified by:** `tests/test_optimizer.py` (training window extraction, Optuna sweep bounds, warm-start fit), `tests/test_model_registry.py` (atomic swap, concurrent reads during swap), `tests/test_golden_holdout.py` (holdout evaluation), `tests/test_integration_m2.py` (full PASS path, FAIL path, lineage completeness, latency under concurrent retrain load).

### 3.4 The Dashboard (M3)

**What it does:** A strictly read-only Streamlit dashboard that polls the core runtime's SQLite database every 500ms and renders four zones.

**The four zones:**

1. **Status Bar** (top, full width) — Three live metrics:
   - *LIVE · Throughput*: current ingestion rate in flows/sec.
   - *Drift Status*: HEALTHY (green badge) or DRIFT (amber badge).
   - *Active Model*: model version hash and timestamp of last swap.
   - A pulsing cyan dot (Pattern A animation) indicates live operation.

2. **Feature Tracking Chart** (middle-left) — A Plotly time-series chart showing:
   - Live metric line in cyan (`status.signal` token).
   - Drift-fire vertical rules in amber with monospace timestamps.
   - ADWIN bounds corridor (dashed lines, 40% opacity fill) when available.
   - Feature selector dropdown (risk_score, flow_duration, fwd_packets_per_sec).

3. **System Log Stream** (middle-right) — Live-tailing monospace log output with semantic coloring: green for DEBUG, cyan for INFO, amber for WARNING, red for ERROR/CRITICAL.

4. **Model Lineage Table** (bottom, full width) — Retrain history with PASS (green badge) / FAIL (red badge) status, accuracy metrics, and timestamps.

**Restyling the dashboard (Iteration Map):**

All visual constants are defined in `ui/theme.py`. No component hardcodes colors, fonts, or spacing.

| Want to... | Change this |
|---|---|
| Change the accent color | `ui/theme.py` → `status.signal` |
| Change alert/error red | `ui/theme.py` → `status.critical` |
| Use a different font | `ui/theme.py` → typography section, then re-sync `.streamlit/config.toml` |
| Widen/narrow the log panel | `ui/app.py` → `_LAYOUT_COLUMN_RATIOS` |
| Change any label/message | `ui/copy.py` only |
| Adjust spacing/density | `ui/theme.py` → spacing scale |

After changing a token, run: `pytest tests/test_theme.py::TestConfigTomlSync` to verify sync.

> **Verified by:** `tests/test_data_access.py` (read-only connection enforcement, all query functions), `tests/test_theme.py` (token completeness, config.toml sync, no hardcoded values in components), `tests/test_copy.py` (all string constants exist), `tests/test_components.py` (empty-state handling), `tests/test_integration_m3.py` (full render against fixture DB, drift events in chart data, read-only end-to-end).

---

## 4. Shutdown

### Via `run.py` (Recommended)

Press **Ctrl+C** (sends SIGINT). The shutdown sequence:

1. Dashboard stops first (it's the read-only consumer — stopping it first avoids it polling the runtime mid-teardown).
2. Core runtime receives SIGINT, drains queues, joins all three threads (sniffer, engine, optimizer) within 5 seconds.
3. `run.py` reports final status: clean shutdown or any orphaned processes.

**Expected log output:**
```
[INFO] run.py: Received SIGINT — initiating coordinated shutdown
[INFO] run.py: Sent SIGTERM to Dashboard (PID 12346)
[INFO] run.py: Dashboard exited (code 0)
[INFO] run.py: Sent SIGINT to Core runtime (PID 12345)
[INFO] run.py: Core runtime exited (code 0)
[INFO] run.py: Clean shutdown completed in 2.35s — no orphaned processes
```

### Direct Core Runtime (Without `run.py`)

```bash
# Press Ctrl+C in the terminal running main.py
```

Expected log output:
```
[INFO] chronoguard.main: Received SIGINT — initiating shutdown
[INFO] chronoguard.main: Clean shutdown completed in 1.24s
```

> **Verified by:** `tests/test_integration_m2.py` (clean shutdown of all three threads — sniffer, engine, optimizer — within 5 seconds via stop_event).

---

## 5. Troubleshooting

### Permission Errors on Capture

**Symptom:**
```
FATAL: Permission denied binding to interface 'lo0'. Raw socket capture typically requires sudo or CAP_NET_RAW.
```

**Fix:**
- **macOS:** Run with `sudo python run.py`.
- **Linux:** Either `sudo python run.py` or grant `CAP_NET_RAW`:
  ```bash
  sudo setcap cap_net_raw+ep $(which python3)
  ```

> **Verified by:** `tests/test_sniffer.py` — sniffer handles `PermissionError` gracefully, logs FATAL, and sets `stop_event`.

### Port Already in Use (Dashboard)

**Symptom:**
```
Address already in use: port 8501
```

**Fix:** Either kill the existing process on port 8501 or use a different port:
```bash
sudo python run.py --dashboard-port 8502
```

### ADR-006: Loopback Noise (Dashboard + Core Runtime)

**What it is:** When the Streamlit dashboard and core runtime run concurrently, the dashboard's own HTTP/WebSocket traffic on the loopback interface (`lo0`) is captured by the sniffer. These non-IP packets are discarded (they don't affect flow classification), but previously generated a debug log line per packet (~100/sec), inflating log volume ~3.4× versus the baseline.

**How it's addressed (M4 — Option B, rate-limited logging):** The "Skipping non-IP packet" debug log in `flow_state.py` is now rate-limited to at most once per second. When suppression occurs, the log line includes a count of suppressed messages. This reduces log noise ~100× while preserving full capture coverage — no packets are filtered, so no legitimate traffic can be accidentally excluded.

**What to expect:** During concurrent operation, you'll see at most one debug line per second like:
```
[DEBUG] flow_state: Skipping non-IP packet (97 similar messages suppressed)
```
instead of ~100 individual lines per second.

> **Verified by:** The rate-limiting is implemented in `flow_state.py` at the `ingest_packet` level. The original "Skipping non-IP packet" behavior is preserved in `tests/test_sniffer.py::TestNonIPSkip` and `tests/test_flow_state.py` — non-IP packets still return `None` and are correctly skipped.

---

## 6. Architecture Summary

```
PCAP → tcpreplay → lo0 interface → Sniffer thread (scapy capture)
     → 5-tuple aggregation (FlowTable) → queue.Queue → Engine thread
     → ONNX scoring → SQLite WAL-mode write (flow_predictions)
                    → ADWIN drift update (risk_score, flow_duration, fwd_packets_per_sec)
                          │
                          ▼ (drift detected)
                    Optimizer thread (dormant until signaled):
                       → SQLite: pull last 2,000 records
                       → Optuna sweep (3 trials, 120s budget)
                       → warm-start LightGBM fit
                       → golden holdout validation
                       → PASS: atomic model hot-swap → PASS lineage row
                       → FAIL: retain champion → FAIL lineage row

     Streamlit dashboard (separate process, unprivileged):
       → polls SQLite (read-only, mode=ro) every 500ms
       → renders: status bar, feature chart, log stream, lineage table
```

For the full specification, see [`docs/SPECIFICATION.md`](SPECIFICATION.md) §5.

### Key Architectural Decisions

| ADR | Decision | Rationale |
|---|---|---|
| ADR-001 | scapy for capture | Built-in dissection; met 1,000 pps target |
| ADR-002 | Threads for sniffer/engine | Capture is I/O-bound; ONNX releases GIL |
| ADR-003 | Dedicated optimizer thread | Retrain is infrequent/bounded; p95 latency 0.67× under load |
| ADR-004 | LightGBM for retraining | Simpler `init_model` warm-start; lighter footprint |
| ADR-005 | `onnxmltools.convert_lightgbm` | Avoids LGBMClassifier wrapper bug on LightGBM ≥4.7 |
| ADR-006 | Rate-limited discard logging | Additive fix; lower risk than BPF filter modification |

See [`docs/DECISIONS.md`](DECISIONS.md) for full rationale on each decision.

---

## Configuration Reference

| Parameter | Env Variable | Default | Milestone |
|---|---|---|---|
| Interface | `CHRONOGUARD_INTERFACE` | `lo0` | M1 |
| Model Path | `CHRONOGUARD_MODEL_PATH` | `model.onnx` | M1 |
| DB Path | `CHRONOGUARD_DB_PATH` | `chronoguard.db` | M1 |
| Log Level | `CHRONOGUARD_LOG_LEVEL` | `DEBUG` | M1 |
| Log File | `CHRONOGUARD_LOG_FILE` | `chronoguard.log` | M3 |
| ADWIN Delta | `CHRONOGUARD_ADWIN_DELTA` | `0.002` | M2 |
| Retrain Window | `CHRONOGUARD_RETRAIN_WINDOW_SIZE` | `2000` | M2 |
| Optuna Trials | `CHRONOGUARD_OPTUNA_N_TRIALS` | `3` | M2 |
| Time Budget | `CHRONOGUARD_OPTUNA_TIME_BUDGET_SEC` | `120` | M2 |
| Accuracy Gate | `CHRONOGUARD_ACCURACY_GATE_THRESHOLD` | `0.95` | M2 |
| Holdout Path | `CHRONOGUARD_HOLDOUT_PATH` | `golden_holdout.csv` | M2 |

All parameters are defined in `config.py` and overridable via environment variables.
