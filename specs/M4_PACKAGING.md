# M4 — Packaging & Deployment Hardening

**Status:** Ready for Implementation
**Owner:** Antigravity (implementer) — reviewed by Meta-Orchestrator
**Depends on:** M1, M2, M3 (all Accepted)
**Blocks:** None — this is the final milestone of ChronoGuard v1.0.

---

## 1. Objective

Turn the accepted M1–M3 system into something a third party can clone, install, and run correctly on the first attempt: unified startup/shutdown across all processes, one reconciled and pinned dependency set, resolution of the ADR-006 loopback-noise finding, a combined final smoke-run that reuses existing acceptance tests, and an accurate usage guide covering every feature actually built.

This milestone is deliberately scoped narrow. It does **not** introduce new stress-test infrastructure, new features, or new architecture — it hardens and documents what already exists and is proven.

---

## 2. Scope

Antigravity must build:

- `run.py` (or `scripts/run_chronoguard.py`) — a single orchestration entrypoint that starts the core runtime and the dashboard together, with correct startup ordering and coordinated clean shutdown on `SIGINT`/`SIGTERM`.
- Consolidated `requirements.txt` — one reconciled, version-pinned file covering all M1–M4 dependencies (replacing/merging the four milestone-incremental files), with a documented resolution for any version conflicts found.
- A resolution for ADR-006 (loopback noise) — implemented in `sniffer.py` or `config.py`, whichever the chosen fix requires (see §6 for the two acceptable options and how to choose between them).
- `scripts/run_smoke_test.py` — a single script that runs the existing M1, M2, and M3 acceptance test suites in sequence and reports one consolidated pass/fail summary. This orchestrates existing tests; it must not define new test logic.
- `docs/USAGE_GUIDE.md` — feature-by-feature usage documentation (see §5 for required contents).
- `docs/DECISIONS.md` update — Antigravity must NOT edit this file directly (per the standing Out of Scope rule); instead, report which option was chosen for ADR-006 resolution so the Meta-Orchestrator can record it.

---

## 3. Out of Scope

Antigravity must **not**:

- Add any new stress-test, load-test, or benchmark infrastructure beyond `run_smoke_test.py`'s orchestration of *existing* tests.
- Add new product features, UI panels, model capabilities, or endpoints of any kind.
- Introduce containerization (Docker/Compose) unless explicitly requested later — not part of this milestone's scope.
- Change any M1/M2/M3 function signature, data contract, or schema.
- Modify `docs/SPECIFICATION.md`, `docs/ROADMAP.md`, `docs/TEST_PLAN.md`, or `docs/DECISIONS.md` directly — report findings for the Meta-Orchestrator to record instead.
- Require the dashboard process to run with elevated (`sudo`/root) privileges — only the packet-capture component may require elevation (see §6).

---

## 4. Architecture Context

M4 adds an orchestration layer above the existing three runtime pieces; it does not change any of them internally:

```
run.py
  ├─► spawns: core runtime (sniffer + engine + optimizer threads) — requires elevated capture privilege
  ├─► spawns: streamlit dashboard (ui/app.py) — runs unprivileged
  └─► on SIGINT/SIGTERM: signals both to shut down cleanly, waits with timeout, reports final state

scripts/run_smoke_test.py
  └─► orchestrates: pytest tests/test_*.py (M1) → tests/test_*_m2.py + unit (M2) → tests/test_*_m3.py (M3)
      reports one consolidated summary; does not invent new assertions
```

---

## 5. Usage Guide Requirements (`docs/USAGE_GUIDE.md`)

Must be anchored to **actual verified behavior** — every claim in this document must trace to something demonstrated in an M1/M2/M3 acceptance test or integration test, not aspirational description. Required sections:

1. **Prerequisites & Installation** — OS/Python version, `pip install -r requirements.txt`, any OS-level capture permission setup (`setcap`/`sudo` requirements), explained plainly.
2. **Quick Start** — the single `run.py` command to bring up the full system, and how to obtain a replay-able PCAP (referencing `scripts/generate_throughput_fixture.py` from M3, and `tcpreplay` usage from M1).
3. **Feature Walkthrough** — one subsection per major capability, each stating what it does and how to observe it working:
   - Live ingestion & scoring (M1) — how to watch flows being classified.
   - Drift detection (M2) — how to trigger and observe an ADWIN drift event.
   - Adaptive retraining (M2) — how to observe a retrain cycle and interpret a `model_lineage` row (PASS vs FAIL).
   - The dashboard (M3) — a walkthrough of each of the four zones and what each one means, including the design token system's Iteration Map (reproduced or linked from `README_M3.md`) for anyone who wants to restyle it later.
4. **Shutdown** — how to cleanly stop everything via `run.py`, what clean shutdown looks like in the logs.
5. **Troubleshooting** — at minimum: permission errors on capture, port-already-in-use for the dashboard, and the ADR-006 loopback-noise behavior (what it is, why it's expected, how the chosen M4 fix addresses it).
6. **Architecture Summary** — a short, accurate recap of the full data flow (may reference `docs/SPECIFICATION.md` §5 rather than duplicate it at length).

---

## 6. Implementation Directives

**Orchestration (`run.py`):** Must start the core runtime first, wait for a confirmed-ready signal (e.g., a log line or a readiness file/socket check — Antigravity's choice, documented), then start the dashboard. On shutdown signal, must stop the dashboard first (it's the read-only consumer; stopping it first avoids it polling a runtime mid-teardown), then signal the core runtime's existing `stop_event` shutdown path (already implemented and tested in M1), and wait with a bounded timeout before reporting any process that didn't exit cleanly.

**Privilege handling:** The core runtime process may require `sudo` for raw-socket capture (as established in M1). `run.py` must not require the *entire* invocation to run as root — document the OS-appropriate options (e.g., `sudo run.py` with an internal privilege-drop after binding the capture socket, or a documented `setcap cap_net_raw+ep` alternative) and pick one, explaining the choice in `USAGE_GUIDE.md`'s Prerequisites section. Do not silently require root for the dashboard.

**Dependency reconciliation:** Diff all `requirements.txt` versions accumulated across M1–M3, resolve any version conflicts (pin to the version that satisfies all milestones' test suites — verify by actually reinstalling into a clean virtual environment and rerunning `run_smoke_test.py`), and document any non-obvious pin choice with a one-line comment in the consolidated file.

**ADR-006 resolution — choose one, document why:**
- **Option A (BPF capture filter):** Restrict the sniffer's capture filter to exclude the dashboard's own port (e.g., exclude Streamlit's port from the `scapy`/`pypcap` filter expression), so dashboard traffic is never captured in the first place.
- **Option B (rate-limited discard logging):** Keep capturing all `lo0` traffic (simpler, lower risk of accidentally filtering out something relevant), but change the "skipping non-IP/discarded packet" log statement to a sampled/rate-limited logger (e.g., log at most once per N occurrences or once per second) so log volume doesn't scale with noise traffic.

Antigravity should default to **Option B** unless Option A is clearly simpler to implement correctly — a capture filter is a more invasive change to a stable, already-accepted M1 component, while rate-limited logging is additive and lower-risk. State the choice and reasoning explicitly in the deliverable notes.

**Smoke Test Orchestration:** `run_smoke_test.py` must call the existing test suites via `pytest` subprocess invocation or direct `pytest.main()` calls grouped by milestone, and print one final summary table (milestone, tests run, passed, failed, duration). It must exit non-zero if any milestone's suite fails.

---

## 7. Acceptance Criteria

- [ ] `run.py` starts the full system (core runtime + dashboard) with one command and correct startup ordering.
- [ ] `run.py` shuts down cleanly on `SIGINT`: dashboard stops first, core runtime drains and stops per its existing M1-tested shutdown path, no orphaned processes remain (verify via process listing after shutdown).
- [ ] A fresh `pip install -r requirements.txt` into a clean virtual environment succeeds with no version conflicts, and `scripts/run_smoke_test.py` passes fully against that fresh install.
- [ ] `scripts/run_smoke_test.py` runs all M1+M2+M3 existing test suites and reports one consolidated summary; exits non-zero on any failure.
- [ ] The dashboard process never requires root privileges; only the capture component does, and this is documented.
- [ ] ADR-006 is resolved (Option A or B implemented) and the resulting log/capture behavior is verified: re-run the M3 concurrent throughput test (dashboard + core runtime together) and confirm the packet-volume/log-noise ratio has measurably improved versus the original 3.4× finding.
- [ ] `docs/USAGE_GUIDE.md` exists, covers all §5 sections, and every factual claim in it is traceable to a test or documented behavior — no aspirational/unverified claims.
- [ ] No M1/M2/M3 function signature, schema, or data contract was altered.

---

## 8. Deliverables

- `run.py` (or `scripts/run_chronoguard.py`)
- Consolidated, pinned `requirements.txt`
- ADR-006 fix (in `sniffer.py` or `config.py`, per chosen option) + accompanying test
- `scripts/run_smoke_test.py`
- `docs/USAGE_GUIDE.md`
- A short written report (not a doc file — just in the response) stating: which ADR-006 option was chosen and why, any dependency-pin conflicts found and how resolved, and confirmation of the clean-virtual-environment install test — this report is what the Meta-Orchestrator will use to write the final `DECISIONS.md` entry.

---

## 9. Validation Checklist

- [ ] All Acceptance Criteria (§7) pass with evidence.
- [ ] All Deliverables (§8) exist.
- [ ] No code touches anything in Out of Scope (§3).
- [ ] `docs/USAGE_GUIDE.md` fact-checked against actual test evidence, not aspirational claims.
- [ ] ADR-006 resolution report received and ready for Meta-Orchestrator to record in `DECISIONS.md`.
- [ ] Explicit sign-off recorded — this is the final milestone; acceptance here completes ChronoGuard v1.0.

---

*End of M4 specification — final milestone. Hand this document to Antigravity as-is.*
