# ChronoGuard v1.0 — Roadmap

**Status:** Living Document
**Version:** 0.1.0

Execution follows strict one-milestone-at-a-time gating. No milestone's spec is generated until its predecessor is implemented, tested, reviewed, and explicitly accepted.

---

## Milestone Sequence

### M1 — Live Ingestion & Static ONNX Scoring
**Status:** 🟢 Complete & Accepted

Established the foundational data path: real packet capture → 5-tuple flow aggregation → static pre-trained ONNX model scoring → SQLite persistence. All Acceptance Criteria and the Validation Checklist in `specs/M1_CORE_ENGINE.md` passed; accepted by project owner.

**Exit Criteria:** ✅ Met.

---

### M2 — Drift Detection & Adaptive Retraining
**Status:** 🟢 Complete & Accepted

Wired River's ADWIN into the live engine path and built the isolated drift-triggered retraining lifecycle (Optuna tune → warm-start fit → golden holdout gate → atomic hot-swap). All Acceptance Criteria in `specs/M2_DRIFT_RETRAIN.md` §7 passed with verified evidence, including two items caught and corrected during acceptance review (a deterministic pass-path proof, and a corrected full-path latency measurement that had originally omitted the SQLite write). Accepted by project owner.

**Exit Criteria:** ✅ Met.

---

### M3 — Control Panel (Streamlit Dashboard)
**Status:** 🟢 Complete & Accepted

Built the strictly read-only Streamlit dashboard (status bar, feature-tracking chart with ADWIN bounds, live log stream, model lineage table), driven by an explicit design token system and a documented Iteration Map for cheap future visual changes. All Acceptance Criteria in `specs/M3_CONTROL_PANEL.md` §8 passed with verified evidence, including three items caught and resolved during acceptance review: a token-purity fix (hardcoded rgba replaced with a theme-derived `hex_to_rgba()` helper), a genuine sustained-throughput concurrent test (the original synthetic version was rejected and replaced with a real 6,000-packet fixture — see ADR-006 for a discovered loopback-noise risk deferred to M4), and a mechanically-verified layout check at 1440px/1024px. Accepted by project owner.

**Exit Criteria:** ✅ Met.

---

### M4 — Packaging & Deployment Hardening
**Status:** 🟡 In Progress (spec being issued next)

Final packaging: startup/shutdown lifecycle hardening, configuration management, dependency pinning, run scripts, operational documentation (including the promised feature usage guide), and a combined final smoke-run reusing M1/M2/M3's existing acceptance tests rather than new stress-test infrastructure.

**Depends on:** M1–M3 all accepted and integrated.

---

## Status Legend
- 🟡 In Progress
- 🟢 Complete & Accepted
- ⚪ Locked (blocked on prior milestone)
- 🔴 Blocked / Under Revision

---

## Change Log
- **v0.1.0** — Initial roadmap established. M1 spec issuance begins.
- **v0.2.0** — M1 accepted (all tests passed). M2 spec issuance begins.
- **v0.3.0** — M2 accepted (deterministic pass-path proof and corrected full-path latency evidence verified). M3 spec pending.
- **v0.4.0** — M3 accepted (token-purity fix, real sustained-throughput test with 6,000-packet fixture, mechanically-verified layout check all confirmed). M4 spec pending.
