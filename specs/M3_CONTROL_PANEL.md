# M3 — Control Panel (Streamlit Dashboard)

**Status:** Ready for Implementation
**Owner:** Antigravity (implementer) — reviewed by Meta-Orchestrator
**Depends on:** M1 (Accepted), M2 (Accepted)
**Blocks:** M4 (Packaging & Deployment Hardening)

---

## 1. Objective

Build a strictly read-only Streamlit dashboard that polls the core runtime's SQLite database (`flow_predictions`, `model_lineage`) every 500ms and renders: live throughput, current drift status, a live feature-tracking chart with ADWIN bounds, a live-tailing system log stream, and a model lineage table. The dashboard must never write to, lock, or otherwise interfere with the core runtime's database.

Equally important: the implementation must be **structured for cheap, targeted iteration**. A future request like "make the accent color less saturated" or "widen the log panel" must resolve to editing one token or one component file — never a multi-file hunt-and-guess.

---

## 2. Scope

Antigravity must build:

- `ui/theme.py` — the single source of truth for all design tokens (colors, type, spacing, radii). No component may hardcode a color, font, or spacing value outside this file.
- `ui/.streamlit/config.toml` — Streamlit's native theme config, generated *from* `theme.py`'s tokens (not hand-duplicated — see §6).
- `ui/components/` — one file per visual component (see §5.2 Component Inventory), each a pure function of (data) → (rendered Streamlit output). No component reads SQLite directly; all data access goes through `ui/data_access.py`.
- `ui/data_access.py` — the sole module permitted to open a (read-only) SQLite connection to the core runtime's database.
- `ui/app.py` — the page entrypoint: layout assembly, polling loop, session-state management. Contains no styling constants and no direct SQLite calls.
- `ui/copy.py` — all user-facing strings (labels, empty states, status messages) centralized, so wording changes never require touching component logic.

---

## 3. Out of Scope

Antigravity must **not**:

- Write to `flow_predictions`, `model_lineage`, or any core runtime table. `data_access.py` must open its connection in read-only mode (e.g., SQLite URI `mode=ro`) and this must be verified in tests.
- Modify `sniffer.py`, `engine.py`, `optimizer.py`, `drift_monitor.py`, `model_registry.py`, `golden_holdout.py`, or any M1/M2 file.
- Implement authentication, multi-user sessions, or remote access controls.
- Add animation/motion beyond what's explicitly specified in §5.4 (Signature Element) and §5.5 (Motion Budget) — no decorative transitions, no page-load sequences, no hover effects beyond simple state changes (e.g., row hover highlight).
- Introduce a second UI framework, CSS framework, or component library. Streamlit-native rendering plus the token-driven CSS in `theme.py` is the entire visual stack.
- Modify `docs/SPECIFICATION.md`, `docs/ROADMAP.md`, `docs/TEST_PLAN.md`, or `docs/DECISIONS.md`.

---

## 4. Architecture Context

```
[M1/M2] SQLite (WAL) ◄── read-only ──┐
                                       │
                          ui/data_access.py (polls every 500ms)
                                       │
                          ui/app.py (session state, layout)
                                       │
              ┌────────────┬──────────┴──────────┬────────────┐
              ▼            ▼                      ▼            ▼
        status_bar.py  feature_chart.py     log_stream.py  lineage_table.py
              │            │                      │            │
              └────────────┴──── ui/theme.py ──────┴────────────┘
                         (tokens: color, type, spacing)
```

`theme.py` is a dependency of every component but a dependent of none — this is what makes tokens the single lever for visual changes.

---

## 5. Design System

### 5.1 Grounding & Direction

**Subject:** a SOC/NOC-style live security monitoring console. **Audience:** an engineer (or interviewer) assessing operational judgment, not a marketing page. **The page's single job:** let someone glance at it and know, within two seconds, whether the system is healthy, and let them drill into *why* if it isn't.

This grounds every choice below: dark, high-contrast surfaces are the deliberate convention for monitoring consoles (extended viewing comfort, status colors read clearly against a neutral base) — not a generic "AI dashboard" default. Monospace typography for data (IPs, ports, hashes, timestamps, log lines) is a functional choice because that's the register network engineers actually read in — not a stylistic flourish.

### 5.2 Token System (`ui/theme.py` — authoritative)

**Color — Surfaces & Text (neutrals):**

| Token | Hex | Usage |
|---|---|---|
| `surface.base` | `#0B0F17` | Page background — deep slate, not pure black (avoids OLED halation/eye fatigue) |
| `surface.raised` | `#161F30` | Panel/card backgrounds — elevated blue-slate |
| `surface.border` | `#2A364F` | Hairline dividers, panel borders |
| `text.primary` | `#F1F5F9` | Headings, primary values — off-white, not pure white (avoids glare) |
| `text.muted` | `#94A3B8` | Labels, secondary text, timestamps, axis labels |

**Color — Semantic status (functional, not decorative — reserved exclusively for system state):**

| Token | Hex | Meaning | Usage constraint |
|---|---|---|---|
| `status.signal` | `#06B6D4` | System active/live (heartbeat, connection state) | Distinct from `status.ok` — signals *liveness*, not a judgment outcome |
| `status.ok` | `#10B981` | PASS / benign judgment outcome | Lineage PASS rows, benign classification badges only |
| `status.warn` | `#F59E0B` | Drift detected, retrain in progress | Never used decoratively |
| `status.critical` | `#EF4444` | FAIL, attack classification, error | Reserved strictly for genuine alerts — never for emphasis |
| `data.line` | `#3B82F6` | Generic/non-status chart data lines | Used for secondary chart series; the signature feature-tracking line uses `status.signal` |

**Anti-patterns (explicitly avoided, with rationale):** saturated neon (`#00FF00`-style greens) — fails contrast accessibility and reads as noise, not signal; sweeping/spinning radar animations — reads as mockup theater, not production tooling; pure black canvas — flattens depth and harshens contrast against data points; unstable/shifting layouts when new rows or log lines arrive (cumulative layout shift) — all live-updating containers (log stream, lineage table) must have explicit fixed height with internal `overflow-y: auto` scrolling, never height that grows with content.

**Typography:**

| Role | Face | Usage |
|---|---|---|
| Display | Inter (600/700 weight) | Panel titles, top-level status value (e.g. throughput number) |
| Body | Inter (400/500 weight) | Labels, descriptions, table headers |
| Data/Mono | JetBrains Mono (400/500 weight) | IPs, ports, protocol, hashes, log lines, timestamps, all numeric metrics |

Type scale (rem, base 16px): `display-lg: 2.25`, `display-md: 1.5`, `body: 0.9375`, `caption: 0.8125`, `mono-data: 0.875`. Line-height 1.4 for body/mono, 1.2 for display. No other sizes are introduced without a token addition.

**Spacing & Shape:**

- Spacing scale (rem): `xs:0.25, sm:0.5, md:1, lg:1.5, xl:2.5` — all component padding/gaps reference this scale, no arbitrary pixel values.
- Radius: `sm:4px` (badges, inline elements), `md:8px` (panels/cards). No fully-rounded (`pill`) elements — sharp-ish, technical, console-like, not consumer-app soft.
- Border weight: 1px hairline (`surface.border`) for all panel divisions. No shadows except a single subtle `0 1px 2px rgba(0,0,0,0.3)` on raised panels — flat, not skeuomorphic.

### 5.3 Layout

Single-page, three-zone grid, no scrolling required at 1440×900 (must remain usable down to 1024px width; below that, stack vertically):

```
┌──────────────────────────────────────────────────────────────────┐
│ STATUS BAR   Throughput | Drift Status | Active Model Version     │
├────────────────────────────────────┬───────────────────────────┤
│ FEATURE TRACKING + ADWIN BOUNDS     │ SYSTEM LOG STREAM          │
│ (signature element — see 5.4)       │ (live-tailing, monospace) │
├────────────────────────────────────┴───────────────────────────┤
│ MODEL LINEAGE TABLE (retrain history, PASS/FAIL, accuracy)       │
└──────────────────────────────────────────────────────────────────┘
```

### 5.4 Signature Element

The feature-tracking chart is the one place to spend visual boldness (per design-lead practice: one signature moment, everything else quiet). Treatment: the live metric line in `status.signal`, the ADWIN bound rendered as a dashed corridor (`text.muted`, 40% opacity fill between bounds), and — the specific signature moment — a drift-fire event renders as a crisp solid vertical rule in `status.warn` with a small monospace timestamp label at its foot, evoking an oscilloscope/seismograph trace rather than a generic dashboard line chart. This is the only place gradients or fills are used anywhere in the UI.

### 5.5 Motion — Three Named Patterns Only

Motion is restrained but present, and every instance must serve status indication, never decoration. Exactly three patterns are permitted; no others:

**Pattern A — Live Beacon.** An 8px circular dot beside the connection/status indicator, pulsing opacity `1.0 → 0.3 → 1.0` over a 2.5s ease-in-out loop, color `status.signal`. Confirms the data feed is actively running without shifting any layout.

```css
@keyframes status-pulse {
  0%, 100% { opacity: 1.0; transform: scale(1); }
  50% { opacity: 0.3; transform: scale(0.95); }
}
.live-beacon { animation: status-pulse 2.5s infinite ease-in-out; }
```

**Pattern B — Chart Transitions.** Feature-tracking chart updates interpolate rather than snap: Plotly `transition={'duration': 400, 'easing': 'cubic-in-out'}`. Smooths tick-to-tick updates so the eye tracks continuous motion instead of re-scanning a redrawn canvas.

**Pattern C — Flash-to-Rest Row Entry.** When a new row enters the log stream or lineage table, its border briefly glows in the relevant semantic color (`status.critical` for FAIL/alert rows, `status.ok` for PASS rows) via `box-shadow: 0 0 8px <color at 40% alpha>`, fading to normal panel styling over 1.2s. Draws the eye to new events without a modal/popup interruption.

**Layout stability requirement:** both the log stream and lineage table containers must have a fixed height with internal `overflow-y: auto` — new rows must never shift surrounding layout (cumulative layout shift is treated as a defect, not a style nitpick).

`prefers-reduced-motion` must disable all three patterns (beacon holds at full opacity, chart updates snap instantly, row entry has no glow — just appears).

---

## 6. Data Contracts

### 6.1 `data_access.py` — Read Functions

```python
def get_connection(db_path: str) -> sqlite3.Connection:
    """Opens a READ-ONLY connection: sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)"""

def get_throughput(conn, window_sec: int = 10) -> float:
    """Flows/sec over the trailing window_sec, from flow_predictions.inference_ts."""

def get_drift_status(conn) -> dict:
    """Returns {'state': 'HEALTHY'|'DRIFT_DETECTED'|'RETRAINING', 'since_ts': float, 'triggering_feature': str|None}"""

def get_active_model_version(conn) -> dict:
    """Returns latest PASS model_lineage row's model_version + completed_ts, or the M1 baseline version if none."""

def get_feature_series(conn, feature: str, limit: int = 200) -> "pandas.DataFrame":
    """Trailing N values of a monitored feature/score plus timestamps, for the chart."""

def get_recent_lineage(conn, limit: int = 50) -> "pandas.DataFrame":
    """Most recent model_lineage rows, newest first."""

def get_log_tail(conn_or_path, limit: int = 100) -> list[str]:
    """Most recent structured log lines (see §6.2 for source)."""
```

### 6.2 Log Stream Source

Core runtime logging (M1/M2) must already write structured log lines to a file (per M1 §6 logging requirements). `get_log_tail` reads this file's tail — it does **not** invent a new logging channel. If the M1/M2 logging setup does not currently write to a file (only stdout), Antigravity must add a `logging.FileHandler` to `config.py`'s logging setup as a minimal, additive (non-breaking) change — flag this explicitly if required, since it technically touches an M1/M2 file; this is the one narrow permitted exception to §3's file-modification restriction, limited strictly to adding a file handler to existing logging config.

### 6.3 Component Function Signatures

```python
# ui/components/status_bar.py
def render_status_bar(throughput: float, drift: dict, model: dict) -> None: ...

# ui/components/feature_chart.py
def render_feature_chart(series_df: "pandas.DataFrame", drift_events: list[float]) -> None: ...

# ui/components/log_stream.py
def render_log_stream(lines: list[str]) -> None: ...

# ui/components/lineage_table.py
def render_lineage_table(lineage_df: "pandas.DataFrame") -> None: ...
```

Every component receives already-fetched data as arguments — never a database handle. This is what keeps components swappable/restylable in isolation.

---

## 7. Implementation Directives

**Polling Model:** Use `st.fragment` (or the version-appropriate Streamlit auto-refresh mechanism) scoped to the smallest re-rendering unit possible — the whole page must not fully rerun every 500ms, only the data-bound fragments, to avoid flicker and preserve scroll position.

**Chart Rendering:** Use Plotly (already available) with incremental data updates (`extendData`-style patterns or full-figure replace only at the configured poll interval, not more often) — avoid rebuilding the entire figure object from scratch if the underlying library supports partial updates.

**Read-Only Enforcement:** `get_connection` must use the `mode=ro` URI pattern; a test must assert that an `INSERT` attempt through this connection raises `sqlite3.OperationalError`.

**Empty/Loading States:** Per copy guidance — no blank panels. Each component must handle the zero-data case (system just started, no flows yet) with a plain-language, interface-voiced message sourced from `ui/copy.py` (e.g., "No flows recorded yet." not "No data available."), not a spinner-forever or blank space.

**Theme Sync:** `.streamlit/config.toml` must be generated or kept in lockstep with `theme.py` — document in code comments which `theme.py` tokens map to which `config.toml` keys (`primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`, `font`), so a token change and a config.toml change never drift apart. A small check script or test asserting this sync is encouraged but not required for M3 acceptance.

---

## 8. Acceptance Criteria

- [ ] Dashboard renders all four zones (status bar, feature chart, log stream, lineage table) correctly against a live M1+M2 runtime instance.
- [ ] No write occurs to the core runtime database at any point — verified by the read-only-connection test in §7.
- [ ] Dashboard polling does not measurably degrade core runtime inference throughput (re-run the M1 throughput acceptance test — §7 of `specs/M1_CORE_ENGINE.md` — with the dashboard running concurrently; result must remain within the original criterion).
- [ ] A live drift-fire event (triggered via the M2 test fixture) visibly appears on the feature chart as the specified vertical-rule signature element within one poll cycle (≤500ms) of the SQLite write.
- [ ] A model hot-swap (triggered via M2 retrain cycle) updates the status bar's active model version within one poll cycle.
- [ ] Zero hardcoded color, font-family, or arbitrary spacing/pixel values exist outside `ui/theme.py` (verify via a grep-based check across `ui/components/*.py` and `ui/app.py` for hex codes / font-family strings).
- [ ] Layout remains usable (no overlap, no horizontal scroll) at both 1440px and 1024px viewport widths.
- [ ] Empty-state messaging (zero flows recorded) renders correctly and matches `ui/copy.py`, not an inline string.
- [ ] `prefers-reduced-motion` disables all three motion patterns (§5.5): beacon, chart transitions, flash-to-rest row entry.
- [ ] New rows in the log stream or lineage table never shift surrounding layout (fixed-height + internal scroll verified — zero cumulative layout shift).

---

## 9. Deliverables

- `ui/theme.py`, `ui/.streamlit/config.toml`, `ui/data_access.py`, `ui/app.py`, `ui/copy.py`
- `ui/components/status_bar.py`, `feature_chart.py`, `log_stream.py`, `lineage_table.py`
- Updated `config.py`/`requirements.txt` if a `FileHandler` addition or new dependency (e.g., `plotly`) is required — flagged explicitly per §6.2.
- Unit tests: read-only connection enforcement, empty-state rendering, each `data_access.py` function against a fixture DB.
- Integration test: full dashboard render against a live M1+M2 fixture run, including a drift-fire-to-chart-update assertion.
- `README_M3.md`: how to launch the dashboard alongside the core runtime, how the token system works, and — critically — the **Iteration Map** below, reproduced verbatim in the README for future reference.

### Iteration Map (reproduce in `README_M3.md`)

| Future request | File/token to change |
|---|---|
| "Change the accent color" | `ui/theme.py` → `status.signal` |
| "Make alerts a different red" | `ui/theme.py` → `status.critical` |
| "Change the chart's secondary line color" | `ui/theme.py` → `data.line` |
| "Use a different font" | `ui/theme.py` → typography section, then re-sync `.streamlit/config.toml` |
| "Widen/narrow the log panel" | `ui/app.py` layout column ratios only |
| "Add a new metric to the status bar" | `ui/components/status_bar.py` + extend `data_access.get_*` if new data needed |
| "Change wording of any label/message" | `ui/copy.py` only — never touch component logic |
| "Adjust chart appearance" | `ui/components/feature_chart.py`, sourcing colors from `theme.py` |
| "Change spacing/density" | `ui/theme.py` spacing scale |
| "Add/remove a panel" | `ui/app.py` layout + new/removed file in `ui/components/` |
| "Make the live beacon pulse faster/slower" | `ui/theme.py` or `status_bar.py` → Pattern A keyframe duration |
| "Add/remove/adjust an animation" | §5.5 Motion — must stay within the three named patterns or be proposed as a spec amendment, not ad-hoc added |

---

## 10. Validation Checklist

- [ ] All Acceptance Criteria (§8) pass with evidence.
- [ ] All Deliverables (§9) exist and match Data Contracts (§6) exactly.
- [ ] No code touches anything in Out of Scope (§3), including the one narrow logging exception being explicitly flagged if used.
- [ ] Design System (§5) tokens are the sole source of visual constants — verified via the grep check in §8.
- [ ] Iteration Map is present in `README_M3.md`, accurate to the actual file structure delivered.
- [ ] Explicit sign-off recorded before M4 spec is generated.

---

*End of M3 specification. Hand this document to Antigravity as-is. Do not proceed to M4 until this milestone is implemented, tested, reviewed, and accepted.*
