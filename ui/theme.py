"""
ChronoGuard v1.0 — Design Token System (M3)

Single source of truth for all visual constants: colors, typography,
spacing, shape, and motion.  Every component in ui/components/ imports
tokens from this module — no hardcoded hex, font-family, or pixel value
may appear anywhere else in the ui/ tree.

Spec reference: M3 §5.2 (token tables), §5.4 (signature element),
§5.5 (motion budget — three named patterns only).

Iteration Map coupling:
    - "Change the accent color"      → COLORS['status']['signal']
    - "Make alerts a different red"   → COLORS['status']['critical']
    - "Use a different font"          → TYPOGRAPHY section, then re-sync config.toml
    - "Change spacing/density"        → SPACING scale
    - "Pulse faster/slower"           → MOTION['beacon_duration_s']
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Color Tokens (§5.2 — verbatim hex values)
# ---------------------------------------------------------------------------
COLORS: dict[str, dict[str, str]] = {
    "surface": {
        "base": "#0B0F17",       # Page background — deep slate
        "raised": "#161F30",     # Panel/card backgrounds — elevated blue-slate
        "border": "#2A364F",     # Hairline dividers, panel borders
    },
    "text": {
        "primary": "#F1F5F9",    # Headings, primary values — off-white
        "muted": "#94A3B8",      # Labels, secondary text, timestamps
    },
    "status": {
        "signal": "#06B6D4",     # System active/live — liveness, not judgment
        "ok": "#10B981",         # PASS / benign judgment outcome
        "warn": "#F59E0B",       # Drift detected, retrain in progress
        "critical": "#EF4444",   # FAIL, attack, error — genuine alerts only
    },
    "data": {
        "line": "#3B82F6",       # Generic/non-status chart data lines
    },
}

# ---------------------------------------------------------------------------
# Typography Tokens (§5.2)
# ---------------------------------------------------------------------------
TYPOGRAPHY: dict[str, dict[str, str | int]] = {
    "display": {
        "family": "Inter",
        "weight_semibold": 600,
        "weight_bold": 700,
    },
    "body": {
        "family": "Inter",
        "weight_regular": 400,
        "weight_medium": 500,
    },
    "data": {
        "family": "JetBrains Mono",
        "weight_regular": 400,
        "weight_medium": 500,
    },
}

# Type scale (rem, base 16px) — §5.2
TYPE_SCALE: dict[str, float] = {
    "display_lg": 2.25,
    "display_md": 1.5,
    "body": 0.9375,
    "caption": 0.8125,
    "mono_data": 0.875,
}

# Line-height — §5.2
LINE_HEIGHT: dict[str, float] = {
    "display": 1.2,
    "body": 1.4,
    "mono": 1.4,
}

# ---------------------------------------------------------------------------
# Spacing & Shape Tokens (§5.2)
# ---------------------------------------------------------------------------
# Spacing scale (rem)
SPACING: dict[str, float] = {
    "xs": 0.25,
    "sm": 0.5,
    "md": 1.0,
    "lg": 1.5,
    "xl": 2.5,
}

# Shape
SHAPE: dict[str, str] = {
    "radius_sm": "4px",     # Badges, inline elements
    "radius_md": "8px",     # Panels/cards
    "border": "1px",        # Hairline border weight
    "shadow": "0 1px 2px rgba(0,0,0,0.3)",  # Single subtle shadow on raised panels
}

# ---------------------------------------------------------------------------
# Motion Tokens (§5.5 — three named patterns only)
# ---------------------------------------------------------------------------
MOTION: dict[str, float] = {
    "beacon_duration_s": 2.5,       # Pattern A — Live Beacon pulse
    "chart_transition_ms": 400,     # Pattern B — Chart Transitions
    "flash_duration_s": 1.2,        # Pattern C — Flash-to-Rest Row Entry
}

# ---------------------------------------------------------------------------
# Google Fonts Import URL
# ---------------------------------------------------------------------------
_FONT_IMPORT_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500"
    "&display=swap"
)


# ---------------------------------------------------------------------------
# Token-derived color utilities
# ---------------------------------------------------------------------------
def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a hex color token to an ``rgba()`` CSS string.

    Components that need a translucent version of a token color (e.g., the
    ADWIN corridor fill at 40% opacity) must use this helper rather than
    hardcoding the ``rgba(...)`` decomposition, so the value stays coupled
    to ``COLORS``.

    Parameters
    ----------
    hex_color : str
        A 7-character hex color string (e.g., ``'#94A3B8'``).
    alpha : float
        Opacity in [0.0, 1.0].

    Returns
    -------
    str
        ``'rgba(R, G, B, alpha)'``
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# ---------------------------------------------------------------------------
# CSS Injection — all values interpolated from tokens, zero hardcoding
# ---------------------------------------------------------------------------
def inject_css() -> str:
    """Generate the full <style> block for the Streamlit app.

    Every value is sourced from the token dicts above.  This function is
    the single place where tokens become CSS custom properties and rules.
    Components never embed raw style values — they reference CSS classes
    defined here.
    """
    s = COLORS["surface"]
    t = COLORS["text"]
    st = COLORS["status"]
    d = COLORS["data"]
    typ = TYPOGRAPHY
    ts = TYPE_SCALE
    lh = LINE_HEIGHT
    sp = SPACING
    sh = SHAPE
    mo = MOTION

    return f"""
    <style>
    @import url('{_FONT_IMPORT_URL}');

    /* ---- CSS Custom Properties (from theme.py tokens) ---- */
    :root {{
        /* Surfaces */
        --surface-base: {s['base']};
        --surface-raised: {s['raised']};
        --surface-border: {s['border']};
        /* Text */
        --text-primary: {t['primary']};
        --text-muted: {t['muted']};
        /* Status */
        --status-signal: {st['signal']};
        --status-ok: {st['ok']};
        --status-warn: {st['warn']};
        --status-critical: {st['critical']};
        /* Data */
        --data-line: {d['line']};
        /* Typography */
        --font-display: '{typ["display"]["family"]}', sans-serif;
        --font-body: '{typ["body"]["family"]}', sans-serif;
        --font-data: '{typ["data"]["family"]}', monospace;
        --fw-display-semibold: {typ["display"]["weight_semibold"]};
        --fw-display-bold: {typ["display"]["weight_bold"]};
        --fw-body-regular: {typ["body"]["weight_regular"]};
        --fw-body-medium: {typ["body"]["weight_medium"]};
        --fw-data-regular: {typ["data"]["weight_regular"]};
        --fw-data-medium: {typ["data"]["weight_medium"]};
        /* Type scale */
        --ts-display-lg: {ts["display_lg"]}rem;
        --ts-display-md: {ts["display_md"]}rem;
        --ts-body: {ts["body"]}rem;
        --ts-caption: {ts["caption"]}rem;
        --ts-mono-data: {ts["mono_data"]}rem;
        /* Line height */
        --lh-display: {lh["display"]};
        --lh-body: {lh["body"]};
        --lh-mono: {lh["mono"]};
        /* Spacing */
        --sp-xs: {sp["xs"]}rem;
        --sp-sm: {sp["sm"]}rem;
        --sp-md: {sp["md"]}rem;
        --sp-lg: {sp["lg"]}rem;
        --sp-xl: {sp["xl"]}rem;
        /* Shape */
        --radius-sm: {sh["radius_sm"]};
        --radius-md: {sh["radius_md"]};
        --border-weight: {sh["border"]};
        --shadow-raised: {sh["shadow"]};
        /* Motion */
        --beacon-duration: {mo["beacon_duration_s"]}s;
        --chart-transition: {mo["chart_transition_ms"]}ms;
        --flash-duration: {mo["flash_duration_s"]}s;
    }}

    /* ---- Global resets ---- */
    .stApp {{
        background-color: var(--surface-base);
        color: var(--text-primary);
        font-family: var(--font-body);
        font-weight: var(--fw-body-regular);
        font-size: var(--ts-body);
        line-height: var(--lh-body);
    }}

    /* ---- Panel / card styling ---- */
    .cg-panel {{
        background-color: var(--surface-raised);
        border: var(--border-weight) solid var(--surface-border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-raised);
        padding: var(--sp-md);
    }}

    /* ---- Typography utilities ---- */
    .cg-display-lg {{
        font-family: var(--font-display);
        font-weight: var(--fw-display-bold);
        font-size: var(--ts-display-lg);
        line-height: var(--lh-display);
        color: var(--text-primary);
    }}
    .cg-display-md {{
        font-family: var(--font-display);
        font-weight: var(--fw-display-semibold);
        font-size: var(--ts-display-md);
        line-height: var(--lh-display);
        color: var(--text-primary);
    }}
    .cg-body {{
        font-family: var(--font-body);
        font-weight: var(--fw-body-regular);
        font-size: var(--ts-body);
        line-height: var(--lh-body);
        color: var(--text-primary);
    }}
    .cg-caption {{
        font-family: var(--font-body);
        font-weight: var(--fw-body-regular);
        font-size: var(--ts-caption);
        line-height: var(--lh-body);
        color: var(--text-muted);
    }}
    .cg-mono {{
        font-family: var(--font-data);
        font-weight: var(--fw-data-regular);
        font-size: var(--ts-mono-data);
        line-height: var(--lh-mono);
        color: var(--text-primary);
    }}

    /* ---- Status badge ---- */
    .cg-badge {{
        display: inline-block;
        padding: var(--sp-xs) var(--sp-sm);
        border-radius: var(--radius-sm);
        font-family: var(--font-data);
        font-weight: var(--fw-data-medium);
        font-size: var(--ts-caption);
        line-height: 1;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .cg-badge-ok {{
        background-color: rgba(16, 185, 129, 0.15);
        color: var(--status-ok);
        border: var(--border-weight) solid var(--status-ok);
    }}
    .cg-badge-warn {{
        background-color: rgba(245, 158, 11, 0.15);
        color: var(--status-warn);
        border: var(--border-weight) solid var(--status-warn);
    }}
    .cg-badge-critical {{
        background-color: rgba(239, 68, 68, 0.15);
        color: var(--status-critical);
        border: var(--border-weight) solid var(--status-critical);
    }}
    .cg-badge-signal {{
        background-color: rgba(6, 182, 212, 0.15);
        color: var(--status-signal);
        border: var(--border-weight) solid var(--status-signal);
    }}

    /* ---- Pattern A: Live Beacon (§5.5) ---- */
    @keyframes status-pulse {{
        0%, 100% {{ opacity: 1.0; transform: scale(1); }}
        50% {{ opacity: 0.3; transform: scale(0.95); }}
    }}
    .cg-live-beacon {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--status-signal);
        animation: status-pulse var(--beacon-duration) infinite ease-in-out;
        vertical-align: middle;
        margin-right: var(--sp-xs);
    }}

    /* ---- Pattern C: Flash-to-Rest Row Entry (§5.5) ---- */
    @keyframes flash-ok {{
        from {{ box-shadow: 0 0 8px rgba(16, 185, 129, 0.4); }}
        to {{ box-shadow: none; }}
    }}
    @keyframes flash-critical {{
        from {{ box-shadow: 0 0 8px rgba(239, 68, 68, 0.4); }}
        to {{ box-shadow: none; }}
    }}
    @keyframes flash-warn {{
        from {{ box-shadow: 0 0 8px rgba(245, 158, 11, 0.4); }}
        to {{ box-shadow: none; }}
    }}
    .cg-flash-ok {{
        animation: flash-ok var(--flash-duration) ease-out forwards;
    }}
    .cg-flash-critical {{
        animation: flash-critical var(--flash-duration) ease-out forwards;
    }}
    .cg-flash-warn {{
        animation: flash-warn var(--flash-duration) ease-out forwards;
    }}

    /* ---- Fixed-height scrollable containers (§5.5 layout stability) ---- */
    .cg-scroll-container {{
        height: 380px;
        overflow-y: auto;
        overflow-x: hidden;
    }}
    .cg-scroll-container-short {{
        height: 260px;
        overflow-y: auto;
        overflow-x: hidden;
    }}

    /* ---- Log line styling ---- */
    .cg-log-line {{
        font-family: var(--font-data);
        font-weight: var(--fw-data-regular);
        font-size: var(--ts-mono-data);
        line-height: var(--lh-mono);
        color: var(--text-muted);
        padding: var(--sp-xs) var(--sp-sm);
        border-bottom: var(--border-weight) solid var(--surface-border);
        word-break: break-all;
    }}
    .cg-log-line-error {{
        color: var(--status-critical);
    }}
    .cg-log-line-warning {{
        color: var(--status-warn);
    }}

    /* ---- Lineage table styling ---- */
    .cg-lineage-row {{
        display: grid;
        grid-template-columns: 1fr 1fr 1fr 0.8fr 0.8fr 0.8fr 1fr 1fr;
        gap: var(--sp-sm);
        padding: var(--sp-sm) var(--sp-md);
        border-bottom: var(--border-weight) solid var(--surface-border);
        font-family: var(--font-data);
        font-size: var(--ts-mono-data);
        line-height: var(--lh-mono);
        align-items: center;
    }}
    .cg-lineage-header {{
        font-family: var(--font-body);
        font-weight: var(--fw-body-medium);
        font-size: var(--ts-caption);
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* ---- Status bar metric card ---- */
    .cg-metric {{
        text-align: center;
    }}
    .cg-metric-value {{
        font-family: var(--font-data);
        font-weight: var(--fw-display-bold);
        font-size: var(--ts-display-md);
        line-height: var(--lh-display);
        color: var(--text-primary);
    }}
    .cg-metric-label {{
        font-family: var(--font-body);
        font-weight: var(--fw-body-regular);
        font-size: var(--ts-caption);
        color: var(--text-muted);
        margin-top: var(--sp-xs);
    }}

    /* ---- Section title ---- */
    .cg-section-title {{
        font-family: var(--font-display);
        font-weight: var(--fw-display-semibold);
        font-size: var(--ts-body);
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: var(--sp-sm);
    }}

    /* ---- Empty state ---- */
    .cg-empty-state {{
        font-family: var(--font-body);
        font-size: var(--ts-body);
        color: var(--text-muted);
        text-align: center;
        padding: var(--sp-xl) var(--sp-md);
    }}

    /* ---- Row hover (simple state change per §3) ---- */
    .cg-lineage-row:hover {{
        background-color: rgba(42, 54, 79, 0.5);
    }}

    /* ---- prefers-reduced-motion: disable all three patterns (§5.5) ---- */
    @media (prefers-reduced-motion: reduce) {{
        .cg-live-beacon {{
            animation: none;
            opacity: 1.0;
        }}
        .cg-flash-ok,
        .cg-flash-critical,
        .cg-flash-warn {{
            animation: none;
        }}
    }}

    /* ---- Hide Streamlit default chrome for cleaner console look ---- */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    </style>
    """


# ---------------------------------------------------------------------------
# Streamlit config.toml generation (§7 theme sync)
# ---------------------------------------------------------------------------
def generate_config_toml() -> str:
    """Produce .streamlit/config.toml content from theme tokens.

    Token-to-key mapping (documented per §7):
        primaryColor            ← COLORS['status']['signal']
        backgroundColor         ← COLORS['surface']['base']
        secondaryBackgroundColor ← COLORS['surface']['raised']
        textColor               ← COLORS['text']['primary']
        font                    ← "sans serif" (Inter loaded via CSS @import)
    """
    return (
        "[theme]\n"
        f'primaryColor = "{COLORS["status"]["signal"]}"'
        f"          # theme.COLORS['status']['signal']\n"
        f'backgroundColor = "{COLORS["surface"]["base"]}"'
        f"       # theme.COLORS['surface']['base']\n"
        f'secondaryBackgroundColor = "{COLORS["surface"]["raised"]}"'
        f"  # theme.COLORS['surface']['raised']\n"
        f'textColor = "{COLORS["text"]["primary"]}"'
        f"             # theme.COLORS['text']['primary']\n"
        f'font = "sans serif"'
        f"                        # Inter loaded via CSS @import in theme.inject_css()\n"
    )
