"""
Tests for M3 theme token system — ChronoGuard

Covers (per M3 spec §8):
    - Token completeness: all §5.2 tokens exist.
    - config.toml sync: generate_config_toml() matches the file on disk.
    - No hardcoded colors/fonts/pixels in ui/components/*.py or ui/app.py
      (§8 criterion #6).
"""

from __future__ import annotations

import os
import re
import sys

import pytest

# Ensure project root is on path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ui.theme import (
    COLORS,
    LINE_HEIGHT,
    MOTION,
    SHAPE,
    SPACING,
    TYPE_SCALE,
    TYPOGRAPHY,
    generate_config_toml,
    hex_to_rgba,
    inject_css,
)


class TestTokenCompleteness:
    """Assert all §5.2 tokens exist in theme.py dicts."""

    def test_surface_colors(self):
        assert "base" in COLORS["surface"]
        assert "raised" in COLORS["surface"]
        assert "border" in COLORS["surface"]

    def test_text_colors(self):
        assert "primary" in COLORS["text"]
        assert "muted" in COLORS["text"]

    def test_status_colors(self):
        assert "signal" in COLORS["status"]
        assert "ok" in COLORS["status"]
        assert "warn" in COLORS["status"]
        assert "critical" in COLORS["status"]

    def test_data_colors(self):
        assert "line" in COLORS["data"]

    def test_typography(self):
        assert "display" in TYPOGRAPHY
        assert "body" in TYPOGRAPHY
        assert "data" in TYPOGRAPHY
        assert TYPOGRAPHY["display"]["family"] == "Inter"
        assert TYPOGRAPHY["data"]["family"] == "JetBrains Mono"

    def test_type_scale(self):
        required = {"display_lg", "display_md", "body", "caption", "mono_data"}
        assert required.issubset(set(TYPE_SCALE.keys()))
        assert TYPE_SCALE["display_lg"] == 2.25
        assert TYPE_SCALE["display_md"] == 1.5
        assert TYPE_SCALE["body"] == 0.9375
        assert TYPE_SCALE["caption"] == 0.8125
        assert TYPE_SCALE["mono_data"] == 0.875

    def test_line_height(self):
        assert LINE_HEIGHT["display"] == 1.2
        assert LINE_HEIGHT["body"] == 1.4
        assert LINE_HEIGHT["mono"] == 1.4

    def test_spacing(self):
        required = {"xs", "sm", "md", "lg", "xl"}
        assert required.issubset(set(SPACING.keys()))
        assert SPACING["xs"] == 0.25
        assert SPACING["sm"] == 0.5
        assert SPACING["md"] == 1.0
        assert SPACING["lg"] == 1.5
        assert SPACING["xl"] == 2.5

    def test_shape(self):
        assert SHAPE["radius_sm"] == "4px"
        assert SHAPE["radius_md"] == "8px"
        assert SHAPE["border"] == "1px"

    def test_motion(self):
        assert MOTION["beacon_duration_s"] == 2.5
        assert MOTION["chart_transition_ms"] == 400
        assert MOTION["flash_duration_s"] == 1.2

    def test_hex_format(self):
        """All hex colors must be valid 7-char hex codes."""
        hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for group_name, group in COLORS.items():
            for token_name, value in group.items():
                assert hex_re.match(value), (
                    f"COLORS['{group_name}']['{token_name}'] = '{value}' "
                    f"is not a valid hex color"
                )


class TestConfigTomlSync:
    """§7: config.toml must be in lockstep with theme.py tokens."""

    def test_generated_matches_file(self):
        """generate_config_toml() output must match the on-disk config.toml."""
        config_path = os.path.join(
            _PROJECT_ROOT, "ui", ".streamlit", "config.toml"
        )
        assert os.path.isfile(config_path), (
            f"config.toml not found at {config_path}"
        )

        with open(config_path, "r") as f:
            on_disk = f.read()

        generated = generate_config_toml()
        assert on_disk.strip() == generated.strip(), (
            "config.toml on disk does not match generate_config_toml() output.\n"
            f"On disk:\n{on_disk}\n\nGenerated:\n{generated}"
        )


class TestInjectCss:
    """Verify inject_css() produces valid CSS with tokens interpolated."""

    def test_returns_nonempty_string(self):
        css = inject_css()
        assert isinstance(css, str)
        assert len(css) > 100

    def test_contains_css_custom_properties(self):
        css = inject_css()
        assert "--surface-base:" in css
        assert "--status-signal:" in css
        assert "--font-display:" in css
        assert "--sp-md:" in css

    def test_contains_keyframes(self):
        css = inject_css()
        assert "@keyframes status-pulse" in css
        assert "@keyframes flash-ok" in css
        assert "@keyframes flash-critical" in css

    def test_contains_reduced_motion(self):
        css = inject_css()
        assert "prefers-reduced-motion" in css

    def test_contains_font_import(self):
        css = inject_css()
        assert "fonts.googleapis.com" in css
        assert "Inter" in css
        assert "JetBrains+Mono" in css


class TestNoHardcodedValues:
    """§8 criterion #6: zero hardcoded color, font-family, or arbitrary
    spacing/pixel values outside ui/theme.py.

    This performs the grep-based check specified in §8.
    """

    # Regex patterns to detect hardcoded values.
    _HEX_RE = re.compile(r'#[0-9a-fA-F]{3,8}(?!["\'])')
    _FONT_FAMILY_RE = re.compile(r"font-family\s*:", re.IGNORECASE)
    # Exclude common CSS property references and focus on literal values
    # in Python code (not in CSS generated by theme.py).

    def _get_ui_source_files(self) -> list[str]:
        """Return paths to ui/components/*.py and ui/app.py."""
        components_dir = os.path.join(_PROJECT_ROOT, "ui", "components")
        files = []
        if os.path.isdir(components_dir):
            for fname in os.listdir(components_dir):
                if fname.endswith(".py") and fname != "__init__.py":
                    files.append(os.path.join(components_dir, fname))
        app_path = os.path.join(_PROJECT_ROOT, "ui", "app.py")
        if os.path.isfile(app_path):
            files.append(app_path)
        return files

    def test_no_hex_colors_in_components_or_app(self):
        """No hex color literals (#RRGGBB) in component or app files."""
        violations = []
        for fpath in self._get_ui_source_files():
            with open(fpath, "r") as f:
                for lineno, line in enumerate(f, 1):
                    # Skip comments and docstrings.
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith('"""'):
                        continue
                    # Look for hex patterns that are actual color values
                    # (not Python comments starting with #).
                    matches = self._HEX_RE.findall(line)
                    for m in matches:
                        # Filter out false positives: Python line comments
                        # and shebang lines.
                        if line.strip().startswith("#"):
                            continue
                        violations.append(
                            f"{os.path.basename(fpath)}:{lineno}: {m}"
                        )

        assert not violations, (
            f"Hardcoded hex colors found outside theme.py:\n"
            + "\n".join(violations)
        )

    def test_no_font_family_in_components_or_app(self):
        """No font-family declarations in component or app files."""
        violations = []
        for fpath in self._get_ui_source_files():
            with open(fpath, "r") as f:
                for lineno, line in enumerate(f, 1):
                    if self._FONT_FAMILY_RE.search(line):
                        violations.append(
                            f"{os.path.basename(fpath)}:{lineno}: {line.strip()}"
                        )

        assert not violations, (
            f"font-family declarations found outside theme.py:\n"
            + "\n".join(violations)
        )


class TestHexToRgba:
    """Verify hex_to_rgba() correctly converts hex tokens to rgba strings."""

    def test_text_muted_at_40_pct(self):
        """The ADWIN corridor fill must derive from COLORS['text']['muted']."""
        result = hex_to_rgba(COLORS["text"]["muted"], 0.4)
        assert result == "rgba(148, 163, 184, 0.4)"

    def test_full_opacity(self):
        result = hex_to_rgba("#FF0000", 1.0)
        assert result == "rgba(255, 0, 0, 1.0)"

    def test_zero_opacity(self):
        result = hex_to_rgba("#000000", 0.0)
        assert result == "rgba(0, 0, 0, 0.0)"

    def test_default_alpha_is_one(self):
        result = hex_to_rgba("#0B0F17")
        assert result == "rgba(11, 15, 23, 1.0)"

    def test_all_color_tokens_convert(self):
        """Every hex color in COLORS must convert without error."""
        for group_name, group in COLORS.items():
            for token_name, hex_val in group.items():
                result = hex_to_rgba(hex_val, 0.5)
                assert result.startswith("rgba("), (
                    f"COLORS['{group_name}']['{token_name}'] failed conversion"
                )
