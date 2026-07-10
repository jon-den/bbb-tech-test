"""Shared dataviz palette and style helpers for all figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

# ── Palette ───────────────────────────────────────────────────────────────────
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
INK_PRI = "#0b0b0b"
INK_SEC = "#52514e"
INK_MUT = "#898781"

C_OBS = "#2a78d6"  # observed  / blue
C_PRED = "#1baf7a"  # predicted / aqua-green
C_REF = "#c3c2b7"  # reference line (45° perfect calibration)
C_SPLIT = "#898781"  # train-test divider
C_HIST = "#104281"  # histogram fill (deep navy)
C_MED = "#e88a1a"  # MC median (orange)
C_MODE = "#7c9dc7"  # mode-product marker (muted blue)
C_THRESHOLD = "#c0392b"  # threshold / cutoff line (muted red)

# 4-step ordinal ramp (light → dark) for archetype bars / highlight sequences
C_ARCH = ["#86b6ef", "#5598e7", "#2a78d6", "#104281"]

# Extended ramp (6 steps) for highlighting multiple features in LASSO paths
_HIGHLIGHT_RAMP = ["#c5d8f5", "#86b6ef", "#5598e7", "#2a78d6", "#1a5fa8", "#104281"]


def highlight_colors(n: int) -> list[str]:
    """Return n evenly-spaced colours from the light-to-dark blue ramp."""
    ramp = _HIGHLIGHT_RAMP
    if n <= len(ramp):
        # pick evenly spaced, always including the darkest
        step = max(1, len(ramp) // n)
        return [ramp[min(i * step, len(ramp) - 1)] for i in range(n)]
    # more features than ramp entries — cycle
    return [ramp[i % len(ramp)] for i in range(n)]


def style_ax(
    ax: plt.Axes,
    *,
    grid_axis: str = "y",
    hide_top_right: bool = False,
) -> plt.Axes:
    """Apply shared palette to axes: background, spines, ticks, optional grid.

    Args:
        ax: Target axes.
        grid_axis: Which axis to add grid lines to — "x", "y", "both", or "none".
        hide_top_right: If True, make top and right spines invisible (open frame).

    Returns:
        The same axes, for chaining.
    """
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK_MUT, labelsize=9)
    if grid_axis in ("x", "both"):
        ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    if grid_axis in ("y", "both"):
        ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    return ax
