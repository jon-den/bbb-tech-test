"""Shared dataviz palette and style helpers for all figures.

Visual language: clean editorial look — bottom spine only, no tick marks,
faint warm-grey horizontal grid, generous title padding. Colour palette
uses a vibrant primary blue with warm orange and green accents.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# ── Palette ───────────────────────────────────────────────────────────────────
SURFACE = "#ffffff"
GRID = "#ECE9E5"  # faint warm grey — horizontal grid guides
SPINE = "#C2BAB5"  # warm grey — bottom baseline spine
INK_PRI = "#000000"  # titles — pure black for clear hierarchy
INK_SEC = "#544F4F"  # axis labels, body text — dark warm grey
INK_MUT = "#706B69"  # tick labels, secondary annotations — mid warm grey

C_OBS = "#0B41CD"  # primary blue (observed / main series)
C_PRED = "#ED4A0D"  # warm orange (predicted / secondary series)
C_REF = "#C2BAB5"  # reference line (45° perfect calibration)
C_SPLIT = "#706B69"  # train-test divider
C_HIST = "#0B41CD"  # histogram fill (primary blue)
C_MED = "#ED4A0D"  # MC median (warm orange)
C_MODE = "#BDE3FF"  # mode-product marker (extra-light blue)
C_THRESHOLD = "#C40000"  # threshold / cutoff line (dark red)

# 4-step ordinal ramp (light → dark) for archetype bars / highlight sequences
C_ARCH = ["#BDE3FF", "#1482FA", "#0B41CD", "#022366"]

# Extended ramp (6 steps) for highlighting multiple features in LASSO paths
_HIGHLIGHT_RAMP = ["#BDE3FF", "#86b6ef", "#1482FA", "#0B41CD", "#022366", "#011133"]


def highlight_colors(n: int) -> list[str]:
    """Return n evenly-spaced colours from the light-to-dark blue ramp."""
    ramp = _HIGHLIGHT_RAMP
    if n <= len(ramp):
        step = max(1, len(ramp) // n)
        return [ramp[min(i * step, len(ramp) - 1)] for i in range(n)]
    return [ramp[i % len(ramp)] for i in range(n)]


def style_ax(
    ax: plt.Axes,
    *,
    grid_axis: str = "y",
    hide_top_right: bool = False,
) -> plt.Axes:
    """Apply shared palette to axes: editorial clean look.

    Default: bottom spine only, no tick marks, faint warm-grey grid.

    Args:
        ax: Target axes.
        grid_axis: Which axis to add grid lines to — "x", "y", "both", or "none".
        hide_top_right: If True, also hide top and right spines (legacy compat;
            now the default hides top/right/left anyway).

    Returns:
        The same axes, for chaining.
    """
    ax.set_facecolor(SURFACE)

    # Editorial spine convention: bottom only, thin warm grey
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["bottom"].set_linewidth(0.8)

    # No tick marks — labels float with padding
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        colors=INK_MUT,
        labelsize=9,
        pad=5,
    )

    # Grid
    ax.set_axisbelow(True)
    if grid_axis in ("x", "both"):
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    if grid_axis in ("y", "both"):
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)

    return ax
