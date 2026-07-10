"""Top-down epidemiological funnel for the Camzyos US TAM.

    TAM = US adults × diagnosed HCM prevalence × Camzyos-eligible fraction

Each fraction is a triangular distribution over published bounds; Monte Carlo
propagates the joint uncertainty to a TAM distribution. The 2026 prevalence
triangle is source-anchored (Husser 2018 / Butzner 2021 / Massera 2023) with
no growth adjustment; growth scenarios apply only from 2026 forward for the
over-time projection.

Every parameter carries a citation in `FunnelParams`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FunnelParams:
    """Every parameter with a citation. Change one number here — nothing else."""

    # US adult population, 2026 projection (US Census)
    us_adults: float = 264e6

    # Diagnosed HCM prevalence for 2026, per 100k. Directly-cited published
    # sources span the range; no growth adjustment applied (temporal +
    # geographic variation is captured by the triangle width).
    #   Min:  Husser 2018 (Germany 2015, ~5M-patient claims): 70/100k
    #         (clinical HCM = 0.07% = 1/1,372 patients).
    #   Mode: Butzner 2021 (US commercial claims HIRD 2019): 80/100k.
    #   Max:  Massera 2023 (imaging-phenotype ceiling ~1:500): 200/100k —
    #         biological cap if underdiagnosis were fully eliminated.
    prev_min: float = 70.0
    prev_mode: float = 80.0
    prev_max: float = 200.0

    # Growth scenarios applied ONLY for over-time projection from `anchor_year`
    # (2026) forward. The 2026 triangle above is already source-anchored and
    # does NOT use these.
    # Floor: Butzner 2026 measured HCM incidence 2017-2023 (50->56/100k, ~1.9%/yr).
    # Ceiling: Butzner 2021 measured HCM prevalence 2013-2019 (52->80/100k, 7.44%/yr).
    # Base: midpoint of the two measured rates (no editorial adjustment).
    growth_floor: float = 0.02
    growth_base: float = 0.047
    growth_ceiling: float = 0.074

    # Obstructive share of diagnosed HCM.
    #   min  = 0.49  (Schultze 2022 population estimate, Germany)
    #   max  = 0.70  (Batzner 2019 Dtsch Arztebl Int review)
    #   mode = 0.60  (midpoint of the "half-to-two-thirds" range)
    obstr_min: float = 0.49
    obstr_mode: float = 0.60
    obstr_max: float = 0.70

    # Symptomatic NYHA II-III share of oHCM (Camzyos label indication is
    # specifically NYHA class II-III; NYHA I asymptomatic and NYHA IV are
    # excluded).
    # Min:  Butzner 2026 (Symphony IDV claims): 53/117 = 45% — claims-based
    #       symptomatic detection via ICD-10 codes; lower bound reflecting
    #       imperfect symptom-code sensitivity. Includes some NYHA IV; kept as
    #       the conservative floor since claims can't reliably filter by NYHA.
    # Mode: Wang 2023 (US HCP cohort, n=754): NYHA II + III = 44.0 + 30.2 =
    #       74.2%. Excludes NYHA I (20.0%) and NYHA IV (5.7%).
    # Max:  Charron 2026 (France nationwide oHCM registry): NYHA II + III =
    #       32 + 60 = 92%. Excludes NYHA IV (4%).
    sympt_min: float = 0.45
    sympt_mode: float = 0.74
    sympt_max: float = 0.92

    # Time anchors
    anchor_year: int = (
        2026  # "today" for the headline TAM; also the base year for growth extrapolation
    )
    target_years: tuple[int, ...] = (2026, 2028, 2030)


def prevalence_triangle_2026(p: FunnelParams) -> tuple[float, float, float]:
    """2026 prevalence triangle (per 100k) as (min, mode, max).

    Source-anchored, no growth adjustment:
      min  = Husser 2018 (Germany 2015)
      mode = Butzner 2021 (US 2019)
      max  = Massera 2023 (imaging-phenotype ceiling)
    """
    return (p.prev_min, p.prev_mode, p.prev_max)


def eligibility_triangle(p: FunnelParams) -> tuple[float, float, float]:
    """Eligibility triangle as (min, mode, max) = product of the two component triangles.

    Obstructive × symptomatic, each at its respective bound.
    """
    return (
        p.obstr_min * p.sympt_min,
        p.obstr_mode * p.sympt_mode,
        p.obstr_max * p.sympt_max,
    )


def run_mc(
    p: FunnelParams | None = None,
    seed: int = 42,
    n_draws: int = 10_000,
) -> dict[str, np.ndarray]:
    """Monte Carlo over the 2026 prevalence × eligibility triangles.

    Uses legacy `np.random.seed` + `np.random.triangular` for stable
    reproducibility with published doc numbers.

    Args:
        p: funnel parameters (defaults to `FunnelParams()`).
        seed: NumPy RNG seed.
        n_draws: number of MC draws.

    Returns:
        Dict with arrays: 'prev' (per-capita), 'elig' (fraction), 'tam' (patients).
    """
    if p is None:
        p = FunnelParams()

    prev_lo, prev_mode, prev_hi = prevalence_triangle_2026(p)
    elig_lo, elig_mode, elig_hi = eligibility_triangle(p)

    np.random.seed(seed)
    prev_per_100k = np.random.triangular(prev_lo, prev_mode, prev_hi, n_draws)
    elig = np.random.triangular(elig_lo, elig_mode, elig_hi, n_draws)
    prev = prev_per_100k / 1e5
    tam = p.us_adults * prev * elig

    return {"prev": prev, "elig": elig, "tam": tam}


def deterministic_scenarios(p: FunnelParams | None = None) -> list[dict]:
    """Bear / Base / Bull point estimates (product of chosen triangle values).

    Returns:
        List of dicts with keys: scenario, prev_per_100k, elig_fraction, tam.
    """
    if p is None:
        p = FunnelParams()
    plo, pmode, phi = prevalence_triangle_2026(p)
    elo, emode, ehi = eligibility_triangle(p)
    return [
        {
            "scenario": "Bear",
            "prev_per_100k": plo,
            "elig_fraction": elo,
            "tam": int(p.us_adults * plo / 1e5 * elo),
        },
        {
            "scenario": "Base",
            "prev_per_100k": pmode,
            "elig_fraction": emode,
            "tam": int(p.us_adults * pmode / 1e5 * emode),
        },
        {
            "scenario": "Bull",
            "prev_per_100k": phi,
            "elig_fraction": ehi,
            "tam": int(p.us_adults * phi / 1e5 * ehi),
        },
    ]


def over_time_from_anchor(
    tam_anchor_2026: float,
    p: FunnelParams | None = None,
    years: Iterable[int] | None = None,
) -> list[dict]:
    """Project a 2026 TAM anchor forward under each growth scenario.

    All three scenarios share the same 2026 anchor (typically the MC median)
    and diverge from `anchor_year` (2026) forward under floor/base/ceiling
    growth rates.

    Args:
        tam_anchor_2026: TAM value at anchor_year — pass the MC median from
            `run_mc()` so the projection is consistent with the headline number.
        p: funnel parameters.
        years: iterable of target years (default `p.target_years`).

    Returns:
        List of dicts with keys: scenario, growth_rate, year, tam.
    """
    if p is None:
        p = FunnelParams()
    if years is None:
        years = p.target_years

    rows = []
    for name, g in [
        ("floor_2pct", p.growth_floor),
        ("base_4_7pct", p.growth_base),
        ("ceiling_7_4pct", p.growth_ceiling),
    ]:
        for year in years:
            growth_factor = (1 + g) ** (year - p.anchor_year)
            rows.append(
                {
                    "scenario": name,
                    "growth_rate": g,
                    "year": year,
                    "tam": int(tam_anchor_2026 * growth_factor),
                }
            )
    return rows


def variance_decomposition_log(prev: np.ndarray, elig: np.ndarray) -> dict[str, float]:
    """Share of log-variance in TAM attributable to prevalence vs eligibility.

    Valid because log(TAM) = log(constants) + log(prev) + log(elig), and the
    two components are independent by construction (independent MC draws).
    """
    lv_prev = float(np.var(np.log(prev)))
    lv_elig = float(np.var(np.log(elig)))
    total = lv_prev + lv_elig
    return {
        "prevalence": lv_prev / total,
        "eligibility": lv_elig / total,
    }


def plot_tam_distribution_2026(
    mc_tam: np.ndarray,
    p: FunnelParams | None = None,
    out_path: str | Path = "outputs/task2_tam/02_top_down_tam_2026.png",
) -> Path:
    """Histogram of 2026 MC TAM draws with median, mode-product, and 80% CI markers.

    Args:
        mc_tam: TAM draws from `run_mc()`, in patients.
        p: funnel parameters (for the deterministic mode-product marker).
        out_path: destination PNG path.

    Returns:
        The saved Path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    from src.style import C_HIST, C_MED, C_MODE, INK_MUT, INK_PRI, INK_SEC, SURFACE, style_ax

    if p is None:
        p = FunnelParams()

    median = float(np.percentile(mc_tam, 50))
    p10 = float(np.percentile(mc_tam, 10))
    p90 = float(np.percentile(mc_tam, 90))
    # Deterministic mode-product = US adults × mode prev × mode eligibility
    mode_product = p.us_adults * (p.prev_mode / 1e5) * (p.obstr_mode * p.sympt_mode)

    tam_k = mc_tam / 1000
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)

    # Histogram — cap x-axis at the 99.5th percentile so the visible bars
    # fill the frame; the far right tail beyond ~1% is dead space.
    x_upper = float(np.percentile(tam_k, 99.5))
    ax.hist(
        tam_k,
        bins=40,
        color=C_HIST,
        alpha=0.80,
        edgecolor=SURFACE,
        linewidth=0.5,
        zorder=2,
    )
    ax.set_xlim(left=max(0, float(np.percentile(tam_k, 0.5)) - 5), right=x_upper + 5)

    # 80% CI shaded band
    ax.axvspan(
        p10 / 1000,
        p90 / 1000,
        color=C_HIST,
        alpha=0.08,
        zorder=1,
        label=f"80% CI ({p10 / 1000:.0f}k – {p90 / 1000:.0f}k)",
    )

    # MC median line
    ax.axvline(
        median / 1000,
        color=C_MED,
        linewidth=2.4,
        linestyle="--",
        zorder=3,
        label=f"MC median = {median / 1000:.0f}k",
    )

    # Deterministic mode-product marker
    ax.axvline(
        mode_product / 1000,
        color=C_MODE,
        linewidth=1.8,
        linestyle=":",
        zorder=3,
        label=f"Deterministic mode-product = {mode_product / 1000:.0f}k",
    )

    ax.set_xlabel("US addressable market, 2026 (patients)", fontsize=9, color=INK_SEC)
    ax.set_ylabel("MC draws", fontsize=9, color=INK_SEC)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x)}k"))
    style_ax(ax, grid_axis="y", hide_top_right=True)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title(
        "Monte Carlo distribution of 2026 TAM (n=10,000 draws)",
        fontsize=10,
        fontweight="bold",
        color=INK_PRI,
        loc="left",
        pad=10,
    )
    ax.text(
        0.0,
        -0.22,
        "MC median (orange) is above the deterministic mode-product (blue dotted)\n"
        "because the prevalence triangle is right-skewed — the MC integrates the long upper tail.",
        transform=ax.transAxes,
        fontsize=8,
        color=INK_MUT,
    )

    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_path
