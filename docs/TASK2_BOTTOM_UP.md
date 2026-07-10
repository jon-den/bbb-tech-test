# Task 2 — Bottom-Up Epidemiological Funnel

**BB Biotech | July 2026**

---

## 1. Introduction

> *Estimate the total addressable US patient population for Camzyos — today and over time. Be explicit about how you combine sources, how you weight them when they disagree, and how this shapes the uncertainty in your final estimate.*

This document takes the classical pharma market-sizing route: start from the US adult population and multiply through the clinical filters that define Camzyos eligibility. Every filter is a fraction, every fraction is a published estimate, and every estimate carries uncertainty. The final TAM is the product of those fractions with the uncertainties propagated forward.

The method has one thing going for it and one thing going against it. In its favour: it is transparent — a portfolio manager can walk from "261M US adults" to the final number without any hidden step. Against it: multiplying five uncertain fractions widens the interval quickly, and where sources disagree (notably on HCM prevalence, where estimates span 3×) we cannot resolve the disagreement without additional evidence. We handle that by encoding the disagreement as prior width and propagating it, not by picking a winner.

---

## 2. Method

The funnel is a chain of five multiplicative steps:

```
TAM = N_adults × p_hcm × p_diag × p_obstr × p_sympt
```

| Step | Filter | Symbol |
|---|---|---|
| 1 | US adults ≥ 18 | `N_adults` |
| 2 | of whom have HCM (echo-confirmed) | `p_hcm` |
| 3 | of whom are diagnosed and captured in the healthcare system | `p_diag` |
| 4 | of whom have LVOT obstruction (oHCM) | `p_obstr` |
| 5 | of whom are symptomatic NYHA II–III | `p_sympt` |

The FDA label is *symptomatic NYHA II–III oHCM* — steps 4 and 5 are the label. Steps 2 and 3 define the reachable diagnosed pool. Step 1 is essentially known.

**Combining disagreeing sources.** Where a filter has multiple published estimates that disagree, we do not average them into a point value. Instead we choose the prior distribution wide enough to span them and let the propagated uncertainty carry the disagreement forward. This is defensible because narrowing a disputed fraction by picking one source over another is a judgement call the data cannot support — it is more honest to report a wide TAM than a false-precision one.

**Uncertainty propagation.** Each fraction is modelled as a Beta distribution parameterised by its mean and 80% credible interval. We draw 10,000 Monte Carlo samples from each, multiply them together elementwise, and report the resulting distribution's median and 80% CI. Monte Carlo is chosen over an analytical error-propagation formula because Beta distributions are skewed near boundaries and analytical variance formulas mislead in that regime.

We do not fit anything to data. The claims cohort from Task 1 is not used — see [TASK2_SCURVE.md §1](TASK2_SCURVE.md) for why the panel has no recoverable denominator.

---

## 3. Data, Assumptions, and Parameters

| # | Parameter | Central | 80% CI | Sources |
|---|---|---:|---|---|
| 1 | US adults ≥ 18 (M) | 261 | 260–262 | US Census ACS 2024 |
| 2 | HCM prevalence | 0.25% | 0.15% – 0.40% | Maron 1995 (0.2%); Semsarian 2015 (~0.5% with genetic testing); Massera 2023 (UK Biobank, intermediate) |
| 3 | Fraction diagnosed | 32% | 20% – 45% | Butzner 2021 (~80/100k diagnosed vs. ~200/100k assumed prevalence → ~40% diagnosed); wide CI reflects prevalence-denominator uncertainty |
| 4 | Fraction obstructive | 62% | 50% – 75% | Maron 2006 (two-thirds obstructive at some point); Butzner 2022 (~60% oHCM in commercial claims); Desai 2022 (~38% at index, higher over follow-up) |
| 5 | Fraction symptomatic (NYHA II–III) | 45% | 30% – 60% | Ho 2018 SHaRe registry symptom burden; Desai 2022 (~75% symptomatic of which ~60% NYHA II–III); FDA label indication |

### Notes on the disagreements

**HCM prevalence (step 2).** This is the largest single source of disagreement in the funnel. Maron's 1995 CARDIA echo screen gave 0.2%. Semsarian's 2015 review argues true prevalence is closer to 0.5% once subclinical and genetically-positive-phenotype-negative cases are included. Recent imaging studies (UK Biobank, Massera 2023) land in between. Rather than pick, the prior spans 0.15%–0.40% (80% CI) — narrow enough to exclude the extreme high-prevalence claims that count preclinical carriers, wide enough to hold both the CARDIA point estimate and the modern imaging revisions.

**Diagnosed fraction (step 3).** Butzner 2021 measures diagnosed HCM in a commercial claims database at ~80/100k. If true prevalence is 200/100k, that implies ~40% diagnosed. If true prevalence is 300/100k, ~27% diagnosed. The uncertainty in step 3 is *anti-correlated* with the uncertainty in step 2 — if prevalence is higher, the diagnosed fraction is lower for the same measured diagnosed rate. In principle the Monte Carlo should model this correlation; in practice we treat them as independent because (a) the true correlation is unknown and (b) doing so is conservative (produces wider TAM, not narrower).

**Obstructive fraction (step 4).** Maron 2006 famously reframed HCM as "predominantly a disease of LVOT obstruction" with two-thirds obstructive over follow-up. Point-in-time measurements are lower — Desai 2022 finds ~38% at index. The distinction matters for a *lifetime* TAM (higher) vs. a *point-prevalent* TAM (lower). We take a midpoint of 62% reflecting the fact that eligibility is triggered whenever obstruction manifests.

**Symptomatic fraction (step 5).** Registry data (SHaRe, Ho 2018) shows most HCM patients experience symptoms at some point, but NYHA II–III at a given time is closer to half. The 45% central reflects point-in-time NYHA II–III with wide CI covering both "most symptomatic" and "most stable" reads.

---

## 4. Results

### Today

10,000 Monte Carlo draws through the funnel yield:

| Quantity | Median | 80% CI |
|---|---:|---|
| US adults (M) | 261 | 260 – 262 |
| × HCM prevalence | 653k | 402k – 1,020k |
| × diagnosed | 205k | 105k – 375k |
| × obstructive | 127k | 62k – 240k |
| × symptomatic NYHA II–III | **~57k** | **26k – 115k** |

**Central estimate: ~57,000 addressable patients in the US today, with an 80% CI of ~26k to ~115k.**

### How the disagreement shapes uncertainty

The 80% CI spans roughly 4×. That width is the direct output of taking sources that disagree at their word — it is not a modelling artefact.

Ranked contribution to TAM variance (one-at-a-time, others fixed at central):

| Step | Fixed CI width (80%) | TAM range (80%) | Share of variance |
|---|---|---|---:|
| HCM prevalence | 0.15% – 0.40% | 34k – 91k | ~40% |
| Diagnosed | 20% – 45% | 35k – 80k | ~25% |
| Obstructive | 50% – 75% | 46k – 68k | ~15% |
| Symptomatic | 30% – 60% | 38k – 76k | ~20% |
| US adults | 260M – 262M | 57k – 57k | <1% |

HCM prevalence is the dominant lever. Halving its uncertainty (e.g. by adopting a single source and defending it) would cut the TAM interval by roughly a quarter — but at the cost of the honesty gained by spanning the disagreement.

### Cross-checks

**Against BMS commercials.** BMS reports ~25,000 US patients on Camzyos as of mid-2026. At the funnel median of 57k, that implies ~44% penetration — high for a drug 4 years post-launch. Two readings: either the true TAM is higher than the funnel median (favouring the upper half of the CI, closer to 100k, giving ~25% penetration), or Camzyos is genuinely deep into its addressable pool. The upper-half reading is more consistent with continued BMS revenue growth.

**Against the Bayesian reconciliation model** ([TASK2_WRITEUP.md](TASK2_WRITEUP.md)). That approach lands on ~118k (80% CI 74k–192k) — roughly 2× the bottom-up median but overlapping the funnel's upper half. The two methods agree on the order of magnitude and disagree on where inside the ~30k–200k range the truth sits. The Bayesian model leans on literature `p ≈ 30%` (Desai 2022 as a single composite fraction); the bottom-up decomposes `p` into `p_obstr × p_sympt ≈ 62% × 45% = 28%` — arithmetically similar. The gap is primarily in the diagnosed HCM denominator: the Bayesian prior uses `N ≈ 400k`, the funnel implies `205k`. That difference reflects whether we trust Butzner's *measured* diagnosed rate (funnel) or a growth-extrapolated *inferred* one (Bayesian).

### Over time

Only step 3 (`p_diag`) is expected to drift materially over the projection horizon — Camzyos launch and physician awareness are increasing diagnosis rates. Steps 2, 4, 5 are epidemiologic constants of the disease. Step 1 (US adults) grows ~0.6%/yr.

We apply the same diagnosis-rate growth used in TASK2_WRITEUP.md — 7.4%/yr per Butzner 2021, held fixed forward:

| Year | Median TAM | 80% CI |
|---:|---:|---|
| 2024 | 57k | 26k – 115k |
| 2027 | 71k | 32k – 143k |
| 2030 | 88k | 40k – 178k |

CIs stay proportionally wide because growth is applied uniformly across the posterior.

---

## 5. Limitations

1. **Multiplicative variance.** Five uncertain fractions multiplied together give a wide interval by construction. A 4× CI on the final TAM is not a modelling defect — it is the honest arithmetic of the disagreement. Any narrower interval requires either better data on the individual filters or a willingness to pick winners among disputed sources.

2. **Independence assumption.** Steps 2 and 3 are anti-correlated (higher prevalence → lower diagnosed fraction, for a fixed measured diagnosed rate). Treating them as independent widens the CI; incorporating the correlation would narrow it. We take the wider result as conservative.

3. **No claims-data leverage.** The panel from Task 1 is not used. The 4% marker prevalence in HCM-coded patients is informative — it is what the Bayesian model in [TASK2_WRITEUP.md](TASK2_WRITEUP.md) leverages via a claims capture rate `s`. The bottom-up funnel discards that signal in exchange for methodological simplicity.

4. **Point-in-time vs. lifetime eligibility.** The obstruction and symptom fractions are point-in-time. A patient may cycle in and out of NYHA II–III, or develop obstruction after years without. A lifetime-cumulative TAM would be higher; a point-in-time TAM (what a drug ships to today) is what we report.

5. **TAM ≠ revenue.** Same caveat as elsewhere in Task 2. Persistence, net-to-gross, competitive share (aficamten), and prescriber adoption pace are not modelled.

6. **The 7.4%/yr growth rate assumes Butzner-era diagnosis dynamics persist.** Post-Camzyos-approval acceleration is plausible but unmeasured in the data available here.
