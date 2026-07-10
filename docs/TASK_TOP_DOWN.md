# Task 2 — Top-down epidemiological funnel

**BB Biotech | July 2026**

---

## 1. Introduction

> *Estimate the total addressable US patient population for Camzyos — today and over time. Be explicit about how you combine sources, how you weight them when they disagree, and how this shapes the uncertainty in your final estimate.*

A top-down epidemiological funnel chains published fractions from the whole US adult population down to the Camzyos-eligible pool. 

This method allows to combine multiple sources: if two credible papers report different values for the same fraction, both are used as range of a triangular distribution rather than picking a winner. Monte Carlo simulation then propagates each input's uncertainty through the funnel to produce a TAM distribution.

**Why not reuse the Task 1 data.** The dataset from task 1 has ~30k cardiac patients whose selection criteria are unspecified. There is no sampling fraction, so no panel count can be scaled to a national level. A top-down funnel must anchor on numbers that carry a US denominator (Census, published US prevalence studies).

---

## 2. Method

The addressable market is the product of three quantities:

```
TAM  =  US adults  ×  diagnosed HCM prevalence  ×  Camzyos-eligible fraction
```

Each fraction is uncertain, and published sources disagree. To quantify uncertainty, the approach is as follows:

1. For each parameter, collect the published estimates.
2. Fit a **triangular distribution** with `min` = lowest defensible value, `mode` = central/most-cited value, `max` = highest defensible value. **Wider disagreement between sources → wider triangle.**.
3. Draw 10,000 Monte Carlo samples, multiply through the funnel, and report the resulting distribution of TAMs.

Triangular distributions are the standard choice when only min/mode/max are known — no distributional assumptions beyond "values near the mode are more likely than values at the extremes."

---

## 3. Data, assumptions, and parameters

### US adults (2026)

Fixed at **264M** (US Census 2026 projection).

### Diagnosed HCM prevalence, per 100k — triangle `(92, 105, 200)`

Sources: Butzner 2021 [@butzner2021]; Massera 2023 [@massera2023].

Butzner 2021 (HealthCore HIRD, US commercial claims) reports 80/100k diagnosed HCM in 2019 and a measured rate of 7.44%/yr for 2013–2019. No published post-2019 measurement exists, so the 2026 anchor is a growth-rate extrapolation from Butzner's 2019 baseline. The triangle bounds correspond to the same growth scenarios used in the over-time table:

- **Min = 92/100k** = `80 × 1.02⁷` — 2%/yr from 2019 (floor: diagnostic catch-up complete, demographic baseline only).
- **Mode = 105/100k** = `80 × 1.04⁷` — 4%/yr from 2019 (base: roughly half of Butzner's measured 2013–2019 rate).
- **Max = 200/100k** — Massera 2023 imaging-phenotype ceiling (~1:500), i.e. the biological cap reached only if clinical underdiagnosis were fully eliminated. Massera's reported ~2.7× underdiagnosis factor applied to Butzner's 80/100k gives 216/100k, consistent with this bound.

Butzner's measured 7.44%/yr rate is treated as the growth **ceiling**, not the base — it captured one-time diagnostic catch-up (ICD-10 rollout Oct 2015, HCM-centre expansion) that is unlikely to persist.

### Camzyos-eligible fraction — triangle `(22%, 36%, 53%)`

Sources: Schultze 2022 [@schultze2022]; Batzner 2019 [@batzner2019]; Ho 2018 [@ho2018].

Camzyos's label is symptomatic obstructive HCM on maximally tolerated first-line therapy (beta-blocker, CCB, or disopyramide). The eligible fraction of diagnosed HCM is built as the product of:

- **Obstructive share of HCM — triangle `(0.49, 0.60, 0.70)`.** Obstructive HCM makes up roughly half to two-thirds of diagnosed HCM. Schultze 2022 (UK/Germany population estimates): 68% UK, 49% Germany. Batzner 2019 (review, *Dtsch Arztebl Int*): ~70% of HCM patients have the obstructive type. Bounds = 0.49 (Schultze Germany) and 0.70 (Batzner review); mode = 0.60 as the midpoint of the "half-to-two-thirds" range.
- **Symptomatic (NYHA II+) share of oHCM — assumed triangle `(0.45, 0.60, 0.75)`.** This is an assumption, not a citation. NYHA class is not recorded in claims, but could be extracted from EHR data. Mode = 0.60 (Ho 2018, SHaRe registry).

Multiplying:

- **Min = 22%** = `0.49 × 0.45`
- **Mode = 36%** = `0.60 × 0.60`
- **Max = 53%** = `0.70 × 0.75`

### Growth over time

`prev(year) = 80/100k × (1 + g)^(year − 2019)`, applied consistently from Butzner's 2019 baseline through the target year. The three scenarios are the same ones that anchor the 2026 prevalence triangle above:

- **2%/yr — floor.** Diagnostic catch-up complete, demographic baseline only.
- **4%/yr — base.** Roughly half of Butzner's measured rate; some ongoing improvement in imaging access and awareness.
- **7.4%/yr — ceiling.** Butzner 2013–2019 measured rate; only realised if ICD-10-era catch-up continues at the historical pace.

---

## 4. Results

### Today (2026)

| Quantity | Median | 80% CI |
|---|---:|---|
| **US addressable market, 2026** | **~124k patients** | **~91k – 173k** |

*Monte Carlo over triangular priors, 10,000 draws, seed 42.*


### Bear / base / bull 


| Scenario | Diagnosed prev | Eligible fraction | TAM |
|---|---|---|---:|
| **Bear** | 92/100k (2%/yr from 2019) | 22% (strict) | **~53k** |
| **Base** | 105/100k (4%/yr from 2019) | 36% (0.60 × 0.60 central) | **~100k** |
| **Bull** | 200/100k (Massera imaging-phenotype ceiling) | 52.5% (permissive) | **~277k** |

The MC median (124k) exceeds the deterministic base estimate (100k) because the prevalence triangle is right-skewed (mode 105, max 200).


### Over time

| Growth rate | Basis | 2026 | 2028 | 2030 |
|---|---|---:|---:|---:|
| 2%/yr | Floor — demographic aging only | ~87k | ~91k | ~95k |
| **4%/yr** | **Base — half of Butzner's rate** | **~100k** | **~108k** | **~117k** |
| 7.4%/yr | Ceiling — Butzner 2021 measured | ~125k | ~145k | ~167k |

Each row applies its growth rate consistently from Butzner's 2019 baseline (80/100k) — so the 2026 column already reflects the scenario, not a shared anchor. Point estimates use mode eligibility 0.36; the eligible fraction is assumed constant over time.

**By 2030 the TAM plausibly reaches ~95k–167k patients**, with a base case of ~117k.

Exponential growth is unbounded, but diagnosed prevalence cannot exceed true prevalence (Massera's ~200/100k imaging-phenotype ceiling). Over the 2026–2030 window the diagnosed rate stays well below that ceiling under all three scenarios, but the growth modelling needs adaptation beyond.


---

## 5. Limitations

1. **Independence assumption.** Monte Carlo multiplies parameters as if their errors are uncorrelated. In reality, if diagnosis rates rise it is partly because of Camzyos-driven awareness, which may also shift the observed symptomatic share. This coupling is not modelled; independence probably makes the CI slightly too tight.

2. **Growth beyond 2019 is unmeasured.** Butzner 2021 stops at 2019. The 2–7.4%/yr range post-2019 is bracketed by scenarios, not observed, and treats Butzner's rate as a ceiling on the assumption that ICD-10-era catch-up will not persist. A published 2020–2025 prevalence update would collapse this uncertainty.

3. **Diagnosed vs true prevalence.** The funnel uses diagnosed prevalence throughout — undiagnosed HCM patients cannot be prescribed a drug. If diagnostic capture improves faster than the assumed growth rate, the funnel understates future TAM.

4. **Adult-only, current-label denominator.** The near-term label catalyst is SCOUT-HCM (adolescents 12 to <18 with oHCM), which met its primary endpoint and has an sNDA under FDA priority review. Approval would add patients *outside* the adult denominator, so the funnel would understate TAM. Non-obstructive HCM is not a lever: ODYSSEY-HCM missed both co-primary endpoints in April 2025.

---

## 6. Extensions with more time

**Bring the claims data in.** The Task 1 panel's HCM-coded fraction carrying a Camzyos marker (I42.1 + Disopyramide) can be modelled as `true eligibility × claims capture rate`, letting a Bayesian update tighten the eligibility prior while jointly inferring the capture rate.

**Refresh the prevalence input.** The prevalence triangle extrapolates 2019 Butzner data forward seven years. A direct 2026 read from BB Biotech's IQVIA/Symphony/Komodo subscription (against a 30M-life denominator) would collapse the ~51% of variance from this parameter.

**Pin the eligible fraction via chart review.** A structured chart review of 200–300 I42.1-coded patients for NYHA class and LVOT gradient would replace the 20–40% triangle with a data-anchored estimate. Highest-value single follow-on — this parameter drives ~49% of TAM variance.

**Age-stratify.** Butzner's commercial-claims anchor underrepresents the ≥65 (Medicare) population where HCM prevalence is higher. Splitting the funnel into <65 and ≥65 strata with separate prevalence inputs would incorporate Medicare directly rather than absorbing it into the triangle's upper tail.
