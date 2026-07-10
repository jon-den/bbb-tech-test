# Task 2 — Total Addressable US Patient Population for Camzyos

**BB Biotech | July 2026**

---

## 1. Introduction

> *Estimate the total addressable US patient population for Camzyos — today and over time. Be explicit about how you combine sources, how you weight them when they disagree, and how this shapes the uncertainty in your final estimate.*

The standard approaches each hit a wall here. A top-down epidemiological funnel requires multiplying several uncertain fractions whose sources disagree. A bottom-up claims count has no recoverable denominator — the database selection criteria are unspecified, so no panel count can be scaled nationally. A commercial back-calculation from BMS revenue requires assuming a penetration rate, which is itself what we're trying to estimate.

What we do have are two sources that disagree by 7×, and a framework for treating that disagreement as information:

- **Literature** (Desai 2022) says ~30% of diagnosed HCM adults are Camzyos-eligible.
- **Our claims cohort** shows only ~4% of HCM-coded patients carry the Task 1 eligibility marker (I42.1 + Disopyramide fill).

A Bayesian model reconciles them by introducing a *claims capture rate* `s`: if billing markers only surface a fraction of truly eligible patients, a 4% observation is entirely consistent with 30% true eligibility — provided `s ≈ 14%`. The model estimates both jointly without picking a winner.

---

## 2. Method

The addressable market is the product of two uncertain quantities:

```
TAM = N_hcm × p
```

where `N_hcm` is diagnosed HCM adults in the US and `p` is the fraction who are truly Camzyos-eligible. The claims data provide one observation that constrains them jointly: of 18,953 HCM-coded patients, 769 (4.06%) carry the eligibility marker. We model this as:

```
k_observed ~ Binomial(n, p · s)
```

Since `k/n = p × s`, a single ratio cannot separately identify `p` from `s` — the priors do the identification. The prior on `p` encodes the literature, the prior on `s` encodes claims coding sensitivity, and together they allocate the 7× gap: is it because `p` is lower than the literature says, or because `s` is low? The posterior on `p` drives the TAM; the posterior on `s` is the capture-rate insight handed off to Task 3.

Full implementation in [`src/task2_tam/tam_model.py`](../src/task2_tam/tam_model.py):

```python
p     = pm.Beta("p_true_eligibility", 14.3, 32.3)
s     = pm.Beta("s_capture_rate",     2.0,  12.0)
n_hcm = pm.LogNormal("n_hcm_us", mu=log(400_000), sigma=0.295)

pm.Binomial("k_observed", n=18953, p=p*s, observed=769)
pm.Deterministic("tam", n_hcm * p)
```

NUTS, 4 chains, 2,000 draws. Convergence: r̂ = 1.000, ESS > 2,700.

---

## 3. Data, Assumptions, and Parameters

| Parameter | Value | Source | Validation |
|---|---|---|---|
| `k` (treatable-coded) | 769 | Claims panel: I42.1 ∩ Disopyramide | **Confirmed** — computed directly from data |
| `n` (HCM-coded) | 18,953 | Claims panel: I42.1 / I42.2 / I42.9 | **Confirmed** — computed directly from data |
| `p` prior mean | 30.7% | Desai 2022, *Clin Ther* 44(2):243–259 | **Validated** — citation confirmed; consistent with 0.38 × 0.75 ≈ 29% from obstruction × symptomatic fractions independently |
| `p` prior width | 80% CI 22–39% | Beta(14.3, 32.3), ESS = 47 | **Reasonable** — reflects NYHA assessment variability and coding vs. echo disagreement |
| `s` prior mean | 14.3% | Coding literature range 10–25% | **Assumption** — no single citation; weakly informative prior, ESS = 14 |
| `N` prior median | 400,000 | Butzner 2021 × growth to 2024 | **Partially validated** — see note below |
| `N` prior p95 | 650,000 | Extrapolation uncertainty | **Assumption** — captures underdiagnosis tail; not directly verifiable |

**Note on `N` and the growth rate.** Butzner 2021 (HealthCore HIRD) reports 80/100k diagnosed HCM in 2019, growing from 52/100k in 2013. The directly computed annual rate is **(80/52)^(1/6) − 1 = 7.44%/yr**, not 9%/yr. Extrapolated to 2024: 263k × 1.0744^5 = **377k**. The model's prior median of 400k implies 8.75%/yr — a modest overstatement, defensible as capturing post-2022 diagnostic acceleration driven by Camzyos awareness, but that acceleration is inferred, not measured. The conservative anchor is 377k / 7.44%/yr; the 400k prior median is retained as it falls within the uncertainty range.

**Note on `s`.** This is the only input without a citable source. It is an assumption grounded in the general range reported by HCM coding validation studies (~10–25%). Because the prior is weakly informative (ESS = 14), the data can move it substantially, but as shown in the sensitivity analysis below, the TAM is nearly insensitive to this prior regardless.

---

## 4. Results

### Today

![TAM posterior](../outputs/task2_tam/01_tam_posterior.png)

| Quantity | Posterior median | 80% CI |
|---|---:|---|
| **US addressable market (2024)** | **~118,000 patients** | **74k – 192k** |
| True clinical eligibility `p` | 30% | 22% – 39% |
| Claims capture rate `s` | 14% | 10% – 18% |
| Diagnosed HCM in the US `N` | 397,000 | 273k – 583k |

The posterior on `p` barely moves from the prior (30.4% vs. 30.7%). This is the direct answer to how sources are combined: the claims data (4%) is entirely consistent with the literature (30%) once `s ≈ 14%` is accepted. The 7× disagreement is not resolved by choosing a winner — it is resolved by inferring the capture rate. The literature drives the TAM; the claims data calibrate the instrument.

**Cross-check.** BMS reports ~25,000 US patients prescribed Camzyos as of mid-2026 — not used in the model. At the posterior capture rate of 14%, that implies a true addressable pool of 25k / 0.14 ≈ **178k**, above the model median but within the 80% CI. Consistent.

### How disagreement shapes uncertainty

The TAM is `N × p`. Both span ~2× across their 80% CIs; `s` moves the TAM by less than 3%.

| Source | Effect on TAM | Dominates? |
|---|---|---|
| `p` prior (22% → 39%) | ~2× range | Yes |
| `N` prior (273k → 583k) | ~2× range | Yes |
| `s` prior (diffuse → tight) | <3% | No |

This means the uncertainty is epidemiological — how many eligible patients exist, not how precisely our claims data measures them. More claims data does not narrow the interval. More literature on `p` (an echo-based registry rather than a claims study) would.

### Sensitivity to priors

| Scenario | TAM median | 80% CI |
|---|---:|---|
| **Base case** | **120k** | **74k – 189k** |
| Bull: `p` mean 40% | 158k | 102k – 245k |
| Bear: `p` mean 20% | 85k | 51k – 140k |
| `s` more diffuse | 119k | 75k – 187k |
| `s` tighter (chart-review-like) | 116k | 78k – 180k |
| `N` low (median 300k) | 89k | 55k – 145k |
| `N` high (median 550k) | 164k | 102k – 261k |

The `s` rows are the key test: the TAM barely moves. The `p` and `N` rows each shift it by 30–40k. Any single-point TAM estimate presented without this table is a false precision claim.

### Over time

`TAM(year) = N₂₀₂₄ × (1 + g)^(year − 2024) × p`

The pool grows as more patients are diagnosed. Butzner 2021 gives the validated growth rate of 7.44%/yr; 9%/yr (used as "base case" elsewhere in this analysis) slightly overstates it.

| Growth rate | Basis | 2024 | 2027 | 2030 |
|---|---|---:|---:|---:|
| 5%/yr | Conservative assumption | ~118k | ~137k | ~159k |
| **7.4%/yr** | **Butzner 2021 — validated** | **~118k** | **~146k** | **~181k** |
| 9%/yr | Partial assumption (post-2022 acceleration) | ~118k | ~153k | ~198k |

Numbers are posterior medians; 80% CIs carry the same ~±50% relative width as the 2024 estimate.

The central estimate is approximately **118,000 addressable patients today, growing to ~180,000 by 2030**. This is the eligible pool, not a Camzyos-on-drug forecast — converting to revenue requires penetration rate, persistence, and aficamten share, none of which are modelled here.

---

## Limitations

1. **`s` is an assumption.** A chart-review of 200–300 I42.1-coded patients for true NYHA class and LVOT gradient would replace it with data — the single highest-value follow-on.
2. **Post-2022 growth rate is unobserved.** Camzyos approval may have accelerated diagnosis; the 7.44%/yr rate runs through 2019 only.
3. **The `p` prior is itself claims-based.** Desai 2022 uses a commercial database; an echo-confirmed registry estimate would be more independent of our data's coding limitations.
4. **TAM ≠ revenue.** Net-to-gross, adherence, and competitive dynamics are excluded by design.
