# Task 2 — Logistic S-curve approach

---

## 1. Approach, assumptions, and why Task 1 data is not used

### Why not Task 1

Task 1 modelled adoption within the Disopyramide-conditioned risk set: ~775 HCM patients who had already received Disopyramide before Camzyos launched. Three properties of that cohort disqualify it as a TAM or trajectory instrument:

1. **The S-curve is degenerate.** Fitting a three-parameter logistic to the 21-month cumulative series returns K = 147 — essentially equal to the 146 events observed. The model reads the deceleration in monthly starts as market saturation. But that deceleration is a feature of the synthetic data generator, not the real market. BMS revenue more than doubled ($84M → $201M/quarter) over the same window, confirming the real market is still accelerating.

2. **No recoverable denominator.** The panel has ~30k cardiac patients from a database with unspecified selection criteria. There is no sampling fraction, so no panel count can be scaled to a national level.

Instead, we anchor on what does have a denominator: BMS's own commercial disclosure (~25k US patients as of mid-2026) and a literature-derived TAM.

### Why the logistic S-curve

Drug adoption follows a constrained diffusion process: slow start as awareness and prescriber comfort build, acceleration as KOL networks propagate, then deceleration as the remaining eligible patients become harder to reach. The logistic is the standard functional form for this — it is the solution to the differential equation "growth rate is proportional to both current penetration and remaining headroom." It has three parameters, which is minimal for the shape we observe.

A secondary benefit: once calibrated, the curve extrapolates forward analytically. We do not need to simulate individual patient-months or project panel trends beyond their observation window.

### Assumptions

- The logistic is the right functional form. Structural alternatives (Gompertz, Bass diffusion) would add parameters without a strong empirical justification for Camzyos specifically.
- K (the ceiling) is fixed across the projection period. TAM grows slowly with increasing HCM diagnosis rates; ignoring that slightly understates 2030. We treat it as a constant and note the direction of the bias.
- The BMS cumulative prescription count (25k) is a reliable anchor. It likely represents cumulative starts, not current patients on drug — if persistence is ~70–80%, point-in-time prevalence is lower. We use cumulative starts throughout, which is appropriate for sizing the addressable market.
- The inflection timing is set by domain judgment, not data. This is the dominant uncertainty and is treated explicitly as a sensitivity.

---

## 2. Parameters — sources and values

| Parameter | Symbol | Value | Source |
|---|---|---|---|
| TAM ceiling | K | **100k** central [80k, 150k] | Epidemiological funnel (our estimate); BMS commercial cross-check implies 83k–167k at 15–30% penetration |
| Cumulative patients at anchor | C(t₀) | **25,000** | BMS commercial disclosure, mid-2026 |
| Anchor date | t₀ | **51 months** post-launch | April 2022 launch → July 2026 |
| Inflection timing | t_inf | **72 months** central [60, 84] | Domain judgment: first-in-class REMS cardiology drug, no competitor until aficamten in 2026–27; specialty drug inflection benchmarks typically year 3–6 |
| Intrinsic growth rate | r | **back-solved** from K, C(t₀), t_inf | No free parameter — see derivation below |

### Deriving r

Given K, t_inf, and C(t₀), the growth rate r is fully determined:

```
C(t₀) = K / (1 + exp(−r · (t₀ − t_inf)))

rearranging:  r = −ln(K / C(t₀) − 1) / (t₀ − t_inf)
```

For central values: r = −ln(100,000 / 25,000 − 1) / (51 − 72) = −ln(3) / (−21) = **0.0523/month**

This is derived, not assumed. The only genuine assumptions are K and t_inf. All outputs are sensitive to these two; r has no independent degree of freedom.

---

## 3. Calculations

**Central case:** K = 100k, t_inf = 72 months (Apr 2028), r = 0.052/month

| Date | Months post-launch | Patients on Camzyos | Note |
|---|---:|---:|---|
| Apr 2022 (launch) | 0 | ~2,300 | Logistic never reaches 0; consistent with trial rollover cohort |
| Dec 2023 | 21 | ~6,500 | Cross-check: BMS Q4 2023 revenue implies ~6,100 — **within 6%** |
| **Jul 2026 (today)** | **51** | **25,000** | Anchor — by construction |
| Apr 2028 (inflection) | 72 | 50,000 | Peak monthly growth rate: ~1,300 new starts/month |
| Dec 2030 | 105 | ~85,000 | 85% of ceiling |

### Q4 2023 revenue cross-check (unfitted)

The model was calibrated solely to the July 2026 BMS figure. Q4 2023 revenue ($84M) at $55k/year list price implies ~6,100 patients on drug. The central S-curve gives 6,488 at month 21 — agreement within 6%, without any fitting to that data point. This is the primary validation.

### Sensitivity: t_inf (K = 100k fixed)

| Scenario | t_inf | r (per month) | Patients Dec 2030 | Share of ceiling |
|---|---|---|---:|---:|
| Early peak | 60 months (Apr 2027) | 0.122 | ~99,600 | ~100% |
| **Central** | **72 months (Apr 2028)** | **0.052** | **~84,900** | **85%** |
| Late peak | 84 months (Apr 2029) | 0.033 | ~66,800 | 67% |

### Sensitivity: K (t_inf = 72 fixed)

| K assumption | Implied penetration today | Patients Dec 2030 |
|---|---|---:|
| 80k (bear) | 31% | ~62,000 |
| 100k (central) | 25% | ~85,000 |
| 150k (bull) | 17% | ~139,000 |

**Reading the range:** The central estimate is ~85k by end-2030. The plausible range — excluding the two extreme corners — is roughly **65k–120k**, with the spread driven almost entirely by where we place the inflection and what we take as the ceiling.

---

## 4. Extensions with more time

**Level 1 — tighten the anchor**

The 25k BMS figure is a single point. BMS reports quarterly revenue, which gives a time series of implied patient counts. Fitting the logistic to the full revenue series (months 1–51) rather than a single anchor would resolve t_inf from data rather than judgment, collapsing the dominant uncertainty.

**Level 2 — competitor adjustment**

Aficamten (BMS) is in late-stage development. The logistic here models Camzyos alone as if the ceiling is fixed. A duopoly extension would split K between mavacamten and aficamten share based on trial outcomes and label differentiation, with K itself potentially expanding as competitor entry drives physician awareness and diagnosis rates.

**Level 3 — connect to Task 1**

Task 1 gives the adoption hazard conditional on being in the Diso-conditioned risk set. That hazard, combined with the capture rate (~1 in 9 eligible patients are visible in claims), gives a flow equation: of the ~1,300 new starts/month implied at the 2028 inflection, roughly 140/month would be observable in a claims panel of this type. This is the bridge to Task 3 — the Bayesian update operates on claims-observable counts, not total market counts, so the capture rate is load-bearing.

**Level 4 — full Bayesian formulation**

Rather than fixing K and t_inf as point assumptions, place priors on both (informed by the funnel and comparables), and let the BMS quarterly revenue series update them via a likelihood. The posterior over K and t_inf propagates directly into a posterior over the 2030 projection. This is the proper treatment of parameter uncertainty rather than the scenario grid above.
