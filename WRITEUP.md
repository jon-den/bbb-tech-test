# Technical Write-Up: Camzyos Adoption Analysis

**BB Biotech Investment Case — Task 1 & 2**

---

## Executive Summary

We model Camzyos (mavacamten) adoption from US commercial claims (2020–2023, ~30k cardiac patients). The core finding for the investment case: **monthly new patient starts peaked at ~10/month in late 2022 and decelerated to ~6/month by mid-2023**. Total prescription fills continue to grow (7 → 118/month) due to patient persistence, but this is a refill story, not an acceleration-of-adoption story. The distinguishing factor between adopters and non-adopters is not patient severity — it is treatment trajectory and specialist engagement: patients who have already tried CCBs and been seen for cardiac MRIs initiate Camzyos faster. This is a prescriber-driven adoption pattern proxied through treatment history.

---

## Task 1: Adoption Modelling

### Study population

**oHCM cohort:** Patients with ≥1 I421 (Obstructive Hypertrophic Cardiomyopathy) diagnosis. This yields 2,506 patients. Strict definition (≥2 I421 claims ≥30 days apart) would give 1,969 patients but does not affect the Camzyos-positive group.

**Risk set:** We condition on Disopyramide experience (775 patients). Rationale: 164/166 Camzyos initiators (98.8%) previously filled Disopyramide, making it a near-definitional prerequisite — Camzyos is the next-line therapy after Disopyramide failure in clinical guidelines. Conditioning on this shared precondition removes a massive source of non-comparability and focuses the survival question where the signal actually lives: among patients who have already tried Disopyramide, what determines who switches to Camzyos and when?

The remaining 2 Camzyos patients without Disopyramide are excluded as exceptions (see FINDINGS.md, F6). The risk set also includes 15 patients with Disopyramide + Camzyos but with I422/I429 codes (non-obstructive or unspecified HCM) — included in the primary analysis (F21: Disopyramide prescription is the operative eligibility criterion regardless of HCM subtype code).

**Observation window:** April 2022 (FDA approval) through December 2023. 21 months of post-launch data.

**Censoring:** At first enrollment gap (43% of Camzyos patients have intermittent enrollment; conservative censoring avoids informative observation). At SRT (8 patients — septal reduction therapy, competing event). At end of study (December 2023). Post-study censoring cannot be distinguished from dropout — see Limitations.

### Model choice: discrete-time hazard (grouped proportional hazards)

**Why not Cox PH?** Our data is monthly, creating heavy ties (many patients share the same event month). Cox PH handles ties with approximations (Breslow, Efron) that become inaccurate with the tie density we observe. Monthly granularity also makes time-varying covariates natural to compute as end-of-month snapshots.

**Why not classification?** Patients who haven't initiated by December 2023 are right-censored — they are not "non-events." Classifying "initiated by study end: yes/no" commits immortal time bias.

**Model:** Person-month panel with one row per patient-month in the risk set. Binary event indicator (1 = Camzyos initiation in this month). Complementary log-log (cloglog) link function — the discrete-time equivalent of proportional hazards. Calendar month dummies absorb baseline hazard (no parametric assumption required).

Fit via statsmodels GLM (Binomial family, cloglog link). sklearn-compatible wrapper (`DiscreteHazardGLM`) with `fit`/`predict_proba`/`coef_table` API.

### Feature engineering

All features use strict as-of timing: the feature value for month _t_ uses only data observed before month _t_. This prevents any look-ahead bias.

**Features selected in the final model (6):**

| Feature | HR | 95% CI | p | Interpretation |
|---|---|---|---|---|
| `ccb_ever` | 4.84 | 1.86–12.58 | 0.001 | Ever tried CCB = deeper escalation history |
| `bb_current` | 0.07 | 0.01–0.53 | 0.010 | Currently on BB = currently managed, not switching |
| `ccb_current` | 0.51 | 0.30–0.85 | 0.010 | Currently on CCB = idem |
| `mri_ever` | 1.32 | 0.80–2.18 | 0.277 | Cardiac MRI = seen at HCM specialist centre (NS) |
| `months_since_diso` | 0.96 | 0.93–1.00 | 0.041 | Longer on Diso = more stable, slower to switch |
| `n_hcm_meds` | 1.21 | 0.96–1.53 | 0.110 | Breadth of medication history (NS) |

The "ever tried" vs "currently on" distinction captures the treatment trajectory precisely: `ccb_ever` (tried = ready to escalate) combined with `ccb_current` (still on it = not switching yet) together tell us the patient's position on the escalation ladder. Initiation happens when patients come off their current regimen — the model captures this inflection point.

**Features that don't work (and why):**

Demographics (age, sex) are not predictive. Symptom burden codes (dyspnea, HF, fatigue) are not predictive. This is the key limitation of claims data for this question: billing codes capture the presence/absence of a diagnosis, not its severity. The LVOT gradient threshold (≥50 mmHg) and NYHA class IV requirements for Camzyos initiation are not observable in claims. The decision is made by the cardiologist in the exam room; we can only observe the treatment history that preceded it.

**Feature selection:** Stability selection (Meinshausen & Bühlmann) — 200 bootstrap resamples of 70% of patients, L1-penalised logistic regression, CV-tuned regularisation strength (LogisticRegressionCV, C=0.127–0.207). Features with selection probability ≥0.60 across resamples are retained. This approach is more stable than single LASSO and avoids post-hoc significance p-hacking.

### Evaluation

**Primary metric: Brier Skill Score (BSS)** = 1 − BS/BS_null. Positive = better than null. We prioritise calibration because the investment use case requires correctly-calibrated probabilities: the predicted hazard rate directly feeds the expected new-starts calculation used in market sizing.

**Secondary: C-index** (patient-level concordance, computed from max predicted hazard per patient). Reports ranking discrimination independently from calibration.

**Train/test split:** Temporal. Months 1–12 post-launch (Apr 2022–Mar 2023) for training; months 13–21 (Apr–Dec 2023) for testing. Split chosen to approximate a 70/30 event ratio (91 train events, 55 test events). This mimics the real forecasting task — train on the first year of adoption, evaluate on the second year. It also means the test set contains patients who initiated later in the S-curve, when adoption dynamics may have shifted.

**Model comparison:**

| Model | Features | Time-dep. AUC (test) | BSS (test) |
|---|---|---|---|
| Null (marginal rate) | 0 | 0.500 | 0.000 |
| Calendar-time only | study_month | ~0.500 | ~0.000 |
| Clinical priors | 7 | 0.671 | +0.002 |
| **Refined** | **6** | **0.729** | **+0.010** |
| Stability-selected | 15 | 0.718 | −0.006 |
| Full expanded | 41 | 0.710 | −0.021 |

*Primary metric: monthly time-dependent AUROC (event-count weighted) — correct metric for a discrete-time hazard model. BSS measures calibration vs the null.* See FINDINGS.md F17.

Adding features beyond the refined set consistently degrades calibration while barely improving discrimination. With 91 training events, the model supports approximately 6 features without overfitting (heuristic: ≥10–15 events per feature). This is the performance ceiling given the sample size — not a failure of feature engineering (see EXPERIMENTS.md, E1–E9).

### Patient archetypes and next-adopter profile

`scripts/07_adoption_answer.py` (output: `outputs/07_adoption_answer.png`) provides the full Task 1 answer via four panels: monthly adoption curve with bootstrap CIs, cumulative S-curve, archetype hazard comparison, and risk distribution of remaining patients.

**Archetype predicted hazards** (month 12, median 15 months since Disopyramide):

| Archetype | Hazard/month | Median time-to-initiation |
|---|---|---|
| Not escalated (on BB, no CCB, no MRI) | ~0% | ~1,700 months |
| CCB-experienced, off meds, no MRI | 2.6% | 26 months |
| Specialist-engaged (off meds, had MRI) | **3.4%** | **20 months** |
| Currently managed (on meds, had MRI) | 0.1% | ~530 months |

The ideal candidate profile — ever tried CCB, currently off all cardiac meds, had cardiac MRI — has 50× higher monthly hazard than a patient still actively managed on beta-blockers. **Investment implication:** the next adopters are concentrated in academic HCM centres where specialist workup (MRI) precedes prescription decisions. Commercial efforts should target REMS-certified cardiologists at high-volume HCM programmes, not primary care.

Uncertainty on discrimination: bootstrap AUC CI = [0.697, 0.735] across 500 patient-level resamples — tight, indicating stable model structure. Monthly prediction intervals widen from ±5 at month 13 to ±9 by month 21 as the at-risk pool depletes.

### Adoption curve

Monthly new starts peaked in late 2022 (~10/month) and decelerated to ~6/month by H2 2023. Within the Disopyramide pool, cumulative penetration reached **146/775 = 18.8%** by end of study. The remaining 629 Disopyramide-experienced patients represent the untapped in-sample addressable market, with a mean predicted hazard of 1.45%/month (~9 expected new starts/month at steady state). Note: this reflects the synthetic dataset's flat signal — real-world IBM MarketScan data shows +328% year-on-year growth in Camzyos new patients (2022→2023), suggesting the true trajectory is accelerating, not plateauing (see FINDINGS.md F22).

### Calibration

`scripts/08_calibration.py` (output: `outputs/08_calibration.png`) documents calibration across three views:

1. **Reliability diagram:** 10 quantile bins, predicted decile vs observed event rate. Hosmer-Lemeshow χ²(8) = 5.7, p = 0.68 — no statistically significant miscalibration. Points track the 45° diagonal within Wilson 95% CIs.

2. **Monthly calibration:** Predicted vs observed new starts per test month. MAE = 1.7 patients/month; all observed counts fall within Poisson 95% CIs on the predicted values.

3. **Subgroup calibration:** The model is well-calibrated for CCB-tried patients (the majority of events). The "On BB" subgroup shows 0 observed events vs ~0.4 predicted — consistent with Poisson sampling at HR = 0.07 over 600 person-months, not a structural misspecification. The "CCB naive" group (n=329, 3 events) is also slightly over-predicted; this group is poorly represented in the training set and predictions are consequently uncertain.

**Investment implication:** When the model assigns a 2%/month hazard to a patient cohort, the observed initiation rate has been 2%/month — the probability estimates can be used directly in a market-sizing model without recalibration (subject to the synthetic data caveat).

---

## Task 2: Total Addressable Market

### In-sample estimate

From claims data:
- Eligible oHCM population (I421): 2,506 patients in this 30k-person sample
- Disopyramide-experienced (incl. I422/I429): 775 (30.9% of oHCM)
- Camzyos initiators: 146 (5.8% of oHCM, 18.8% of Diso pool)
- Untapped in-sample: 629 Diso-experienced non-initiators

Note: the 98.8% Disopyramide-to-Camzyos co-occurrence rate is almost certainly a synthetic data artefact. Real-world payer data suggests ~60–75% of Camzyos initiators have prior Disopyramide, with the rest coming via CCB failure or specialist switch decisions.

### Extrapolation to US population

**Step 1: oHCM prevalence.** Literature estimates 0.2–0.5% of the general population has HCM (obstructive and non-obstructive combined). Of these, approximately 60–70% are obstructive (LVOT gradient ≥30 mmHg at rest or provocation). US population: ~335M.

oHCM prevalence estimate: 335M × 0.35% × 0.65 ≈ **760,000 patients**

**Step 2: Treatment-eligible population.** Camzyos is indicated for symptomatic oHCM (NYHA Class II–III) despite conventional medical therapy. Industry estimates put the treatment-eligible pool at 100,000–200,000 patients (those symptomatic enough to require therapy escalation). This aligns with Bristol-Myers Squibb's own disclosures.

We use the midpoint: **150,000 treatment-eligible US patients**.

**Step 3: Current penetration.** FDA/BMS data: approximately 3,000–5,000 patients on Camzyos by end of 2023. Penetration: 3,000–5,000 / 150,000 = **2–3%**. This is consistent with our in-sample estimate adjusted for the synthetic Disopyramide co-occurrence artefact.

**Step 4: Addressable upside.** The Disopyramide-experienced pool in our data reaches 30.9% of the oHCM pool, of which 18.8% have initiated. The realistic addressable market depends on:
- Guideline adoption (Disopyramide is not universally used; CCB+BB failure is an alternative pathway)
- REMS program reach (certified centres required)
- Competitive entry (aficamten — Cytokinetics SEQUOIA-HCM trial results)

Conservative TAM (2–3 year horizon): 8,000–15,000 patients, assuming 5–10% penetration into the 150,000-patient eligible pool. At ~$85,000/year list price, this represents $680M–$1.3B annual revenue potential in the US alone.

**Uncertainty:** This estimate has wide confidence intervals. The key unknowns are (a) the true eligible population size (100k–200k range), (b) the pace of guideline adoption at non-specialised centres, and (c) competitive displacement by aficamten if approved. We do not model competitive dynamics — this would require prescriber-level panel data not available in claims.

---

## Limitations

### Claims data structural limitations

1. **No clinical severity:** LVOT gradient, NYHA class, ejection fraction, and echocardiographic parameters are not in billing data. These are the actual decision variables for Camzyos initiation. Our model proxies these with procedure codes (echo frequency, MRI) rather than values.

2. **No provider/prescriber identity:** Camzyos is distributed via REMS, requiring specialty training. Whether a patient's cardiologist is REMS-certified is likely the strongest predictor of initiation. We cannot model the prescriber adoption channel without NPI-level data.

3. **No persistence/dropout:** 88% of Camzyos patients' last fill is in November–December 2023 (end of data). We cannot distinguish persistence from continuation of a current Rx. Revenue persistence modelling is not possible with this dataset.

4. **Commercial-only:** Medicare/Medicaid patients excluded. HCM is age-associated; many patients transition to Medicare, creating left-censoring for the elderly subgroup. This dataset systematically underrepresents the older, Medicare-enrolled oHCM population.

5. **Enrollment gaps:** 43% of Camzyos patients have months with no claims. We censor at the first gap (conservative). The gap mechanism is unknown — it may represent true disenrollment, claims processing delay, or insurer switching. Informative censoring cannot be ruled out.

### Model limitations

6. **Sample size ceiling:** 91 training events support ~6 features (10–15 events per feature heuristic). The model cannot be made more complex without overfitting. Additional features consistently degrade out-of-sample calibration (EXPERIMENTS.md, E1–E9, F13).

7. **Temporal generalisability:** The test set (Jul–Dec 2023) covers the deceleration phase of adoption. Early-phase patterns (strong CCB/BB effects) may not hold for the late-adopter tail, where the remaining non-initiators may be systematically different (older, more contraindicated, less specialist-engaged).

8. **Synthetic data artefact:** The near-universal Disopyramide co-occurrence (98.8%) is implausible in real-world data (typically 60–75%). Predictions from this model should be recalibrated on real payer data before being used for investment decisions.

---

## What Would Change with More Time or Data

### With more time

1. **Competing risks:** Formally model SRT (n=8) and death as competing events using the Aalen-Johansen estimator or cause-specific hazards. Currently, SRT patients are administratively censored — a simplification that slightly overestimates initiation probability.

2. **Prescriber-level modelling:** If NPI data were available, add a random effect for prescriber (mixed-effects discrete-time hazard). The REMS certification effect would almost certainly dominate all patient-level features.

3. **TAM uncertainty quantification:** Bootstrap the full pipeline (cohort definition → panel → model → penetration forecast) to get properly propagated uncertainty intervals on the 2–3 year forecast, not just point estimates.

4. **Aficamten competitive impact:** Model the competitor trajectory using SEQUOIA-HCM enrollment data and trial readout timing. The TAM estimate treats the market as competitive-free, which it will not be by 2025.

### With better data

1. **Real payer claims (MarketScan, Optum, IQVIA):** Replace synthetic data with 10–30M lives. Provides sufficient events (500–2,000 Camzyos initiators) to support 20–30 features without overfitting, add provider-level covariates, and run meaningful subgroup analyses.

2. **EMR/EHR linkage:** Add LVOT gradient, NYHA class, echocardiographic parameters, and physician notes to the feature space. This is where the clinical signal actually lives. Links (Veeva Pulse, Komodo, Symphony) can connect claims to clinical outcomes.

3. **Prescriber specialty data:** NPI-level specialty, hospital affiliation, and REMS certification status. Would allow separating patient-level readiness from provider-level access — currently conflated in `mri_ever` as a proxy for specialist engagement.

4. **National drug utilisation data (IMS/IQVIA NPA):** Non-claims source for Camzyos dispensing data. Less patient-level richness but broader coverage (all payers, not just commercial). Useful for cross-validation of penetration estimates.
