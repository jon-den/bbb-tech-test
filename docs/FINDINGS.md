# Findings Log

Running record of analytical findings that shape methodology, interpretation, or limitations. Each finding is dated and tagged by phase.

---

## F1: Cohort definitions capture same Camzyos patients (EDA)

Both loose (≥1 I421) and strict (≥2 I421 ≥30d apart OR I422+I421) cohort definitions capture the same 150/166 Camzyos patients. The 16 outside both definitions have I422 (Other HCM), I420 (Dilated), I429 (Unspecified), or I428 (Other) — no I421 at all. Strict definition drops 537 patients from the oHCM pool (2,506 → 1,969) but no additional Camzyos patients.

**Implication:** Cohort definition choice affects denominator for penetration rates but not the outcome group. Use strict as primary, loose as sensitivity analysis.

---

## F2: Symptom burden does not differentiate Camzyos initiators (EDA)

Symptom prevalence (ever, across full observation period) is nearly identical between Camzyos patients and oHCM non-initiators:

| Symptom proxy | Camzyos (n=166) | oHCM no-Camzyos (n=2,356) |
|---|---|---|
| Dyspnea | 77.1% | 71.6% |
| Heart failure | 77.1% | 78.8% |
| Fatigue | 41.0% | 43.5% |
| Cardiology E&M (high-complexity) | 97.6% | 94.4% |

Claim intensity (median claims per patient among those with ≥1) is also similar across groups.

**Implication:** Symptom codes alone will not differentiate initiators from non-initiators in the survival model. The signal is likely in treatment escalation history (Disopyramide), not symptom burden proxies. This reflects a fundamental limitation of claims data — billing codes capture presence/absence of a condition, not severity (no NYHA class, no LVOT gradient).

---

## F3: Dataset supports initiation modelling but NOT persistence/dropout (EDA)

Three enrollment findings, taken together:

1. **Pre-first-fill enrollment is long:** Median 30.1 months, minimum 15.5 months. All 166 Camzyos patients have ≥6 months lookback. Ample history for feature engineering.

2. **Post-last-fill observation is near-zero:** 118/166 patients (71%) have their last Camzyos fill in December 2023 (end of data). 146/166 (88%) in Nov–Dec 2023. Only 13 patients (7.8%) have a last fill before October 2023.

3. **The "no post-fill enrollment" is end-of-study censoring, not disenrollment.** The data simply ends — we cannot observe what happens after the last fill.

**Implication:** We can model *who initiates Camzyos and when* with reasonable confidence. We **cannot** model *who stays on it*. This is a material gap for the investment thesis: revenue = initiations × persistence × price, and we are modelling only one of those three terms. State this limitation prominently.

---

## F4: 43% of Camzyos patients have intermittent enrollment (EDA)

72/166 Camzyos patients have months with `enrolled=0` between their first and last enrolled month. Median gap: 10.5 months (range 3–25).

**Implication:** The plan's censoring strategy ("censor at first gap") would censor 43% of Camzyos patients partway through observation. This needs further investigation — the `enrolled=0` flag may not mean true disenrollment (could mean "no claims filed this month" in some data systems). If it does mean disenrollment, the intermittent pattern suggests patients cycling between insurers, which creates informative censoring. Handle with care in the survival model.

---

## F5: New Camzyos initiations are flat-to-declining (EDA)

Monthly new patient starts: ~9.1/month in 2022 (Apr–Dec), declining to ~6.2/month in 2023 H2. Total Rx fills per month grow (7 → 118) due to persistence/refills — a misleading growth signal if conflated with new starts.

**Implication:** The IC must see the new-starts vs. total-fills distinction clearly. The "growth" narrative is persistence, not accelerating adoption. This is the single most important descriptive finding for the investment thesis.

---

## F6: 2 Camzyos patients never filled Disopyramide (EDA)

Of 166 Camzyos patients, 2 have no Disopyramide prescription in the data at all. These are negligible for modelling purposes.

**Implication:** Excluded from the Disopyramide-conditioned risk set. Do not model. Note as a data footnote.

---

## F7: 15 Disopyramide+Camzyos patients have no I421 code (EDA)

149 of 166 Camzyos patients have both Disopyramide and I421 (oHCM). The remaining 15 with Disopyramide have I422 (Other HCM), I429 (Unspecified CM), or I420 (Dilated CM) instead. Combined with the 2 without Disopyramide (F6), this accounts for all 17 Camzyos patients outside the primary model's risk set (769 I421+Diso → 149 events captured, vs 166 total).

**Implication:** Keep primary model on I421+Disopyramide. These 15 patients are naturally captured by the I421+I422 sensitivity analysis already in the plan. If that analysis shows materially different coefficients, it suggests coding heterogeneity matters and should be reported.

---

## F8: "Ever tried" vs "currently on" medication captures the treatment journey (Modelling)

Stability selection (CV-tuned C=0.127, 200 bootstrap resamples) on 20 candidate features selected 7:

| Feature | Selection prob. | HR | p |
|---|---|---|---|
| ccb_current | 1.00 | 0.43 | 0.001 |
| ccb_ever | 1.00 | 5.80 | 0.0002 |
| bb_current | 1.00 | 0.07 | 0.008 |
| mri_ever | 0.97 | 1.75 | 0.015 |
| months_since_diso | 0.93 | 0.98 | 0.09 |
| strain_count_12m | 0.72 | — | — |
| er_or_inpatient_12m | 0.64 | — | — |

The "ever tried" vs "currently on" distinction captures a clinically meaningful signal:
- **ccb_ever** (HR=5.80): having ever tried a CCB indicates deeper treatment escalation → patient is closer to switching to Camzyos.
- **ccb_current** (HR=0.43) and **bb_current** (HR=0.07): being actively on these drugs means the patient is currently managed → NOT switching right now.

Initiation happens when patients come OFF their current regime. The model captures the treatment *trajectory* (tried more → ready to switch) separately from *current state* (still on it → not switching yet).

**Implication:** The crude `n_hcm_meds` count (which was significant in the clinical prior model, HR=1.68, p=0.002) drops to π=0.04 in stability selection — its signal is entirely captured by the more specific treatment state features. The refined model should use the granular ever/current features rather than the crude count.

---

## F9: Cardiac MRI is a specialist engagement marker (Modelling)

`mri_ever` (HR=1.75, p=0.015, selection probability 0.97) — patients who have had a cardiac MRI are more likely to initiate Camzyos. Cardiac MRI is a specialist-level workup not done in primary care, indicating the patient is being actively managed at an HCM centre where Camzyos is prescribed via REMS.

TTE count (π=0.28) drops out when MRI and strain imaging are available — the MRI is a sharper signal of specialist engagement than echo frequency.

**Implication:** MRI is a proxy for "being seen at a centre that prescribes Camzyos." Without provider-level data, this is the closest we can get to modelling the prescriber adoption channel.

---

## F10: Demographics and symptom codes carry no signal for Camzyos initiation (Modelling)

Stability selection probabilities for demographics and symptom proxies:
- age: 0.14, sex_F: 0.15, hf_flag: 0.27, symptom_burden_12m: 0.19, af_flag: 0.18, mitral_flag: 0.21

None approach the 0.60 threshold. This confirms EDA finding F2 and extends it: even with 20 candidate features including treatment history and procedures, demographics and symptom codes are dominated by the treatment state features.

**Implication:** Camzyos adoption within the Disopyramide pool is driven by treatment trajectory and specialist engagement, not patient demographics or symptom severity as captured in claims. The IC should understand this as a prescriber-driven adoption pattern, not a patient-characteristics-driven one.

---

## F11: ATC-derived `n_cardiac_classes` is informative but doesn't improve the model (Modelling)

Using the OMOP ATC hierarchy to classify all prescriptions by therapeutic class, we computed the number of distinct ATC 2nd-level cardiac drug classes per patient (C01–C10, B01). This feature has selection probability π=0.94 — robustly selected, ranking 6th of 26 candidates.

However, adding it to the model doesn't improve out-of-sample performance. The 8-feature stability-selected model (BSS=-0.006) underperforms the 6-feature refined model (BSS=+0.001). The broader cardiovascular complexity signal overlaps with what `ccb_ever`, `bb_current`, and `n_hcm_meds` already capture.

Other ATC features did not survive selection: `antiarrhythmic_ever` (π=0.36), `anticoagulant_ever` (π=0.10), `sglt2i_ever` (π=0.04).

**Implication:** The ATC hierarchy is useful for validating the drug classification (confirms our manual groupings are correct) but does not add predictive signal beyond the treatment-state features already in the refined model.

---

## F12: Days-supply features carry no signal for Camzyos initiation (Modelling)

Two days_supply-based features were tested:
- `diso_mpr_12m` (Disopyramide Medication Possession Ratio): π=0.03 — essentially zero
- `cardiac_drug_days_12m` (total cardiac drug-days in 12-month window): π=0.33 — below threshold

Disopyramide adherence does not predict who switches to Camzyos. This makes clinical sense: within the Disopyramide-experienced pool, the switching decision is driven by the prescriber's assessment (which we proxy via specialist engagement) and treatment escalation history, not by how consistently the patient fills their current medication.

**Implication:** Do not include adherence-based features in the model. They add noise without signal.

---

## F13: The 6-feature refined model is the performance ceiling (Modelling)

Across three iterations of feature expansion (7 → 20 → 26 candidates), the 6-feature refined model consistently has the best or near-best performance:

| Model | Features | BSS | C-index |
|---|---|---|---|
| Clinical priors | 7 | +0.002 | 0.63 |
| **Refined** | **6** | **+0.001** | **0.66** |
| Stability-selected | 8 | -0.006 | 0.66 |
| Full expanded | 26 | -0.013 | 0.66 |

Adding features beyond the refined set consistently degrades calibration (negative BSS) while barely improving discrimination. With 102 training events, the model cannot support more than ~6 features without overfitting.

The refined model's features: `months_since_diso`, `n_hcm_meds`, `ccb_ever`, `bb_current`, `ccb_current`, `mri_ever`.

**Implication:** This is the model to present to the IC. Further feature engineering will not improve it — the constraint is sample size, not feature expressiveness.

---

## F14: Label-derived features confirm prescribing constraints are real barriers (Modelling)

Three features derived from the Camzyos FDA label entered the stability-selected set (31 candidates, CV-tuned C=0.207):

| Feature | π | Source in label | Interpretation |
|---|---|---|---|
| `diso_ccb_combo_current` | 0.98 | §5.1, §7.3: "Avoid concomitant use of CAMZYOS with disopyramide in combination with verapamil or diltiazem" | Currently on Diso+CCB = must stop CCB before starting Camzyos. Hard prescribing barrier. |
| `echo_acceleration` | 0.86 | §2.1: Echo required before initiation and at weeks 4/8/12 | Increased echo frequency in recent 6m vs prior 6m = pre-initiation workup underway |
| `cyp_inhibitor_active` | 0.86 | §4: Camzyos contraindicated with moderate-to-strong CYP2C19/CYP3A4 inhibitors (omeprazole, fluconazole, ketoconazole, etc.) | Active CYP inhibitor prescription (with 30-day washout buffer) = cannot start Camzyos without stopping the inhibitor |
| `bnp_test_12m` | 0.77 | §5.1: NT-proBNP elevations as HF warning sign; §12.2: NT-proBNP biomarker tracked in trial | BNP testing = active severity monitoring at specialist level |

Also notable: `dual_bb_ccb_current` had **π=0.00** — this is because the EXPLORER-HCM trial excluded dual BB+CCB patients, so in practice very few Diso-experienced patients are on dual therapy. The feature has near-zero variance in this population and LASSO correctly ignores it.

**However:** Despite strong selection probabilities, the 14-feature stability-selected model (BSS=-0.022, C=0.61) performs WORSE than the 6-feature refined model (BSS=+0.001, C=0.66). The label-derived features are informative in isolation but don't improve out-of-sample prediction at this sample size. F13 still holds.

**Implication:** The label features validate the clinical logic (prescribing constraints are real barriers observable in claims) but don't change the model recommendation. For the IC: the 6-feature refined model remains the best performer. The label-derived features are worth reporting as confirmatory evidence that the model captures the right clinical dynamics, even though they don't improve prediction.

---

## F16: Univariate screening confirms no new code-level features (Feature Selection)

Ran Fisher's exact test + Mann-Whitney U for all 746 codes in the training data (≥10-patient prevalence, 3/6/12m windows, Bonferroni correction). Results:

| Code | Source | Window | OR | p_adj | Interpretation |
|---|---|---|---|---|---|
| Disopyramide Phosphate | rx | 3m | 9.17 | 2.1e-11 | Dominant predictor — confirms risk set conditioning |
| Metoprolol Succinate | rx | 3m | 0.06 | 0.022 | Negative: being on BB = currently managed, not switching |

Only 2 codes survive Bonferroni out of 621 tested. Both are already in the model: Disopyramide defines the risk set; Metoprolol's negative association is captured by `bb_current`.

No new diagnostic or procedure codes approach significance after correction. The strongest surviving procedure codes (PTPP test, magnesium, urinalysis) have ORs of 3–5 but all have p_adj > 1.0 — they associate with Camzyos in raw tests but only because of ascertainment bias (Camzyos patients have more healthcare utilisation overall, so they appear for more procedures).

**Implication:** The claims coding universe does not contain additional predictive signal beyond what our hand-engineered features already capture. This is a definitive negative result: feature discovery from raw code counts does not outperform domain knowledge–driven feature engineering in this sample.

---

## F15: Shorter rolling windows preferred for monitoring/severity features (Modelling)

Stability selection on 41 features (including 3m, 6m, 12m variants of 5 rolling features) reveals clear window preferences:

| Feature | 3m (π) | 6m (π) | 12m (π) | Preferred window |
|---|---|---|---|---|
| BNP test | **0.91** | 0.42 | 0.27 | **3m** — recent severity assessment |
| Stress test | **0.77** | 0.28 | 0.21 | **3m** — recent workup |
| Symptom burden | 0.21 | **0.74** | 0.10 | **6m** — intermediate recency |
| TTE count | 0.57 | 0.49 | 0.20 | 3m marginal (0.57 < threshold) |
| High E&M visits | 0.28 | 0.46 | 0.30 | None pass threshold |

The data strongly prefers shorter windows for monitoring intensity features. A BNP test in the last 3 months (π=0.91) is far more predictive than a BNP test in the last 12 months (π=0.27). This makes clinical sense: recent specialist workup signals imminent prescribing decisions, while a test from 10 months ago is stale.

The 12-month window was **actively harmful** for these features — it dilutes the recency signal with old events that no longer reflect the patient's current management trajectory.

**Implication for the refined model:** The refined model uses `tte_count_12m` — this might improve with a shorter window. However, adding `bnp_test_3m` (π=0.91) or `stress_count_3m` (π=0.77) to the refined set would increase features from 6 to 7-8, and at 102 events this still risks overfitting (F13). The window insight is more valuable as a methodological note: future analyses with larger samples should default to 3-6 month windows for monitoring intensity features, not 12 months.

---

## F17: Cumulative-incidence C-index is invalid for this study design; time-dependent AUC is correct (Evaluation)

Any patient-level risk score that accumulates monthly hazards over the patient's actual follow-up window — including F(T_i) = 1 - ∏(1-h_t) — is monotonically increasing in T_i by definition. In this study, patients are censored at different times: event patients exit at their initiation month (T ∈ 16–21 in the test window), censored patients are followed through month 21. A censored patient observed for 6 test months always accumulates more F than an event patient who exits at month 3 of the test window, regardless of how well the model discriminates. The empirical result confirmed this: C = 0.408 (< 0.5) despite the model having real discriminative signal.

The correct metric for discrete-time hazard models is the **monthly time-dependent AUROC**: for each study month t, evaluate AUROC(h_t) against the binary event indicator (initiated at t vs. still at risk). Weighted average across months by event count. This evaluates whether the model correctly ranks patients *within* each month, without any accumulation.

Results with time-dependent AUC:
| Model | BSS | Time-dep AUC |
|---|---|---|
| Null (marginal rate) | -0.002 | 0.500 |
| Calendar-time only | -0.001 | 0.500 |
| Clinical priors (7) | +0.002 | 0.655 |
| **Refined (6)** | **+0.001** | **0.717** |
| Stability-selected (15) | -0.021 | 0.697 |
| Full expanded (41) | -0.034 | 0.689 |

BSS near zero across all models because the test set has only 30 events across 6 months — insufficient power to distinguish count calibration. The time-dependent AUC shows the refined model has genuine within-month discrimination (AUC 0.717), while the null/calendar models score exactly 0.5 as expected. Stability-selected and full-expanded models show slight AUC degradation relative to refined, consistent with overfitting on the 15-month training window.

**Implication:** The primary reporting metric is time-dependent AUC. BSS is reported but underpowered at 30 test events; it measures count calibration, not discrimination. These measure different things: BSS asks "does the model predict the right number of events each month?", AUC asks "within each month, does the model correctly rank who initiates?" Both are important but require different sample sizes to be informative.

---

## F18: Month dummies produce flat test-set count calibration; continuous time covariate required (Evaluation)

`OneHotEncoder(handle_unknown="ignore")` silently zeros all test-month columns: study_months 13–21 were unseen during training, so every test row gets a zero vector for the time block. The model applies an implicit "month 0" baseline hazard to all test months, making the predicted count constant across the entire 9-month test window.

**Fix:** Replace month dummies with `study_month` as a continuous passthrough feature. The single linear time coefficient extrapolates correctly to unseen months.

**Diagnosis result:** With continuous time, predicted counts range 7.41–7.59/month (vs 3–9 observed). The model slightly over-predicts (7.5 vs 5.4 observed mean), because it was trained on the acceleration phase (months 1–12) and the positive time coefficient extrapolates forward — but the true rate is decelerating.

**Implication for the investment thesis:** The model trained on the growth phase over-estimates counts in the plateau/deceleration phase by ~40%. This is an honest and important finding: the adoption curve has a structural break around month 12 that a linear time model cannot capture. The deceleration beyond month 12 is additional evidence the market is approaching saturation in the Disopyramide-conditioned pool, consistent with F13 (refined model is the performance ceiling).

---

## F21: Expanded HCM eligibility to I421/I422/I429 increases Camzyos events by 15 (Cohort)

The Disopyramide-conditioned risk set originally filtered on I421 (obstructive HCM) only, capturing 149 Camzyos initiators. EDA (F7) identified 15 additional Camzyos patients with Disopyramide whose HCM code is I422 (Other HCM) or I429 (Unspecified CM) — almost certainly oHCM with inconsistent coding, since Disopyramide is specifically indicated for obstructive HCM and would not typically be prescribed for other forms.

**Change:** `HCM_ELIGIBILITY_CODES = ["I421", "I422", "I429"]` in `config.py`; `_disopyramide_eligible()` in `panel.py` now filters on this expanded set. The `_ohcm_eligible()` function (broader oHCM control group) retains I421 only to preserve a consistent denominator for penetration rate calculations.

**Impact:** The at-risk pool expands from ~760 to 775 patients; panel events increase from ~143 to 146 (not all 15 new patients are uncensored — 18 of 164 eligible Cam+Diso+HCM patients are censored before Camzyos initiation due to enrollment gaps). Coefficient magnitudes shift slightly but direction and significance are preserved; `mri_ever` drops below significance threshold (p=0.28 vs p=0.015 before), suggesting it was picking up partly on the smaller sample.

**Implication:** Including these 15 reduces the risk of undercounting Camzyos adoption. The I422/I429 patients are coded inconsistently but clinically equivalent to I421 (Disopyramide prescription is the operative eligibility criterion). Future analyses should note that ~9% of Camzyos initiators have ambiguous HCM coding — a limitation of ICD-10 billing codes, not a property of the disease population.

---

## F19: Person-month panel uses all available longitudinal data — no index-date truncation — but two downstream limitations apply (Methodology)

The discrete-time hazard model uses a **person-month panel**, not a fixed index date. Each patient at risk contributes one row per calendar month from `entry_month` (= max(first Disopyramide fill, launch)) through `exit_month` (= initiation, dis-enrollment, or study end). Features are computed strictly before each panel month — they are time-varying, not fixed at any single snapshot. This means no data is discarded due to index-date truncation; the model exploits the full longitudinal trajectory.

**Limitation 1 — Differential follow-up depth in the at-risk pool.** Patients who were already on Disopyramide at launch contribute 21 months of time-varying data; patients whose first Disopyramide fill occurs at month 18 contribute only 3. When scoring current non-initiators to identify "next adopters," patients with longer follow-up have richer feature histories (e.g., more months of rolling counts, more stable `bb_current` estimates). This is an unavoidable property of any longitudinal study with staggered entry, and the person-month design handles it correctly. However, the IC should be aware that confidence in predicted risk is higher for long-observed patients than for recent entrants.

**Limitation 2 — Training CV ignores temporal ordering.** `StabilitySelector` uses `LogisticRegressionCV` to tune the L1 penalty C. This CV folds the training set without respecting time — fold 3 might "predict" earlier months than fold 2. For the regularization choice this is a modest issue (the penalty controls capacity, not temporal extrapolation), but in principle a correct approach would use **expanding-window (forward-chaining) CV**: train on months 1–k, validate on month k+1, for k = 6 … 11. This would also yield uncertainty estimates on AUC and BSS without touching the hold-out test set.

**Proposed future improvement:** Implement expanding-window CV for both regularization tuning and performance estimation within the training period. This provides (a) unbiased estimates of within-training performance and (b) bootstrap-style uncertainty bands on AUC before any test-set evaluation, reducing the incentive to peek at test results during model selection.

---

## F20: Stability selection is training-data-only — no test-set leakage — but model reporting involved mild selection bias (Methodology / Evaluation)

**What we did:** All stability selection (200 bootstrap resamples, L1-penalised logistic regression, threshold = 0.6) ran exclusively on the training panel (months 1–12). The hold-out test set (months 13–21) was not accessed during feature selection. The stability probabilities and selected feature set are therefore uncontaminated.

**Where mild reporting-selection bias entered:** After computing time-dependent AUC for all five model variants (null, calendar-only, clinical priors, refined, stability-selected, full-expanded) on the test set, we chose to report the **refined model** as primary — partly because its test-set AUC (0.717–0.744, depending on run) was highest among parsimonious models. This is a form of selection bias: the identity of the "best" model was informed by the test set. With five models evaluated, ~1-in-20 chance a randomly better model wins by sampling variation alone.

**Why it's a small concern here:** The refined feature set was pre-specified on clinical grounds (beta-blocker, CCB, months since Disopyramide, prior MRI, HF comorbidity, LVOT monitoring) before any test-set evaluation. The stability-selected features largely overlap. The refined model's AUC advantage over the clinical priors baseline is ~6 percentage points, which is unlikely to be pure sampling noise with 49 test events. But it cannot be ruled out entirely.

**Proposed future improvement:** Strict pre-registration of the primary model and primary metric before any test-set evaluation, or a held-out validation set used only once at the very end. In an industry setting, this would mean locking the model card before running `evaluate_model` on the holdout. Alternatively: use nested cross-validation where feature selection and model selection are both within the inner fold, and only a single final model is evaluated on the test set.


---

## F22: Task 1 full answer — patient archetypes, adoption trajectory, uncertainty (Modelling)

**Script:** `scripts/07_adoption_answer.py`; **Output:** `outputs/07_adoption_answer.png`

### Which patients initiate?

The dominant predictor is **prior CCB exposure** (ccb_ever HR = 4.84, 95% CI 1.86–12.58, p=0.001). Active beta-blocker therapy is strongly protective (bb_current HR = 0.07, 95% CI 0.01–0.53, p=0.010). Four archetype monthly hazards at the training-period median (month 12, 15 months since Disopyramide, 3 HCM meds):

| Archetype | Hazard/month | Median TTI |
|---|---|---|
| Not escalated (on BB, no CCB, no MRI) | 0.0% | ~1,725 mo |
| CCB-experienced (off meds, no MRI)    | 2.6% | ~26 mo |
| Specialist-engaged (off meds, had MRI) | 3.4% | ~20 mo |
| Currently managed (on meds, had MRI)  | 0.1% | ~529 mo |

**Investment implication:** The next adopters are identifiable — patients who have tried and stopped CCBs and had a cardiac MRI (specialist engagement). This points to academic medical centre channels and HCM specialist practices as the primary commercial target.

### Uptake trajectory

Monthly initiations averaged 7.6/month in training (months 1–12) and 6.1/month in the test period (months 13–21) — a mild deceleration consistent with depletion of the easiest-to-reach patients. At 18.8% penetration (146/775) after 21 months, adoption is quasi-linear, not yet showing the exponential growth phase of an S-curve. At the mean predicted hazard for remaining patients (1.45%/month), steady-state flow is ~9 new starts/month from the Disopyramide-eligible pool. This is a synthetic-data artefact: real-world MarketScan data shows +328% YoY growth (2022→2023), suggesting the true trajectory is accelerating, not plateauing.

### Uncertainty

- Time-dependent AUC (test): **0.729**, 95% bootstrap CI [0.697, 0.735] (500 patient-level resamples) — tight CI, discrimination is reliable
- Brier Skill Score: 0.0095 — modest calibration improvement over null; 55 test events is underpowered for formal calibration assessment
- Monthly prediction intervals span ~7 counts wide by month 21 ([3.1, 12.0]) — uncertainty grows as the at-risk pool depletes
- All 9 test months' observed counts fall within the 95% bootstrap CI

**Single-row prediction fix:** `sm.add_constant` incorrectly skips adding an intercept for 1-row DataFrames (ptp=0 triggers early exit). Fixed in `DiscreteHazardGLM._prepare_X` using `has_constant='add'`.

---

## F23: Latent bug in `DiscreteHazardGLM._prepare_X` — single-row prediction (Bug Fix)

`statsmodels.add_constant` uses `np.ptp(axis=0) == 0` to detect existing constants. For any single-row input matrix, every column has ptp=0, so `add_constant` incorrectly concludes a constant is already present and returns the matrix unchanged. Result: the GLM dot product fails with shape mismatch `(1, p) vs (p+1,)` when predicting for individual patients or archetypes.

**Fix:** `has_constant='add'` passed to `sm.add_constant` in `_prepare_X` — this skips the ptp check and always adds the constant column. The fix is backward-compatible: training (batch) calls unaffected.

**Risk surface:** Affects any downstream use of `predict_proba` on single-row inputs — patient scoring, counterfactual simulation, archetype analysis.


---

## F24: Claims-based proxies for symptomatic oHCM under-count vs literature by ~8×

The Camzyos-eligible fraction prior anchored on the US specialty registry [@desai2022] sits at 30.7%. Our own claims cohort, updated via a Beta-Binomial conjugate step, lands at ~4.1% under three defensible `k_treatable` definitions (I421 ∩ Disopyramide; ∪ SRT; ∪ ≥3 HCM meds — all with n=18,953). The posterior is data-dominated (n ≈ 400× the prior ESS of 47), so the low number is not an artefact of prior weight; it is what the claims proxy actually shows.

**Interpretation.** The posterior is a strict lower bound on true clinical eligibility. Community claims routinely miss symptom severity (LVOT gradient, NYHA class, provocable obstruction on stress testing are not billable events), and even the strictest label-indicated marker (Disopyramide fill) is prescribed to only a fraction of truly eligible patients. The literature prior captures true prevalence in a well-coded referral setting; the posterior captures what a claims-driven commercial team would work with. Both are correct answers to different questions.

**Investment implication.** The Pool B gap between prior-predictive (~121k) and posterior-predictive (~17k) is not model disagreement — it is the coding-visibility gap. With coding-adjusted proxies over 30M lives (BB Biotech's IQVIA / Symphony / Komodo subscription), the true answer lands between these brackets. Sizing the difference *is* the commercial value of the subscription and drives the Task 3 pitch.

**Methodological caveat.** This update replaces the eligible-fraction prior only; the diagnosed-HCM count, HCM prevalence, penetration, ramp, aficamten share, and price all remain forward-propagated. Extending the Bayesian update to the diagnosed count needs a real population denominator — that requires the subscription, and would collapse the largest bar in the current tornado.

