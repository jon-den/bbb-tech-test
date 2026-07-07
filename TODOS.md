# TODOS

## Completed
- [x] Univariate code screening — FINDINGS F16, EXPERIMENTS E9
- [x] Code review and refactoring — panel.py thematic sub-functions, ATC duplication collapsed, config.py sections/comments, models.py predict threshold fixed
- [x] Full evaluation pipeline + sensitivity analysis (scripts/05)
- [x] Technical write-up (WRITEUP.md)
- [x] Agentic AI plan (AGENTIC_AI_PLAN.md)
- [x] Replace cumulative-incidence C-index with monthly time-dependent AUC — FINDINGS F17
- [x] Tests for StabilitySelector and DiscreteHazardGLM (tests/)
- [x] Fix flat test-set count calibration — replace month dummies with continuous study_month covariate — FINDINGS F18

## Open
(none)

## Completed (Task 1 answer session)
- [x] Task 1 full answer — scripts/07_adoption_answer.py: archetypes, bootstrap CIs, S-curve, risk distribution — FINDINGS F22, F23
- [x] Fix DiscreteHazardGLM._prepare_X single-row bug (sm.add_constant ptp=0) — FINDINGS F23
- [x] Calibration plot — scripts/08_calibration.py → outputs/08_calibration.png (reliability diagram, monthly calibration, subgroup calibration)
- [x] WRITEUP.md: corrected stale numbers (HR, events, penetration, train/test split), added archetype section, added calibration section, updated model comparison table to time-dep. AUC

## Completed (latest)
- [x] DataFrame-first model fitting — DiscreteHazardGLM passes DataFrames to statsmodels (column names preserved), CLAUDE.md updated
- [x] Include I422/I429 patients in Disopyramide risk set — HCM_ELIGIBILITY_CODES, FINDINGS F21, panel imports fixed
- [x] Model report card — scripts/06_model_report_card.py (cohort, features, coefficients, metrics, Cox check, limitations)
- [x] models.py: add GBM benchmark (HistGradientBoostingClassifier) as nonlinear comparison vs GLM
- [x] models.py: CoxPH cross-check via lifelines (CoxTimeVaryingFitter) to validate person-month discretization
- [x] Document index-date and feature-selection-isolation limitations — FINDINGS F19, F20
- [x] Confirmed time_dependent_auc ≠ sksurv.metrics.cumulative_dynamic_auc — not replaceable (see below)
