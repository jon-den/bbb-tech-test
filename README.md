# BBB Tech Test — Camzyos Adoption Analysis

Analysis of Camzyos (mavacamten) adoption in US commercial claims data (~30k cardiac patients, 2020–2023). Three tasks: (1) adoption modelling, (2) TAM estimation, (3) agentic AI investment system pitch.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

`pre-commit install` installs git hooks that run **ruff lint + format** on every commit. Hooks auto-fix what they can; the commit is blocked if unfixable issues remain. To run manually against all files: `pre-commit run --all-files`.

## Running the analysis

```bash
# Task 1: adoption model (discrete-time hazard, stability selection, calibration)
python3 scripts/03_adoption_model.py

# Task 1: sensitivity analysis (cohort, risk set, enrollment window variations)
python3 scripts/05_sensitivity_analysis.py

# Feature discovery: univariate code screening across all dx/px/rx codes
python3 scripts/04_univariate_screen.py

# EDA notebook
source .venv/bin/activate && jupyter notebook notebooks/01_eda.ipynb
```

## Project structure

```
src/
  config.py          — All clinical codes, feature sets, dataclass configs
  data_loading.py    — Load raw CSVs into typed RawData namespace
  panel.py           — Person-month panel construction (risk set, censoring, features)
  models.py          — DiscreteHazardGLM (cloglog/logit), MarginalRateModel baselines
  evaluation.py      — Brier score, C-index, calibration plot, count-level calibration
  selection.py       — StabilitySelector (bootstrap + L1), LASSO path plot
  atc.py             — OMOP ATC drug classification with manual fallback

scripts/
  03_adoption_model.py       — Main Task 1 pipeline: panel → models → evaluation
  04_univariate_screen.py    — Exploratory: Fisher/MWU for all codes × windows
  05_sensitivity_analysis.py — Robustness checks: cohort, risk set, enrollment window

notebooks/
  01_eda.ipynb       — EDA: cohort identification, Camzyos fill patterns, enrollment

synthetic_data/      — Input data (patients, diagnoses, procedures, prescriptions, enrollment)
outputs/             — Saved plots and CSVs
```

## Key findings

See [FINDINGS.md](FINDINGS.md) for detailed analytical findings and [EXPERIMENTS.md](EXPERIMENTS.md) for a chronological log of what was tried (including negative results).

**Short version:**
- 149 oHCM+Disopyramide patients initiated Camzyos; 620 in the risk set did not
- Adoption is **prescriber-driven**, not patient-severity-driven: the signal is treatment trajectory (`ccb_ever`, `bb_current`) and specialist engagement (`mri_ever`), not demographics or symptom codes
- Best model: 6-feature refined cloglog GLM (BSS=+0.001, C-index=0.66). Adding features consistently degrades calibration — sample size (102 training events) is the binding constraint

## Technical write-up

See [WRITEUP.md](WRITEUP.md) for full methodology, assumptions, results, limitations, and what would change with more time/data.

## Task 3: Agentic AI pitch

See [AGENTIC_AI_PLAN.md](AGENTIC_AI_PLAN.md) for the 10-minute pitch on building an agentic AI investment system.
