# BBB Tech Test — Camzyos Adoption Analysis

Analysis of Camzyos (mavacamten) adoption in US commercial claims data (~30k cardiac patients, 2020–2023). Three tasks:

1. **Adoption modelling** — discrete-time hazard model of Camzyos initiation
2. **TAM estimation** — Monte Carlo estimate of US addressable pool and revenue trajectory
3. **Agentic AI pitch** — 10-minute pitch on building an agentic investment system

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

`pre-commit install` installs git hooks that run **ruff lint + format** on every commit. To run manually against all files: `pre-commit run --all-files`.

## Running the analysis

Task 1 (adoption modelling):
```bash
.venv/bin/python scripts/task1_adoption/03_adoption_model.py       # main pipeline
.venv/bin/python scripts/task1_adoption/04_univariate_screen.py    # feature discovery
.venv/bin/python scripts/task1_adoption/05_sensitivity_analysis.py # robustness checks
.venv/bin/python scripts/task1_adoption/06_model_report_card.py    # full model characterisation
.venv/bin/python scripts/task1_adoption/07_adoption_answer.py      # IC-facing summary figure
.venv/bin/python scripts/task1_adoption/08_calibration.py          # calibration diagnostics
```

Task 2 (TAM Monte Carlo):
```bash
.venv/bin/python scripts/task2_tam/09_tam_monte_carlo.py
```

Tests:
```bash
.venv/bin/python -m pytest tests/
```

EDA notebook: `jupyter notebook notebooks/01_eda.ipynb`

## Project structure

```
src/
  task1_adoption/          — Task 1 code (adoption modelling)
    config.py              — Clinical codes, feature sets, dataclass configs
    data_loading.py        — Load raw CSVs into typed RawData
    panel.py               — Person-month panel (risk set, censoring, features)
    models.py              — DiscreteHazardGLM, MarginalRateModel baselines
    evaluation.py          — Brier decomposition, C-index, calibration
    selection.py           — StabilitySelector (bootstrap + L1)
    atc.py                 — OMOP ATC drug classification
  task2_tam/               — Task 2 code (TAM Monte Carlo)
    priors.py              — Registry of every prior with citation + source_type
    funnel.py              — Two-pool eligibility funnel (A theoretical / B diagnosed)
    diffusion.py           — Logistic penetration curve, aficamten haircut
    revenue.py             — Patients → USD net revenue
    simulation.py          — Monte Carlo orchestrator
    sensitivity.py         — One-at-a-time tornado

scripts/
  task1_adoption/          — Numbered Task 1 pipeline scripts (03–08)
  task2_tam/               — Task 2 pipeline (09)

outputs/
  task1_adoption/          — Figures and CSVs from Task 1
  task2_tam/               — Figures and CSVs from Task 2

tests/
  task1_adoption/          — Unit tests for Task 1 modules

notebooks/
  01_eda.ipynb             — EDA: cohort identification, Camzyos fill patterns

synthetic_data/            — Input CSVs (patients, diagnoses, procedures, prescriptions, enrollment)
```

## Key findings

See [FINDINGS.md](FINDINGS.md) for detailed analytical findings (F1–F22) and [EXPERIMENTS.md](EXPERIMENTS.md) for a chronological log of what was tried (including negative results).

**Task 1 short version:**
- 149 oHCM+Disopyramide patients initiated Camzyos; 620 in the risk set did not
- Adoption is **prescriber-driven**, not patient-severity-driven: the signal is treatment trajectory (`ccb_ever`, `bb_current`) and specialist engagement (`mri_ever`), not demographics or symptom codes
- Best model: 6-feature refined cloglog GLM (BSS=+0.001, C-index=0.66). Adding features consistently degrades calibration — sample size (102 training events) is the binding constraint

**Task 2 short version:**
- **Pool A** (theoretical US ceiling, symptomatic oHCM whether diagnosed or not): median ~178k, 80% CI 118–255k
- **Pool B** (diagnosed & treatable today): median ~117k, 80% CI 71–188k
- Base-case US Camzyos revenue peaks 2029–2030 at ~$1.6B median (80% CI $0.8–3.0B)
- Biggest lever: `diagnosed_hcm_us_current` — where BB Biotech's IQVIA/Symphony/Komodo data would sharpen the estimate most

## Technical write-up

See [WRITEUP.md](WRITEUP.md) for full methodology, assumptions, results, limitations, and what would change with more time or data (both tasks).

## Task 3: Agentic AI pitch

See [AGENTIC_AI_PLAN.md](AGENTIC_AI_PLAN.md) for the 10-minute pitch on building an agentic AI investment system.
