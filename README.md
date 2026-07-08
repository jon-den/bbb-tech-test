# BBB Tech Test — Camzyos Adoption Analysis

Analysis of Camzyos (mavacamten) adoption in US commercial claims data (~30k cardiac patients, 2020–2023). Three tasks:

1. **Adoption modelling** — discrete-time hazard model of Camzyos initiation
2. **TAM estimation** — Monte Carlo estimate of US addressable pool (prior-predictive and posterior-predictive)
3. **Agentic AI pitch** — 10-minute pitch on building an agentic investment system

## For reviewers — start here

1. **[docs/CASE_STUDY.pdf](docs/CASE_STUDY.pdf)** — primary deliverable. Task 1 + Task 2 write-up with embedded figures, cited references, and reproducible pipeline pointers.
2. **[notebooks/02_camzyos_analysis.ipynb](notebooks/02_camzyos_analysis.ipynb)** — reproducible technical notebook. Runs end-to-end from `synthetic_data/`.
3. **[docs/AGENTIC_AI_PLAN.md](docs/AGENTIC_AI_PLAN.md)** — Task 3 pitch.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

`pre-commit install` installs git hooks that run **ruff lint + format** on every commit.

## Running the analysis

Task 1 (adoption modelling):
```bash
.venv/bin/python scripts/task1_adoption/03_adoption_model.py       # main pipeline
.venv/bin/python scripts/task1_adoption/04_univariate_screen.py    # feature discovery
.venv/bin/python scripts/task1_adoption/05_sensitivity_analysis.py # robustness checks
.venv/bin/python scripts/task1_adoption/06_model_report_card.py    # full model characterisation
.venv/bin/python scripts/task1_adoption/07_adoption_answer.py      # adoption dynamics figure
.venv/bin/python scripts/task1_adoption/08_calibration.py          # calibration diagnostics
.venv/bin/python scripts/task1_adoption/10_hr_forest_plot.py       # hazard ratio forest plot
```

Task 2 (TAM Monte Carlo):
```bash
.venv/bin/python scripts/task2_tam/09_tam_monte_carlo.py
```

Tests:
```bash
.venv/bin/python -m pytest tests/
```

Rebuild the case study PDF (pandoc + Chrome headless):
```bash
scripts/build_pdf.sh
```

## Project structure

```
src/
  task1_adoption/          — Task 1 modules
    config.py              — Clinical codes, feature sets, dataclass configs
    data_loading.py        — Load raw CSVs into typed RawData
    panel.py               — Person-month panel (risk set, censoring, features)
    models.py              — DiscreteHazardGLM, GBMHazardBenchmark (survival Cox), MarginalRateModel
    evaluation.py          — Brier decomposition, time-dependent AUC, calibration
    selection.py           — StabilitySelector (bootstrap + L1)
    atc.py                 — OMOP ATC drug classification
  task2_tam/               — Task 2 modules
    priors.py              — Registry of every prior with citation + source_type
    funnel.py              — Two-pool eligibility funnel (Pool A theoretical / Pool B diagnosed)
    diffusion.py           — Logistic penetration curve (used in fan-chart script only)
    revenue.py             — Patients → USD net revenue (fan-chart script only)
    simulation.py          — Monte Carlo orchestrator
    sensitivity.py         — One-at-a-time tornado
    claims_evidence.py     — Beta-Binomial evidence definitions on the ~30k cohort
    bayesian_update.py     — Beta-Binomial conjugate update for the eligible fraction

scripts/
  task1_adoption/          — Numbered Task 1 pipeline scripts (03–10)
  task2_tam/               — Task 2 pipeline (09)
  build_pdf.sh             — Build docs/CASE_STUDY.pdf (pandoc + Chrome headless)

outputs/
  task1_adoption/          — Figures and CSVs from Task 1
  task2_tam/               — Figures and CSVs from Task 2

tests/
  task1_adoption/          — Unit tests for Task 1 modules
  task2_tam/               — Unit tests for Task 2 modules

notebooks/
  01_eda.ipynb             — EDA: cohort identification, Camzyos fill patterns
  02_camzyos_analysis.ipynb — Reproducible technical report (Task 1 + Task 2)

docs/                      — Narrative markdown (see below)
synthetic_data/            — Input CSVs (patients, diagnoses, procedures, prescriptions, enrollment)
references/                — External PDFs referenced in the write-up (e.g., Camzyos FDA label)
```

## Documentation index

| File | Purpose |
|---|---|
| [docs/CASE_STUDY.md](docs/CASE_STUDY.md) / [.pdf](docs/CASE_STUDY.pdf) | Primary deliverable (Task 1 + Task 2) |
| [docs/RESULTS.md](docs/RESULTS.md) | Slide-structured summary (all three tasks) |
| [docs/METHODS.md](docs/METHODS.md) | Methodology defence for the quant reviewer |
| [docs/AGENTIC_AI_PLAN.md](docs/AGENTIC_AI_PLAN.md) | Task 3 pitch |
| [docs/FINDINGS.md](docs/FINDINGS.md) | Numbered analytical findings (F1–…) |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Chronological experiment log |
| [docs/CANDIDATE_BRIEF.md](docs/CANDIDATE_BRIEF.md) | Original take-home assessment spec |
| [docs/QUESTIONS_FOR_BBB.md](docs/QUESTIONS_FOR_BBB.md) | Open questions for the interviewer |
| [docs/references.bib](docs/references.bib) | BibTeX file used by `--citeproc` when rebuilding the PDF |

## Headline results

**Task 1 — adoption model.** 6-feature discrete-time hazard GLM on 775 Disopyramide-experienced patients (146 initiators). Test-set time-dependent AUC 0.72 (95% CI [0.67, 0.78] from patient-level test bootstrap), count calibration MAE 1.7 patients/month. Treatment-escalation history (`ccb_ever` HR 4.69, p<0.01) dominates; demographics and symptom-burden codes not predictive. See [CASE_STUDY.md § Task 1](docs/CASE_STUDY.md).

**Task 2 — TAM.**
- **Pool A (theoretical ceiling)** — prior-predictive median ~182k, 80% CI 123–258k
- **Pool B (diagnosed treatable today)** — prior-predictive median ~120k, 80% CI 74–192k
- **Posterior-predictive Pool B** — ~17k (11–24k) after Bayesian update on the 30k claims cohort — the ~7× gap between prior and posterior quantifies the value of a real claims subscription
- Top sensitivity lever: diagnosed HCM count (swings Pool B by ~123k)

See [CASE_STUDY.md § Task 2](docs/CASE_STUDY.md) and [outputs/task2_tam/](outputs/task2_tam/).
