# BBB Tech Test — Camzyos Adoption Analysis

Analysis of Camzyos (mavacamten) adoption in US commercial claims data (~30k cardiac patients, 2020–2023). Three tasks:

1. **Adoption modelling** — discrete-time hazard model of Camzyos initiation (Task 1)
2. **TAM estimation** — top-down epidemiological funnel with Monte Carlo propagation (Task 2)
3. **Agentic AI pitch** — 10-minute pitch on building an agentic investment system (Task 3)

## For reviewers — start here

1. **[docs/CASE_STUDY.pdf](docs/CASE_STUDY.pdf)** — primary deliverable. Task 1 + Task 2 write-up with embedded figures, cited references, and reproducible pipeline pointers.
2. **[notebooks/02_camzyos_analysis.ipynb](notebooks/02_camzyos_analysis.ipynb)** — reproducible technical notebook. Runs end-to-end from `synthetic_data/`.
3. **[docs/AGENTIC_AI_PLAN.md](docs/AGENTIC_AI_PLAN.md)** — Task 3 pitch.

## Setup

Requires **Python 3.13**. On macOS: `brew install python@3.13`.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install    # ruff lint + format on every commit
```

The raw data ships as `synthetic_data.zip`. Unzip once:

```bash
unzip -o synthetic_data.zip
```

## Running the analysis

Every script is self-contained and reads directly from `synthetic_data/`.

**Task 1 — adoption model.** Scripts are numbered in the order they should be run.

```bash
.venv/bin/python scripts/task1_adoption/01_adoption_model.py     # main pipeline + benchmarks
.venv/bin/python scripts/task1_adoption/02_sensitivity.py        # robustness across cohort / censoring / features
.venv/bin/python scripts/task1_adoption/03_adoption_figure.py    # four-panel adoption figure (case study)
.venv/bin/python scripts/task1_adoption/04_calibration.py        # reliability + subgroup calibration
.venv/bin/python scripts/task1_adoption/05_hr_forest_plot.py     # hazard ratio forest plot (case study)
```

**Task 2 — top-down TAM funnel.** Single script; asserts every rounded-k value in the case study.

```bash
.venv/bin/python scripts/task2_tam/02_top_down_funnel.py
```

The script runs the Monte Carlo, saves three CSVs and one PNG to `outputs/task2_tam/`, and then runs 17 hard-coded sanity checks against the rounded-k values in `docs/CASE_STUDY.md`. Any drift between code and doc fails the script with an `AssertionError`.

**Tests.**

```bash
.venv/bin/python -m pytest tests/
```

## Rebuilding the case study PDF

**Use `scripts/build_pdf.sh` only.** Editor / browser / VSCode Markdown PDF export or vanilla `pandoc` produce a sans-serif, wide-margin layout that does not match the reviewer-facing format.

```bash
scripts/build_pdf.sh
```

Prerequisites:
- `pandoc ≥ 3.0` — `brew install pandoc`
- Google Chrome at the default macOS install path (`/Applications/Google Chrome.app`)

Output: `docs/CASE_STUDY.pdf` (~800 KB). If you see a ~500 KB sans-serif file, the wrong tool was used — rerun via `scripts/build_pdf.sh`. Two CSS quirks are load-bearing (documented inline in [scripts/build_pdf.sh](scripts/build_pdf.sh)).

## Repository layout

```
src/
  task1_adoption/          — Task 1 modules
    config.py              — Clinical codes, feature sets, dataclass configs
    data_loading.py        — Load raw CSVs into typed RawData
    dataset.py             — Person-month dataset (risk set, censoring, features)
    models.py              — DiscreteHazardGLM, GBMHazardBenchmark, MarginalRateModel
    evaluation.py          — Brier decomposition, time-dependent AUC, calibration
    selection.py           — StabilitySelector (bootstrap + L1)
  task2_tam/
    top_down.py            — FunnelParams, MC over triangular priors, plot helpers

scripts/
  task1_adoption/          — Numbered Task 1 pipeline scripts (01–05)
    experiments_feature_selection.py  — Sensitivity of the stability-selection matrix
  task2_tam/
    02_top_down_funnel.py  — Driver + 17 sanity-check assertions against docs/CASE_STUDY.md
  build_pdf.sh             — Build docs/CASE_STUDY.pdf (pandoc + Chrome headless)

outputs/
  task1_adoption/          — Figures and CSVs from Task 1
  task2_tam/               — Figures and CSVs from Task 2

tests/                     — Concise unit tests for src/ modules
notebooks/
  01_eda.ipynb             — EDA: cohort identification, Camzyos fill patterns
  02_camzyos_analysis.ipynb — Reproducible technical report (Task 1 + Task 2)

docs/
  CASE_STUDY.md / .pdf     — Primary deliverable
  AGENTIC_AI_PLAN.md       — Task 3 pitch
  CANDIDATE_BRIEF.md       — Original take-home spec
  references.bib           — BibTeX used by the PDF build

synthetic_data/            — Input CSVs (patients, diagnoses, procedures, prescriptions, enrollment)
```

## Headline results

**Task 1 — adoption model.** 4-feature discrete-time hazard GLM on 775 Disopyramide-experienced patients (146 initiators). Test-set time-dependent AUC **0.72** (95% CI [0.67, 0.78]), count calibration MAE **2.3 patients/month**, Hosmer–Lemeshow p = 0.21. Treatment-escalation history (`ccb_ever` HR **5.40**, 95% CI 2.64–11.06, p < 0.001) dominates; demographics and symptom-burden codes are not predictive. See [CASE_STUDY.md § Task 1](docs/CASE_STUDY.md).

**Task 2 — top-down TAM.** `TAM = US adults × diagnosed HCM prevalence × Camzyos-eligible fraction`, each factor a triangular distribution over directly-cited published bounds with Monte Carlo propagation.

- **US addressable market, 2026** — MC median **~127k patients**, 80% CI **~84k – 195k**
- **2030 outlook** — anchored at the MC median and projected under 2%/4.7%/7.4%/yr growth: **~137k – 169k**, base case **~152k**
- **Prevalence anchors** — Husser 2018 (Germany 2015, 70/100k floor), Butzner 2021 (US 2019, 80/100k mode), Massera 2023 (imaging-phenotype ceiling, 200/100k)
- **Eligibility anchors** — Schultze 2022 / Batzner 2019 (obstructive share ~half-to-two-thirds) × Butzner 2026 / Wang 2023 / Charron 2026 (NYHA II-III symptomatic share)
- Every headline number is script-asserted (17 checks) against [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md)

See [CASE_STUDY.md § Task 2](docs/CASE_STUDY.md) and [outputs/task2_tam/](outputs/task2_tam/).
