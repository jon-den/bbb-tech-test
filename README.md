# BBB Tech Test — Camzyos Adoption Analysis

Analysis of Camzyos (mavacamten) adoption in US commercial claims data (~30k cardiac patients, 2020–2023). Three tasks:

1. **Adoption modelling** — discrete-time hazard model of Camzyos initiation
2. **TAM estimation** — PyMC Bayesian model of the US addressable pool (literature priors × claims data)
3. **Agentic AI pitch** — 10-minute pitch on building an agentic investment system

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

**Task 1 — adoption model.** Scripts are numbered in the order they should be run; each is self-contained and reads directly from `synthetic_data/`.

```bash
.venv/bin/python scripts/task1_adoption/01_adoption_model.py     # main pipeline + benchmarks
.venv/bin/python scripts/task1_adoption/02_sensitivity.py        # robustness across cohort / censoring / features
.venv/bin/python scripts/task1_adoption/03_adoption_figure.py    # four-panel adoption figure (case study)
.venv/bin/python scripts/task1_adoption/04_calibration.py        # reliability + subgroup calibration
.venv/bin/python scripts/task1_adoption/05_hr_forest_plot.py     # hazard ratio forest plot (case study)
```

**Task 2 — TAM PyMC model.**

```bash
.venv/bin/python scripts/task2_tam/01_tam_model.py
```

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

Output: `docs/CASE_STUDY.pdf` (~800–900 KB). If you see a ~500 KB sans-serif file, the wrong tool was used — rerun via `scripts/build_pdf.sh`. Two CSS quirks are load-bearing (documented inline in [scripts/build_pdf.sh](scripts/build_pdf.sh)).

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
  task2_tam/               — Task 2 modules
    tam_model.py           — PyMC joint Bayesian model (p_true × s_capture × N_hcm)
    claims_evidence.py     — Beta-Binomial evidence from the ~30k cohort

scripts/
  task1_adoption/          — Numbered Task 1 pipeline scripts (01–05)
  task2_tam/               — Task 2 pipeline (01_tam_model.py)
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
references/                — External PDFs cited in the write-up (e.g. Camzyos FDA label)
```

## Headline results

**Task 1 — adoption model.** 6-feature discrete-time hazard GLM on 775 Disopyramide-experienced patients (146 initiators). Test-set time-dependent AUC 0.72 (95% CI [0.67, 0.78] from patient-level test bootstrap), count calibration MAE 1.7 patients/month. Treatment-escalation history (`ccb_ever` HR 4.69, p<0.01) dominates; demographics and symptom-burden codes not predictive. See [CASE_STUDY.md § Task 1](docs/CASE_STUDY.md).

**Task 2 — TAM.** Joint PyMC Bayesian model reconciles literature (~30% of diagnosed HCM adults are Camzyos-eligible, Desai 2022) with the claims cohort (~4% look treatable using the Task 1 escalation markers) via a claims-capture-rate parameter.
- **US addressable market** — posterior median **~118,000 patients**, 80% CI **74k–192k**
- **True clinical eligibility `p`** — 30% (22–39%)
- **Claims capture rate `s`** — 14% (10–18%) — our billing data sees roughly 1 in 7 truly eligible patients; the direct Task 3 hand-off
- Top sensitivity lever: diagnosed HCM count `N` (halving `N` roughly halves the TAM)

See [CASE_STUDY.md § Task 2](docs/CASE_STUDY.md) and [outputs/task2_tam/](outputs/task2_tam/).
