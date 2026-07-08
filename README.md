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

Task 2 (TAM PyMC model):
```bash
.venv/bin/python scripts/task2_tam/09_tam_model.py
```

Tests:
```bash
.venv/bin/python -m pytest tests/
```

## Rebuilding the case study PDF

**IMPORTANT — use `scripts/build_pdf.sh` only.** Do not export the PDF from an editor, browser preview, VSCode Markdown preview, or `pandoc` with default settings. Those produce a sans-serif, wide-margin layout that does not match the reviewer-facing format.

```bash
scripts/build_pdf.sh
```

This runs pandoc (Markdown → HTML with `--citeproc` for [docs/references.bib](docs/references.bib) and `--embed-resources` for images) → Chrome headless (HTML → PDF), with a pinned inline CSS that produces the reviewer-facing layout: **Georgia serif, 820px max-width, 9.5pt body, dense abbreviations block, no page headers/footers**.

Prerequisites:
- `pandoc ≥ 3.0` — install with `brew install pandoc`
- Google Chrome at the default macOS install path (`/Applications/Google Chrome.app`)

Output: `docs/CASE_STUDY.pdf` (~800–900 KB).

If the output is smaller (~500 KB), sans-serif, or has very wide margins, the script was not used — regenerate via `scripts/build_pdf.sh`.

Two CSS quirks that are load-bearing (see comments in [scripts/build_pdf.sh](scripts/build_pdf.sh)):

1. The tempfile suffix **must** be `.html` — Chrome refuses to parse `<style>` in files with unknown extensions and renders the CSS as body text.
2. The CSS is a **single line** in the `header-includes` variable — pandoc's insertion into `<head>` breaks with multi-line content and the `<style>` block leaks into page 1.

Edit the CSS in [scripts/build_pdf.sh](scripts/build_pdf.sh) if the format needs to change; don't add a separate stylesheet.

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
    tam_model.py           — PyMC joint Bayesian model (p_true × s_capture × N_hcm)
    claims_evidence.py     — Beta-Binomial evidence definitions on the ~30k cohort

scripts/
  task1_adoption/          — Numbered Task 1 pipeline scripts (03–10)
  task2_tam/               — Task 2 pipeline (09_tam_model.py)
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
| [docs/AGENTIC_AI_PLAN.md](docs/AGENTIC_AI_PLAN.md) | Task 3 pitch |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Chronological experiment log |
| [docs/CANDIDATE_BRIEF.md](docs/CANDIDATE_BRIEF.md) | Original take-home assessment spec |
| [docs/QUESTIONS_FOR_BBB.md](docs/QUESTIONS_FOR_BBB.md) | Open questions for the interviewer |
| [docs/references.bib](docs/references.bib) | BibTeX file used by `--citeproc` when rebuilding the PDF |

## Headline results

**Task 1 — adoption model.** 6-feature discrete-time hazard GLM on 775 Disopyramide-experienced patients (146 initiators). Test-set time-dependent AUC 0.72 (95% CI [0.67, 0.78] from patient-level test bootstrap), count calibration MAE 1.7 patients/month. Treatment-escalation history (`ccb_ever` HR 4.69, p<0.01) dominates; demographics and symptom-burden codes not predictive. See [CASE_STUDY.md § Task 1](docs/CASE_STUDY.md).

**Task 2 — TAM.** Joint PyMC Bayesian model reconciles literature (~30% of diagnosed HCM adults are Camzyos-eligible, Desai 2022) with the claims cohort (~4% look treatable using the Task 1 escalation markers) via a claims-capture-rate parameter.
- **US addressable market** — posterior mean **~120,000 patients**, 80% CI **74k–192k**
- **True clinical eligibility `p`** — 30.4% (22–39%)
- **Claims capture rate `s`** — 14.0% (10–18%) — i.e., our billing data sees roughly 1 in 7 truly eligible patients; this is the direct Task 3 hand-off
- Top sensitivity lever: diagnosed HCM count `N` (halving `N` roughly halves the TAM)

See [CASE_STUDY.md § Task 2](docs/CASE_STUDY.md) and [outputs/task2_tam/](outputs/task2_tam/).
