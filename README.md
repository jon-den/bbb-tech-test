# BBB Tech Test — Camzyos Adoption Analysis

Analysis of Camzyos (mavacamten) adoption in US commercial claims data (~30k cardiac patients, 2020–2023). Three tasks:

1. **Adoption modelling** — discrete-time hazard model of Camzyos initiation (Task 1)
2. **TAM estimation** — top-down epidemiological funnel with Monte Carlo propagation (Task 2)
3. **Agentic AI pitch** — 10-minute pitch on building an agentic investment system (Task 3)

## For reviewers — start here

1. **[docs/CASE_STUDY.pdf](docs/CASE_STUDY.pdf)** — primary deliverable. Task 1 + Task 2 write-up with embedded figures, cited references, and reproducible pipeline pointers.
2. **[notebooks/02_camzyos_analysis.ipynb](notebooks/02_camzyos_analysis.ipynb)** — reproducible technical notebook. Runs end-to-end from `synthetic_data/`.

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
.venv/bin/python scripts/task1_adoption/consolidate_outputs.py   # merge per-script CSVs/JSONs → task1_outputs.xlsx + task1_scalars.json
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

Output: `docs/CASE_STUDY.pdf`. Two CSS quirks are load-bearing (documented inline in [scripts/build_pdf.sh](scripts/build_pdf.sh)).

Results and methodology are documented in the case study — see [docs/CASE_STUDY.pdf](docs/CASE_STUDY.pdf).
