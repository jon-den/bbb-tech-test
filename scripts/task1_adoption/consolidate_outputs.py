#!/usr/bin/env python
"""Consolidate Task 1 structured outputs into two files.

Reads all per-script CSVs/JSONs from outputs/task1_adoption/ and produces:
  - task1_outputs.xlsx  — all tabular data as named sheets
  - task1_scalars.json  — all scalar KPIs merged

Then removes the ten intermediate files.

Run after all four analysis scripts have completed:
  .venv/bin/python scripts/task1_adoption/consolidate_outputs.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

OUT = Path("outputs/task1_adoption")

# ── 1. Tabular sheets ─────────────────────────────────────────────────────────

SHEETS = [
    ("model_comparison", "01_model_comparison.csv"),
    ("coefficients", "01_coef_refined.csv"),
    ("hazard_ratios", "05_hr_table.csv"),
    ("archetypes", "03_archetypes.csv"),
    ("monthly_predictions", "03_monthly_predictions.csv"),
    ("reliability", "04_reliability.csv"),
    ("monthly_calibration", "04_monthly_calibration.csv"),
    ("subgroup_calibration", "04_subgroup_calibration.csv"),
]

xlsx_path = OUT / "task1_outputs.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    for sheet_name, filename in SHEETS:
        csv_path = OUT / filename
        if not csv_path.exists():
            print(f"  MISSING — skipping sheet '{sheet_name}' ({filename})")
            continue
        df = pd.read_csv(csv_path)
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  Sheet '{sheet_name}' ← {filename}  ({len(df)} rows)")

print(f"\nSaved: {xlsx_path}")

# ── 2. Scalar KPIs ────────────────────────────────────────────────────────────

scalars: dict = {}
for json_file in ["03_summary.json", "04_hl_test.json"]:
    p = OUT / json_file
    if not p.exists():
        print(f"  MISSING — skipping scalars from {json_file}")
        continue
    scalars.update(json.loads(p.read_text()))
    print(f"  Scalars ← {json_file}")

scalars_path = OUT / "task1_scalars.json"
scalars_path.write_text(json.dumps(scalars, indent=2))
print(f"Saved: {scalars_path}")

# ── 3. Remove intermediates ───────────────────────────────────────────────────

intermediates = [f for _, f in SHEETS] + ["03_summary.json", "04_hl_test.json"]
removed = []
for filename in intermediates:
    p = OUT / filename
    if p.exists():
        p.unlink()
        removed.append(filename)

print(f"\nRemoved {len(removed)} intermediate files.")
print("Done. outputs/task1_adoption/ now has:")
for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(f"  {p.name}")
