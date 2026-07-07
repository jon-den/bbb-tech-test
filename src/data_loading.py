"""Load raw data files into a unified namespace."""

from dataclasses import dataclass

import pandas as pd

from src.config import DATA_DIR


@dataclass
class RawData:
    """Typed namespace for raw claims data loaded from synthetic_data/."""

    patients: pd.DataFrame
    diagnoses: pd.DataFrame
    procedures: pd.DataFrame
    prescriptions: pd.DataFrame
    enrollment: pd.DataFrame


def load_data(data_dir=DATA_DIR) -> RawData:
    """Load all raw CSV files from data_dir into a RawData namespace.

    Args:
        data_dir (Path): Directory containing the synthetic claims CSVs.
            Defaults to DATA_DIR from config (synthetic_data/).

    Returns:
        RawData: Dataclass with DataFrames for patients, diagnoses, procedures,
            prescriptions, and enrollment. Date columns are parsed automatically.
    """
    return RawData(
        patients=pd.read_csv(data_dir / "patients.csv"),
        diagnoses=pd.read_csv(data_dir / "diagnoses.csv", parse_dates=["date"]),
        procedures=pd.read_csv(data_dir / "procedures.csv", parse_dates=["date"]),
        prescriptions=pd.read_csv(data_dir / "prescriptions.csv", parse_dates=["date"]),
        enrollment=pd.read_csv(data_dir / "enrollment.csv"),
    )
