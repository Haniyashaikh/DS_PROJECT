"""
common.py
Shared settings and the data loading / cleaning function.
Every other file (eda.py, train_model.py, app.py) imports from here,
so cleaning is done in ONE place only.
"""

from pathlib import Path
import pandas as pd

# ---------- Paths (relative to this file, so it works on any computer) ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Tetuan City power consumption.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------- Column names we will use ----------
WEATHER_COLS = ["Temperature", "Humidity", "Wind Speed",
                "general diffuse flows", "diffuse flows"]
TIME_COLS = ["Hour", "Day", "Month", "DayOfWeek"]
FEATURES = WEATHER_COLS + TIME_COLS          # inputs of the ML model

ZONE_COLS = ["Zone 1 Power Consumption",
             "Zone 2 Power Consumption",
             "Zone 3 Power Consumption"]
TARGET = ZONE_COLS[0]                        # we predict Zone 1

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    The CSV can have extra spaces (for example 'Zone 2  Power Consumption')
    or different upper/lower case. This function renames every column to
    the exact names written above.
    """
    wanted = ["DateTime"] + WEATHER_COLS + ZONE_COLS
    lookup = {name.lower(): name for name in wanted}

    new_names = {}
    for col in df.columns:
        # remove extra spaces and make lowercase so we can match
        key = " ".join(str(col).split()).lower()
        if key in lookup:
            new_names[col] = lookup[key]
    return df.rename(columns=new_names)


def load_data() -> pd.DataFrame:
    """Read the CSV, clean it, and add time features."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Download 'Tetuan City power consumption.csv' from UCI and put it "
            "inside the 'data' folder."
        )

    df = pd.read_csv(DATA_PATH)
    df = _standardise_columns(df)

    # --- check that all needed columns exist ---
    needed = ["DateTime"] + WEATHER_COLS + ZONE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"These columns are missing in the CSV: {missing}\n"
                         f"Columns found: {list(df.columns)}")

    # --- cleaning ---
    df["DateTime"] = pd.to_datetime(df["DateTime"])   # text -> real date/time
    df = df.drop_duplicates()                         # remove duplicate rows
    df = df.sort_values("DateTime").reset_index(drop=True)  # keep time order

    # --- feature engineering: extract time information ---
    df["Hour"] = df["DateTime"].dt.hour
    df["Day"] = df["DateTime"].dt.day
    df["Month"] = df["DateTime"].dt.month
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek     # 0 = Monday ... 6 = Sunday
    df["DayName"] = df["DayOfWeek"].map(dict(enumerate(DAY_NAMES)))

    return df
