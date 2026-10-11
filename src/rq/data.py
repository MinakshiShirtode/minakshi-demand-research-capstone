"""Loading, validation and deterministic derived variables."""
import numpy as np
import pandas as pd

REQUIRED = ["date", "demand", "marketing_event", "holiday"]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df["date"].duplicated().any():
        raise ValueError("duplicate dates")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("dates not sorted ascending")
    if not (df["date"].diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("dates are not consecutive daily")
    for c in ("marketing_event", "holiday"):
        if not set(df[c].unique()) <= {0, 1}:
            raise ValueError(f"{c} must be binary 0/1")
    if not pd.api.types.is_numeric_dtype(df["demand"]):
        raise ValueError("demand must be numeric")
    if (df["demand"].dropna() < 0).any():
        raise ValueError("demand must be non-negative")
    return df


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return validate(df)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = validate(df.copy())
    d["dow"] = d["date"].dt.dayofweek
    d["t"] = np.arange(len(d))
    d["week"] = ((d["date"] - d["date"].min()).dt.days // 7 + 1).astype(int)
    both = (d["marketing_event"] == 1) & (d["holiday"] == 1)
    d["day_type"] = np.select(
        [both, d["holiday"] == 1, d["marketing_event"] == 1],
        ["Both", "Holiday", "Event"],
        default="Regular",
    )
    return d
