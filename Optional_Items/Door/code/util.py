import pandas as pd

def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse native seven-part timestamps or an existing datetime Series."""

    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="raise")

    parts = values.astype(str).str.split("-", expand=True)
    if parts.shape[1] != 7:
        return pd.to_datetime(values, errors="raise")

    numeric = parts.apply(pd.to_numeric, errors="raise")
    base = pd.to_datetime(
        {
            "year": numeric[0],
            "month": numeric[1],
            "day": numeric[2],
            "hour": numeric[3],
            "minute": numeric[4],
            "second": numeric[5],
        }
    )
    return base + pd.to_timedelta(numeric[6], unit="ms")