import pandas as pd 
LIMIT_COLUMNS = ["DCSR", "DCSL", "DLSR", "DLSL"]

def check_required_columns(data: pd.DataFrame, time_column: str) -> None:
    """Raise a readable error if a required input column is missing."""

    # The command and motion columns identify the intended operation.
    required = {
        time_column,
        "Open command",
        "Close command",
        "Door is opening",
        "Door is closing",
        "Door leaf position",
        "Door Opened",
        *LIMIT_COLUMNS,
    }

    # Subtract the available columns to obtain the missing names.
    missing = sorted(required.difference(data.columns))

    # Stop immediately rather than producing an incorrect segmentation.
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
