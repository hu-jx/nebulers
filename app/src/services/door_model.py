import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


TIME_COLUMN = "Datetime"
NORMAL_LABEL = "Normal"
ABNORMAL_LABEL = "Abnormal resistance"
NUMBER_PHASE_BINS = 5

SIGNAL_COLUMNS = {
    "current": "Motor current(mA)",
    "voltage": "Motor Voltage(10mV)",
    "back_emf": "Motor electrodynamic force",
    "position": "Door leaf position",
}

REQUIRED_COLUMNS = {
    TIME_COLUMN,
    "Open command",
    "Close command",
    "Door is opening",
    "Door is closing",
    "Door leaf position",
    "Door Opened",
    "DCSR",
    "DCSL",
    "DLSR",
    "DLSL",
    *SIGNAL_COLUMNS.values(),
}


# kept here so the saved joblib can resolve segment.TimestampGapModel
@dataclass(frozen=True)
class TimestampGapModel:
    time_column: str
    nominal_interval_seconds: float
    gap_cutoff_seconds: float
    lower_cluster_edge_seconds: float
    upper_cluster_edge_seconds: float


# the saved door model expects this class to live in a module called segment
sys.modules["segment"] = sys.modules[__name__]


# parses the seven-part timestamps used by the door dataset
def parse_timestamps(values):
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="raise")

    parts = values.astype(str).str.split("-", expand=True)

    if parts.shape[1] != 7:
        return pd.to_datetime(values, errors="raise")

    numeric = parts.apply(
        pd.to_numeric,
        errors="raise",
    )

    base = pd.to_datetime({
        "year": numeric[0],
        "month": numeric[1],
        "day": numeric[2],
        "hour": numeric[3],
        "minute": numeric[4],
        "second": numeric[5],
    })

    return base + pd.to_timedelta(
        numeric[6],
        unit="ms",
    )


# applies the saved timestamp-gap model to split a stream into door cycles
def segment_cycle(dataframe, model):
    frame = dataframe.copy()

    if frame.empty:
        raise ValueError("door input is empty")

    frame["_parsed_time"] = parse_timestamps(
        frame[model.time_column]
    )

    frame = frame.sort_values(
        "_parsed_time",
        kind="stable",
    ).reset_index(drop=True)

    frame["_time_gap_seconds"] = (
        frame["_parsed_time"]
        .diff()
        .dt.total_seconds()
    )

    frame["_time_boundary"] = (
        frame["_time_gap_seconds"]
        .gt(model.gap_cutoff_seconds)
    )

    frame.loc[0, "_time_boundary"] = True

    frame["segment_id"] = (
        frame["_time_boundary"]
        .cumsum()
        .astype(int)
    )

    original_columns = [
        column
        for column in dataframe.columns
        if column != "source_file"
    ]

    segmented_rows = frame[
        [*original_columns, "segment_id"]
    ].copy()

    boundaries = (
        frame.groupby(
            "segment_id",
            sort=True,
        )
        .agg(
            start_time=(model.time_column, "first"),
            end_time=(model.time_column, "last"),
        )
        .reset_index()
    )

    return segmented_rows, boundaries


# gets whole-cycle stats for one signal
def whole_cycle_statistics(values, prefix):
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    finite = numeric[
        np.isfinite(numeric)
    ]

    if finite.size == 0:
        return {
            f"{prefix}__mean": np.nan,
            f"{prefix}__std": np.nan,
            f"{prefix}__min": np.nan,
            f"{prefix}__max": np.nan,
        }

    return {
        f"{prefix}__mean": float(np.mean(finite)),
        f"{prefix}__std": float(np.std(finite, ddof=0)),
        f"{prefix}__min": float(np.min(finite)),
        f"{prefix}__max": float(np.max(finite)),
    }


# summarizes motor current across equal phases of a cycle
def phase_binned_current(values):
    current = pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)

    number_rows = len(current)

    if number_rows == 0:
        return {
            f"current__phase_{index + 1}_mean": np.nan
            for index in range(NUMBER_PHASE_BINS)
        }

    bin_ids = np.floor(
        np.arange(number_rows)
        * NUMBER_PHASE_BINS
        / number_rows
    ).astype(int)

    bin_ids = np.minimum(
        bin_ids,
        NUMBER_PHASE_BINS - 1,
    )

    features = {}

    for bin_index in range(NUMBER_PHASE_BINS):
        values_in_bin = current[
            bin_ids == bin_index
        ]

        finite = values_in_bin[
            np.isfinite(values_in_bin)
        ]

        features[
            f"current__phase_{bin_index + 1}_mean"
        ] = (
            float(np.mean(finite))
            if finite.size
            else np.nan
        )

    return features


# converts each segmented cycle into the classifier feature row
def extract_cycle_features(segmented_rows, time_column):
    required = {
        "segment_id",
        time_column,
        *SIGNAL_COLUMNS.values(),
    }

    missing = sorted(
        required.difference(
            segmented_rows.columns
        )
    )

    if missing:
        raise ValueError(
            f"door input is missing classification columns: {missing}"
        )

    records = []

    for segment_id, unsorted_segment in segmented_rows.groupby(
        "segment_id",
        sort=True,
    ):
        segment = unsorted_segment.copy()

        segment["_parsed_time"] = parse_timestamps(
            segment[time_column]
        )

        segment = segment.sort_values(
            "_parsed_time",
            kind="stable",
        )

        duration = float(
            (
                segment["_parsed_time"].iloc[-1]
                - segment["_parsed_time"].iloc[0]
            ).total_seconds()
        )

        position = pd.to_numeric(
            segment[SIGNAL_COLUMNS["position"]],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite_position = position[
            np.isfinite(position)
        ]

        position_change = (
            float(
                finite_position[-1]
                - finite_position[0]
            )
            if finite_position.size
            else np.nan
        )

        record = {
            "segment_id": int(segment_id),
            "duration_seconds": duration,
            "n_rows": int(len(segment)),
            "position_change": position_change,
            "operation__Open": float(
                np.isfinite(position_change)
                and position_change > 0
            ),
            "operation__Close": float(
                np.isfinite(position_change)
                and position_change < 0
            ),
        }

        for prefix, column in SIGNAL_COLUMNS.items():
            record.update(
                whole_cycle_statistics(
                    segment[column],
                    prefix,
                )
            )

        record.update(
            phase_binned_current(
                segment[SIGNAL_COLUMNS["current"]]
            )
        )

        records.append(record)

    return pd.DataFrame.from_records(
        records
    )


# wraps the saved door bundle so the rest of the app can call .predict()
class DoorModel:
    def __init__(self, bundle):
        required = {
            "gap_model",
            "classifier_bundle",
        }

        missing = required - set(bundle)

        if missing:
            raise ValueError(
                f"door model bundle is missing: {sorted(missing)}"
            )

        self.gap_model = bundle["gap_model"]
        self.classifier_bundle = bundle["classifier_bundle"]

    # returns one classified row per detected door cycle
    def predict(self, dataframe):
        if "source_file" not in dataframe.columns:
            raise ValueError(
                "door input is missing source_file"
            )

        outputs = []

        for source_file, group in dataframe.groupby(
            "source_file",
            sort=False,
        ):
            raw = group.drop(
                columns=["source_file"]
            ).reset_index(drop=True)

            missing = sorted(
                REQUIRED_COLUMNS.difference(
                    raw.columns
                )
            )

            if missing:
                raise ValueError(
                    f"{source_file}: missing required columns: {missing}"
                )

            segmented_rows, boundaries = segment_cycle(
                raw,
                self.gap_model,
            )

            features = extract_cycle_features(
                segmented_rows,
                time_column=str(
                    self.classifier_bundle["time_column"]
                ),
            )

            feature_columns = list(
                self.classifier_bundle["feature_columns"]
            )

            missing_features = sorted(
                set(feature_columns).difference(
                    features.columns
                )
            )

            if missing_features:
                raise ValueError(
                    f"{source_file}: missing model features: {missing_features}"
                )

            model = self.classifier_bundle["model"]

            probabilities = model.predict_proba(
                features[feature_columns]
            )[:, 1]

            predicted = (
                probabilities >= 0.5
            ).astype(int)

            output = boundaries[
                [
                    "segment_id",
                    "start_time",
                    "end_time",
                ]
            ].copy()

            output["prediction"] = np.where(
                predicted == 1,
                ABNORMAL_LABEL,
                NORMAL_LABEL,
            )

            outputs.append(
                output[
                    [
                        "start_time",
                        "end_time",
                        "prediction",
                    ]
                ]
            )

        if not outputs:
            return pd.DataFrame(
                columns=[
                    "start_time",
                    "end_time",
                    "prediction",
                ]
            )

        return pd.concat(
            outputs,
            ignore_index=True,
        )
