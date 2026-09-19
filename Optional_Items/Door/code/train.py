# type: ignore
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# from predict import predict_segments
from segment import fit_timestamp_gap_model, segment_cycle
from util import parse_timestamps
from constants import DOOR_MODEL_PATH, NORMAL_LABEL, ABNORMAL_LABEL, SIGNAL_COLUMNS, NUMBER_PHASE_BINS

def _whole_cycle_statistics(values: pd.Series, prefix: str) -> dict[str, float]:
    """Calculate mean/std/min/max for one signal within one cycle."""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
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


def _phase_binned_current(
    values: pd.Series,
    *,
    number_bins: int = NUMBER_PHASE_BINS,
) -> dict[str, float]:
    """Calculate mean current in equal bins of normalized cycle progress due to 
    significant deviation across current in phases when abnormal resistance is discovered."""

    current = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    number_rows = len(current)
    if number_rows == 0:
        return {
            f"current__phase_{index + 1}_mean": np.nan
            for index in range(number_bins)
        }

    # floor(row_index * bins / rows) assigns every row to one equal phase.
    bin_ids = np.floor(
        np.arange(number_rows) * number_bins / number_rows
    ).astype(int)
    bin_ids = np.minimum(bin_ids, number_bins - 1)

    features: dict[str, float] = {}
    for bin_index in range(number_bins):
        bin_values = current[bin_ids == bin_index]
        finite = bin_values[np.isfinite(bin_values)]
        features[f"current__phase_{bin_index + 1}_mean"] = (
            float(np.mean(finite)) if finite.size else np.nan
        )
    return features


def extract_cycle_features(
    segmented_rows: pd.DataFrame,
    *,
    time_column: str = "Datetime",
) -> pd.DataFrame:
    """Convert each variable-length segment into one fixed-width feature row."""

    required = {"segment_id", time_column, *SIGNAL_COLUMNS.values()}
    missing = sorted(required.difference(segmented_rows.columns))
    if missing:
        raise ValueError(f"Missing classification columns: {missing}")

    records: list[dict[str, float | int]] = []
    for segment_id, unsorted_segment in segmented_rows.groupby(
        "segment_id", sort=True
    ):
        segment = unsorted_segment.copy()
        segment["_parsed_time"] = parse_timestamps(segment[time_column])
        segment = segment.sort_values("_parsed_time", kind="stable")

        duration = float(
            (
                segment["_parsed_time"].iloc[-1]
                - segment["_parsed_time"].iloc[0]
            ).total_seconds()
        )
        position = pd.to_numeric(
            segment[SIGNAL_COLUMNS["position"]], errors="coerce"
        ).to_numpy(dtype=float)
        finite_position = position[np.isfinite(position)]
        position_change = (
            float(finite_position[-1] - finite_position[0])
            if finite_position.size
            else np.nan
        )

        # Operation comes only from measured direction, never answer.operation.
        operation_open = float(np.isfinite(position_change) and position_change > 0)
        operation_close = float(
            np.isfinite(position_change) and position_change < 0
        )
        segment_id: str = segment_id
        record: dict[str, float | int] = {
            "segment_id": int(segment_id),
            "duration_seconds": duration,
            "n_rows": int(len(segment)),
            "position_change": position_change,
            "operation__Open": operation_open,
            "operation__Close": operation_close,
        }
        for prefix, column in SIGNAL_COLUMNS.items():
            record.update(_whole_cycle_statistics(segment[column], prefix))
        record.update(
            _phase_binned_current(segment[SIGNAL_COLUMNS["current"]])
        )
        records.append(record)

    return pd.DataFrame.from_records(records)


def attach_status_labels(
    features: pd.DataFrame,
    boundaries: pd.DataFrame,
    answers: pd.DataFrame,
) -> pd.DataFrame:
    """Attach status to exactly matching predicted training intervals."""

    predicted = boundaries.rename(
        columns={
            "start_time": "predicted_start_time",
            "end_time": "predicted_end_time",
        }
    ).copy()
    truth = answers.rename(
        columns={
            "start_time": "true_start_time",
            "end_time": "true_end_time",
        }
    ).copy()
    predicted["_start"] = parse_timestamps(predicted["predicted_start_time"])
    predicted["_end"] = parse_timestamps(predicted["predicted_end_time"])
    truth["_start"] = parse_timestamps(truth["true_start_time"])
    truth["_end"] = parse_timestamps(truth["true_end_time"])

    labelled_boundaries = predicted.merge(
        truth[
            ["_start", "_end", "true_start_time", "true_end_time", "status"]
        ],
        on=["_start", "_end"],
        how="inner",
        validate="one_to_one",
    )
    labelled = features.merge(
        labelled_boundaries[
            [
                "segment_id",
                "predicted_start_time",
                "predicted_end_time",
                "true_start_time",
                "true_end_time",
                "status",
            ]
        ],
        on="segment_id",
        how="inner",
        validate="one_to_one",
    )
    if len(labelled) != len(features):
        raise ValueError(
            "Not every predicted training segment exactly matched an answer: "
            f"matched {len(labelled)} of {len(features)}."
        )
    return labelled.sort_values("segment_id").reset_index(drop=True)

def build_model(*, random_state: int = 42) -> Pipeline:
    """Create the class-balanced Random Forest pipeline."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

def train_classifiers(
    segmented_rows: pd.DataFrame,
    boundaries: pd.DataFrame,
    answers: pd.DataFrame,
    *,
    time_column: str = "Datetime",
    random_state: int = 42,
) -> dict[str, object]:
    """Build features, run repeated CV, then fit Random Forest on all cycles."""

    features = extract_cycle_features(segmented_rows, time_column=time_column)
    labelled = attach_status_labels(features, boundaries, answers)
    feature_columns = [c for c in features.columns if c != "segment_id"]
    X = labelled[feature_columns]
    y = (labelled["status"] == ABNORMAL_LABEL).astype(int)

    fitted_model = build_model(random_state=random_state)
    fitted_model.fit(X, y)
    bundle: dict[str, object] = {
        "format_version": 3,
        "model_type": "random_forest",
        "model": fitted_model,
        "feature_columns": feature_columns,
        "number_phase_bins": NUMBER_PHASE_BINS,
        "time_column": time_column,
        "labels": {0: NORMAL_LABEL, 1: ABNORMAL_LABEL},
    }
    return bundle

def predict_segments(
    segmented_rows: pd.DataFrame,
    boundaries: pd.DataFrame,
    bundle: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every segment without using or filtering through answers."""

    features = extract_cycle_features(
        segmented_rows, time_column=str(bundle["time_column"])
    )
    feature_columns = list(bundle["feature_columns"])
    missing = sorted(set(feature_columns).difference(features.columns))
    if missing:
        raise ValueError(f"Inference features are missing: {missing}")

    model: Pipeline = bundle["model"]
    probability = model.predict_proba(features[feature_columns])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    segment_predictions = features[["segment_id"]].copy()
    segment_predictions["prediction"] = np.where(
        prediction == 1, ABNORMAL_LABEL, NORMAL_LABEL
    )
    segment_predictions["abnormal_probability"] = probability

    output = boundaries[["segment_id", "start_time", "end_time"]].merge(
        segment_predictions,
        on="segment_id",
        how="inner",
        validate="one_to_one",
    )
    if len(output) != len(boundaries):
        raise ValueError("At least one predicted boundary has no feature row.")
    return output, features

def train_final_model(datastream: pd.DataFrame, answers: pd.DataFrame):
    finalGapModel = fit_timestamp_gap_model(
        datastream,
        time_column="Datetime",
    )

    finalSegmentedRows, finalBoundaries = (
        segment_cycle(
            datastream,
            finalGapModel,
        )
    )

    finalModelBundle = train_classifiers(
            segmented_rows=finalSegmentedRows,
            boundaries=finalBoundaries,
            answers=answers,
            random_state=42,
        )

    completeDoorModel = {
        "gap_model": finalGapModel,
        "classifier_bundle": finalModelBundle,
    }

    joblib.dump(
        completeDoorModel,
        DOOR_MODEL_PATH,
    )
