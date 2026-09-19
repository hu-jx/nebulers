import numpy as np
from dataclasses import dataclass
from util import parse_timestamps 
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TimestampGapModel:
    """Parameters learned from a calibration stream and frozen for inference."""
    time_column: str
    nominal_interval_seconds: float
    gap_cutoff_seconds: float
    lower_cluster_edge_seconds: float
    upper_cluster_edge_seconds: float


def _positive_time_differences(
    data: pd.DataFrame,
    time_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Chronologically sort a copy and calculate positive adjacent intervals."""

    if data.empty:
        raise ValueError("Cannot process an empty DataFrame.")
    if time_column not in data.columns:
        raise ValueError(f"Missing timestamp column: {time_column!r}")

    frame = data.copy()
    frame["_parsed_time"] = parse_timestamps(frame[time_column])
    frame = frame.sort_values("_parsed_time", kind="stable").reset_index(drop=True)
    datetime_col: pd.Series[pd.Timestamp] = frame["_parsed_time"]
    frame["_time_gap_seconds"] = datetime_col.diff().dt.total_seconds()

    positive = frame.loc[
        frame["_time_gap_seconds"].gt(0), "_time_gap_seconds"
    ]
    if positive.empty:
        raise ValueError("No positive consecutive timestamp differences exist.")
    return frame, positive


def fit_timestamp_gap_model(
    calibration_data: pd.DataFrame,
    *,
    time_column: str = "Datetime",
) -> TimestampGapModel:
    """Learn and freeze the gap cutoff from raw calibration timestamps only."""

    _, positive = _positive_time_differences(calibration_data, time_column)

    # Rounding removes floating-point representations of identical intervals.
    unique = np.unique(np.round(positive.to_numpy(dtype=float), 9))
    if unique.size < 2:
        raise ValueError(
            "Only one positive interval scale was observed; a gap cutoff cannot "
            "be identified from this calibration stream."
        )

    # Ratios compare observed scales without introducing a numeric cutoff.
    consecutive_ratios = unique[1:] / unique[:-1]
    split_index = int(np.argmax(consecutive_ratios))
    lower = float(unique[split_index])
    upper = float(unique[split_index + 1])

    # The geometric midpoint is halfway between the edges in logarithmic space.
    cutoff = float(np.sqrt(lower * upper))

    return TimestampGapModel(
        time_column=time_column,
        nominal_interval_seconds=float(np.median(positive)),
        gap_cutoff_seconds=cutoff,
        lower_cluster_edge_seconds=lower,
        upper_cluster_edge_seconds=upper,
    )


def segment_cycle(
    data: pd.DataFrame,
    model: TimestampGapModel,
    *,
    include_diagnostics: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a previously fitted and frozen timestamp-gap model."""

    original_columns = data.columns.tolist()
    frame, _ = _positive_time_differences(data, model.time_column)

    # Apply the fitted cutoff unchanged; do not refit it on inference data.
    frame["_time_boundary"] = frame["_time_gap_seconds"].gt(
        model.gap_cutoff_seconds
    )
    frame.loc[0, "_time_boundary"] = True

    # Number every row consecutively from segment 1 through segment N.
    frame["segment_id"] = frame["_time_boundary"].cumsum().astype(int)

    # Preserve every raw measurement for the separate status classifier.
    row_columns = [*original_columns, "segment_id"]
    if include_diagnostics:
        row_columns.extend(["_time_gap_seconds", "_time_boundary"])
    segmented_rows = frame[row_columns].copy()

    # Public predictions contain neutral boundaries only.
    segment_boundaries = (
        frame.groupby("segment_id", sort=True)
        .agg(
            start_time=(model.time_column, "first"),
            end_time=(model.time_column, "last"),
        )
        .reset_index()
    )

    metadata = {
        "nominal_interval_seconds": model.nominal_interval_seconds,
        "gap_cutoff_seconds": model.gap_cutoff_seconds,
        "lower_cluster_edge_seconds": model.lower_cluster_edge_seconds,
        "upper_cluster_edge_seconds": model.upper_cluster_edge_seconds,
    }
    segmented_rows.attrs.update(metadata)
    segment_boundaries.attrs.update(metadata)
    return segmented_rows, segment_boundaries

def eval_segment(
    true_segments: pd.DataFrame,
    pred_segments: pd.DataFrame,
):

    if len(true_segments) != len(pred_segments):
        raise ValueError(
            f"Expected equal counts, but received "
            f"{len(true_segments)} true segments and "
            f"{len(pred_segments)} predicted segments."
        )

    true_segments = true_segments.reset_index(drop=True)
    pred_segments = pred_segments.reset_index(drop=True)

    true_start_vals = parse_timestamps(true_segments["start_time"])
    true_end_vals = parse_timestamps(true_segments["end_time"])
    pred_start_vals = parse_timestamps(pred_segments["start_time"])
    pred_end_vals = parse_timestamps(pred_segments["end_time"])

    iou_scores = []

    for true_start, true_end, pred_start, pred_end in zip(
        true_start_vals,
        true_end_vals,
        pred_start_vals,
        pred_end_vals,
    ):
        overlap = (
            min(true_end, pred_end)
            - max(true_start, pred_start)
        )

        intersection = max(pd.Timedelta(0), overlap)

        union = (
            (true_end - true_start)
            + (pred_end - pred_start)
            - intersection
        )

        iou = (
            0.0
            if union <= pd.Timedelta(0)
            else float(intersection / union)
        )

        iou_scores.append(iou)

    return sum(iou_scores) / len(iou_scores)