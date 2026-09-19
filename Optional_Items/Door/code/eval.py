# type: ignore
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from constants import ABNORMAL_LABEL, NORMAL_LABEL
from segment import fit_timestamp_gap_model, segment_cycle
from train import train_classifiers, predict_segments
from util import parse_timestamps

def interval_iou(
    true_start: pd.Timestamp,
    true_end: pd.Timestamp,
    predicted_start: pd.Timestamp,
    predicted_end: pd.Timestamp,
) -> float:
    """Calculate temporal intersection-over-union for two intervals."""

    intersection = max(
        0.0,
        (min(true_end, predicted_end) - max(true_start, predicted_start))
        .total_seconds(),
    )
    union = (
        (true_end - true_start).total_seconds()
        + (predicted_end - predicted_start).total_seconds()
        - intersection
    )
    return float(intersection / union) if union > 0 else 0.0

def repeated_stratified_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    segment_ids: pd.Series,
    intervals: pd.DataFrame,
    *,
    number_splits: int = 5,
    number_repeats: int = 10,
    random_state: int = 42,
    model
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Random Forest over all repeated stratified folds."""

    if set(y.unique()) != {0, 1}:
        raise ValueError("Both Normal and Abnormal resistance are required.")
    if int(y.value_counts().min()) < number_splits:
        raise ValueError("The minority class is smaller than the fold count.")

    splitter = RepeatedStratifiedKFold(
        n_splits=number_splits,
        n_repeats=number_repeats,
        random_state=random_state,
    )
    splits = list(splitter.split(X, y))
    metric_records: list[dict[str, float | int | str]] = []
    prediction_records: list[dict[str, float | int | str]] = []

    for split_number, (train_index, validation_index) in enumerate(splits):
        repeat = split_number // number_splits + 1
        fold = split_number % number_splits + 1
        model.fit(X.iloc[train_index], y.iloc[train_index])

        probability = model.predict_proba(X.iloc[validation_index])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        truth = y.iloc[validation_index].to_numpy(dtype=int)
        fold_intervals = intervals.iloc[validation_index].reset_index(drop=True)

        true_segments = pd.DataFrame(
            {
                "start_time": parse_timestamps(
                    fold_intervals["true_start_time"]
                ),
                "end_time": parse_timestamps(fold_intervals["true_end_time"]),
                "label": truth,
            }
        )
        predicted_segments = pd.DataFrame(
            {
                "start_time": parse_timestamps(
                    fold_intervals["predicted_start_time"]
                ),
                "end_time": parse_timestamps(
                    fold_intervals["predicted_end_time"]
                ),
                "label": prediction,
            }
        )

        metric_records.append(
            {
                "repeat": repeat,
                "fold": fold,
                "iou_weighted_f1": iou_weighted_f1(
                    true_segments, predicted_segments
                ),
                "pr_auc": float(average_precision_score(truth, probability)),
                "abnormal_precision": float(
                    precision_score(truth, prediction, zero_division=0)
                ),
                "abnormal_recall": float(
                    recall_score(truth, prediction, zero_division=0)
                ),
            }
        )
        for local_index, source_index in enumerate(validation_index):
            prediction_records.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "segment_id": int(segment_ids.iloc[source_index]),
                    "true_status": (
                        ABNORMAL_LABEL if truth[local_index] else NORMAL_LABEL
                    ),
                    "prediction": (
                        ABNORMAL_LABEL if prediction[local_index] else NORMAL_LABEL
                    ),
                    "abnormal_probability": float(probability[local_index]),
                }
            )

    return pd.DataFrame(metric_records), pd.DataFrame(prediction_records)


def summarize_metrics(fold_metrics: pd.DataFrame) -> dict[str, object]:
    """Return Random Forest metric summaries over all 50 folds."""

    metrics = (
        "iou_weighted_f1",
        "pr_auc",
        "abnormal_precision",
        "abnormal_recall",
    )
    return {
        metric: {
            "mean": float(fold_metrics[metric].mean()),
            "std": float(fold_metrics[metric].std(ddof=1)),
            "min": float(fold_metrics[metric].min()),
            "max": float(fold_metrics[metric].max()),
        }
        for metric in metrics
    }


def iou_weighted_f1(
    true_segments: pd.DataFrame,
    predicted_segments: pd.DataFrame,
) -> float:
    """Apply official same-label, greedy, one-to-one IoU-weighted F1."""

    truth = true_segments.reset_index(drop=True).copy()
    predictions = predicted_segments.reset_index(drop=True).copy()
    candidates: list[tuple[float, int, int]] = []

    for true_index, true_row in truth.iterrows():
        for predicted_index, predicted_row in predictions.iterrows():
            if true_row["label"] != predicted_row["label"]:
                continue
            iou = interval_iou(
                true_row["start_time"],
                true_row["end_time"],
                predicted_row["start_time"],
                predicted_row["end_time"],
            )
            if iou > 0:
                candidates.append((iou, int(true_index), int(predicted_index)))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_true: set[int] = set()
    used_predicted: set[int] = set()
    total_iou = 0.0
    for iou, true_index, predicted_index in candidates:
        if true_index in used_true or predicted_index in used_predicted:
            continue
        used_true.add(true_index)
        used_predicted.add(predicted_index)
        total_iou += iou

    soft_recall = total_iou / len(truth) if len(truth) else 0.0
    soft_precision = total_iou / len(predictions) if len(predictions) else 0.0
    denominator = soft_recall + soft_precision
    return (
        float(2 * soft_recall * soft_precision / denominator)
        if denominator > 0
        else 0.0
    )

def evaluate_segments(
    true_segments: pd.DataFrame,
    predicted_segments: pd.DataFrame,
    *,
    true_start_column: str = "start_time",
    true_end_column: str = "end_time",
    predicted_start_column: str = "start_time",
    predicted_end_column: str = "end_time",
    true_label_column: str | None = None,
    predicted_label_column: str | None = None,
) ->  dict[str, float | int | bool]:
    """Calculate greedy one-to-one IoU-weighted precision, recall, and F1.

    Omit both label-column parameters for segmentation-only evaluation. Provide
    both for the official label-aware evaluation. Supplying only one label
    column is rejected because labels cannot be compared asymmetrically.

    Returns
    -------
    summary:
        Counts, total matched IoU, soft recall, soft precision, and final score.
    matches:
        The greedily selected one-to-one matches and the IoU credited to each.
    """

    # Label-aware evaluation requires labels on both sides.
    if (true_label_column is None) != (predicted_label_column is None):
        raise ValueError(
            "Provide both true_label_column and predicted_label_column, "
            "or omit both for segmentation-only evaluation."
        )

    label_aware = true_label_column is not None

    # Check required columns before attempting timestamp conversion.
    true_required = {true_start_column, true_end_column}
    predicted_required = {predicted_start_column, predicted_end_column}
    if label_aware:
        true_required.add(str(true_label_column))
        predicted_required.add(str(predicted_label_column))

    missing_true = sorted(true_required.difference(true_segments.columns))
    missing_predicted = sorted(
        predicted_required.difference(predicted_segments.columns)
    )
    if missing_true:
        raise ValueError(f"Missing true-segment columns: {missing_true}")
    if missing_predicted:
        raise ValueError(f"Missing predicted-segment columns: {missing_predicted}")

    # Reset indices so match identifiers are stable integers from zero onward.
    truth = true_segments.reset_index(drop=True).copy()
    predictions = predicted_segments.reset_index(drop=True).copy()

    # Parse timestamps independently because their source formats may differ.
    truth["_start"] = parse_timestamps(truth[true_start_column])
    truth["_end"] = parse_timestamps(truth[true_end_column])
    predictions["_start"] = parse_timestamps(
        predictions[predicted_start_column]
    )
    predictions["_end"] = parse_timestamps(predictions[predicted_end_column])

    # An interval whose end precedes its start is invalid rather than low quality.
    if truth["_end"].lt(truth["_start"]).any():
        raise ValueError("At least one true segment ends before it starts.")
    if predictions["_end"].lt(predictions["_start"]).any():
        raise ValueError("At least one predicted segment ends before it starts.")

    # Build every valid candidate pair before greedy one-to-one matching.
    candidates: list[dict[str, object]] = []

    for true_index, true_row in truth.iterrows():
        for predicted_index, predicted_row in predictions.iterrows():
            if label_aware:
                # Wrong-label intervals cannot match, even with perfect overlap.
                if (
                    true_row[str(true_label_column)]
                    != predicted_row[str(predicted_label_column)]
                ):
                    continue

            iou = interval_iou(
                true_row["_start"],
                true_row["_end"],
                predicted_row["_start"],
                predicted_row["_end"],
            )

            # The specification permits matches only when IoU is positive.
            if iou <= 0:
                continue

            candidates.append(
                {
                    "true_index": int(true_index),
                    "predicted_index": int(predicted_index),
                    "iou": float(iou),
                }
            )

    # Highest-IoU candidates receive first choice in the greedy matching.
    candidates.sort(
        key=lambda row: (
            -float(row["iou"]),
            int(row["true_index"]),
            int(row["predicted_index"]),
        )
    )

    used_true: set[int] = set()
    used_predicted: set[int] = set()
    selected: list[dict[str, object]] = []

    for candidate in candidates:
        true_index = int(candidate["true_index"])
        predicted_index = int(candidate["predicted_index"])

        # Skip candidates when either interval was already matched more strongly.
        if true_index in used_true or predicted_index in used_predicted:
            continue

        used_true.add(true_index)
        used_predicted.add(predicted_index)

        true_row = truth.iloc[true_index]
        predicted_row = predictions.iloc[predicted_index]

        match: dict[str, object] = {
            "true_index": true_index,
            "predicted_index": predicted_index,
            "true_start": true_row[true_start_column],
            "true_end": true_row[true_end_column],
            "predicted_start": predicted_row[predicted_start_column],
            "predicted_end": predicted_row[predicted_end_column],
            "iou": float(candidate["iou"]),
        }
        if label_aware:
            match["true_label"] = true_row[str(true_label_column)]
            match["predicted_label"] = predicted_row[
                str(predicted_label_column)
            ]
        selected.append(match)

    matches = pd.DataFrame(selected)

    # Every selected match contributes its IoU rather than a flat one point.
    sum_iou = float(matches["iou"].sum()) if not matches.empty else 0.0
    number_true = int(len(truth))
    number_predicted = int(len(predictions))

    # Empty denominators receive zero under the supplied scoring definition.
    soft_recall = sum_iou / number_true if number_true > 0 else 0.0
    soft_precision = (
        sum_iou / number_predicted if number_predicted > 0 else 0.0
    )

    # Calculate the harmonic mean, returning zero when both inputs are zero.
    denominator = soft_recall + soft_precision
    score = (
        2.0 * soft_recall * soft_precision / denominator
        if denominator > 0
        else 0.0
    )

    summary: dict[str, float | int | bool] = {
        "label_aware": label_aware,
        "number_true_segments": number_true,
        "number_predicted_segments": number_predicted,
        "number_matches": int(len(matches)),
        "number_missed_true_segments": number_true - len(used_true),
        "number_unmatched_predictions": number_predicted - len(used_predicted),
        "sum_matched_iou": sum_iou,
        "soft_recall": float(soft_recall),
        "soft_precision": float(soft_precision),
        "score": float(score),
    }
    return summary

def evaluate():
    datastream = pd.read_csv(
        "nebulers/Optional_Items/Door/datasets/Train.csv"
    )
    answers = pd.read_csv(
        "nebulers/Optional_Items/Door/datasets/Train_Segments_Answer.csv"
    )
    
    # Parse before sorting because native timestamps are not zero-padded.
    answers["_parsed_start"] = parse_timestamps(
        answers["start_time"]
    )

    answers = (
        answers
        .sort_values("_parsed_start", kind="stable")
        .drop(columns="_parsed_start")
        .reset_index(drop=True)
    )


    # Use the first 80% of complete cycles for training.
    numberTrainCycles = int(len(answers) * 0.80)

    numberTrainRows = int(
        answers.iloc[:numberTrainCycles]["n_rows"].sum()
    )


    trainAnswers = (
        answers.iloc[:numberTrainCycles]
        .copy()
        .reset_index(drop=True)
    )

    validationAnswers = (
        answers.iloc[numberTrainCycles:]
        .copy()
        .reset_index(drop=True)
    )


    trainData = (
        datastream.iloc[:numberTrainRows]
        .copy()
        .reset_index(drop=True)
    )

    validationData = (
        datastream.iloc[numberTrainRows:]
        .copy()
        .reset_index(drop=True)
    )

    gapModel = fit_timestamp_gap_model(
        trainData,
        time_column="Datetime",
    )
    trainSegmentedRows, trainBoundaries = (
        segment_cycle(
            trainData,
            gapModel,
        )
    )
    modelBundle = train_classifiers(
        segmented_rows=trainSegmentedRows,
        boundaries=trainBoundaries,
        answers=trainAnswers,
        time_column="Datetime",
        random_state=42,
    )

    validationSegmentedRows, validationBoundaries = (
        segment_cycle(
            validationData,
            gapModel,
        )
    )

    validationPredictions, validationFeatures = predict_segments(
        segmented_rows=validationSegmentedRows,
        boundaries=validationBoundaries,
        bundle=modelBundle,
    )

    trueValidationSegments = pd.DataFrame(
        {
            "start_time": parse_timestamps(
                validationAnswers["start_time"]
            ),
            "end_time": parse_timestamps(
                validationAnswers["end_time"]
            ),
            "label": validationAnswers["status"].to_numpy(),
        }
    )

    predictedValidationSegments = pd.DataFrame(
        {
            "start_time": parse_timestamps(
                validationPredictions["start_time"]
            ),
            "end_time": parse_timestamps(
                validationPredictions["end_time"]
            ),
            "label": validationPredictions[
                "prediction"
            ].to_numpy(),
        }
    )

    summary = evaluate_segments(true_segments=trueValidationSegments, predicted_segments=predictedValidationSegments)
    print(summary)

evaluate()