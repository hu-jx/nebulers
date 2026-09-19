import os
from pathlib import Path
import joblib
import pandas as pd

from constants import ABNORMAL_LABEL, DOOR_MODEL_PATH, PREDICTED_TEST_FOR_DOOR
from train import train_final_model, predict_segments
from segment import  segment_cycle
from clean import check_required_columns
from constants import NORMAL_LABEL, ABNORMAL_LABEL

TIME_COLUMN = "Datetime"

def predict(test_path: str) -> None:
    ext = Path(test_path).suffix.lower()
    
    if ext == '.csv':
        test_df = pd.read_csv(test_path)
    elif ext == '.xlsx':
        test_df = pd.read_excel(test_path)
    else:
        raise Exception("Unsupported data type")

    check_required_columns(test_df, time_column=TIME_COLUMN)
    if not os.path.exists(DOOR_MODEL_PATH):
        print("Directory is empty. Running train script...")
        answers = pd.read_csv("nebulers/Optional_Items/Door/datasets/Train_Segments_Answer.csv")
        datastream = pd.read_csv("nebulers/Optional_Items/Door/datasets/Train.csv")
        train_final_model(datastream=datastream, answers=answers)
    model = joblib.load(Path(DOOR_MODEL_PATH))
    classifier = model["classifier_bundle"]
    gap_model = model["gap_model"]

    segmented_rows, segment_boundaries = segment_cycle(
        test_df,
        gap_model,
    )
     
    classified_segments, _ = predict_segments(
        segmented_rows,
        segment_boundaries,
        classifier,
    )

    submission = classified_segments[
        ["start_time", "end_time", "prediction"]
    ].copy()

    allowed_labels = {NORMAL_LABEL, ABNORMAL_LABEL}
    unexpected_labels = set(submission["prediction"]).difference(allowed_labels)
    if unexpected_labels:
        raise RuntimeError(
            f"Classifier produced unsupported labels: {sorted(unexpected_labels)}"
        )
    if len(submission) != len(segment_boundaries):
        raise RuntimeError("Not every detected segment received a prediction.")
    final_df = submission.reset_index(drop=True)
    final_df.to_csv(PREDICTED_TEST_FOR_DOOR)

predict("nebulers/Optional_Items/Door/datasets/Test.csv")