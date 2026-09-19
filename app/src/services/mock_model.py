import random

import pandas as pd


class MockModel:
    def __init__(self, subsystem):
        self.subsystem = subsystem

    def predict(self, dataframe):
        if self.subsystem.key == "door":
            return _predict_door(len(dataframe))
        if self.subsystem.key == "rail_corrugation":
            return _predict_per_file(dataframe, lambda: random.choice(["Normal", "Side I", "Side II"]))
        if self.subsystem.key == "shm":
            return _predict_per_file(dataframe, lambda: round(random.uniform(0, 1), 4))
        return dataframe


def _predict_per_file(dataframe, value_fn):
    file_ids = dataframe["source_file"].unique()
    return pd.DataFrame({"file_id": file_ids, "prediction": [value_fn() for _ in file_ids]})


def _predict_door(n):
    rows = []
    second = 0
    for _ in range(n):
        start_second = second
        start_ms = random.randint(0, 900)
        duration = random.randint(3, 5)
        end_second = start_second + duration
        end_ms = random.randint(0, 900)
        label = random.choice(["Normal", "Normal", "Normal", "Abnormal resistance"])
        rows.append({
            "start_time": _stamp(start_second, start_ms),
            "end_time": _stamp(end_second, end_ms),
            "prediction": label,
        })
        second = end_second + random.randint(20, 40)
    return pd.DataFrame(rows)


def _stamp(total_seconds, ms):
    hour = total_seconds // 3600
    minute = (total_seconds % 3600) // 60
    sec = total_seconds % 60
    return f"2023-7-5-{hour}-{minute}-{sec}-{ms}"