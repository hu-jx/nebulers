import numpy as np
import pandas as pd
import rainflow


# gets the rainflow-based fatigue features used by the shm model
def rainflow_features(stress):
    cycles = list(
        rainflow.extract_cycles(stress)
    )

    ranges = np.array([
        cycle[0]
        for cycle in cycles
    ])

    counts = np.array([
        cycle[2]
        for cycle in cycles
    ])

    if len(ranges) == 0:
        return [
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    total_cycles = counts.sum()
    max_range = ranges.max()
    mean_range = np.average(
        ranges,
        weights=counts,
    )

    pseudo_damage = np.sum(
        (ranges ** 3) * counts
    )

    return [
        total_cycles,
        max_range,
        mean_range,
        pseudo_damage,
    ]


# converts one stress recording into the 19 features used by the model
def extract_features(dataframe, source_name):
    if dataframe.shape[1] == 0:
        raise ValueError(
            f"{source_name}: no stress column found"
        )

    stress = pd.to_numeric(
        dataframe.iloc[:, 0],
        errors="coerce",
    ).dropna().to_numpy()

    if len(stress) == 0:
        raise ValueError(
            f"{source_name}: no numeric stress values found"
        )

    q25, median, q75 = np.percentile(
        stress,
        [25, 50, 75],
    )

    basic = [
        np.mean(stress),
        np.std(stress),
        np.min(stress),
        np.max(stress),
        np.ptp(stress),
        median,
        q25,
        q75,
        q75 - q25,
        np.sqrt(
            np.mean(
                stress ** 2
            )
        ),
        np.mean(
            np.abs(stress)
        ),
    ]

    changes = np.diff(stress)

    dynamic = [
        np.mean(
            np.abs(changes)
        ),
        np.std(changes),
        np.max(
            np.abs(changes)
        ),
        np.sum(
            np.abs(changes)
        ),
    ]

    fatigue = rainflow_features(
        stress
    )

    return basic + dynamic + fatigue


# wraps the saved shm regressor so the app can call .predict()
class SHMModel:
    def __init__(self, model):
        self.model = model

    # returns one fatigue-damage prediction per uploaded recording
    def predict(self, dataframe):
        if "source_file" not in dataframe.columns:
            raise ValueError(
                "shm input is missing source_file"
            )

        rows = []

        for source_file, group in dataframe.groupby(
            "source_file",
            sort=False,
        ):
            raw = group.drop(
                columns=["source_file"]
            ).reset_index(drop=True)

            features = extract_features(
                raw,
                source_file,
            )

            X = np.asarray(
                features,
                dtype=float,
            ).reshape(1, -1)

            log_prediction = self.model.predict(
                X
            )[0]

            prediction = float(
                np.exp(
                    log_prediction
                )
            )

            rows.append({
                "file_id": source_file,
                "prediction": prediction,
            })

        return pd.DataFrame(
            rows,
            columns=[
                "file_id",
                "prediction",
            ],
        )
