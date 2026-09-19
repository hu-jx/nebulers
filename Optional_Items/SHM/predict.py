import argparse
import joblib
import numpy as np
import pandas as pd
import rainflow

from pathlib import Path


def rainflow_features(stress):
    cycles = list(rainflow.extract_cycles(stress))

    ranges = np.array([cycle[0] for cycle in cycles])
    counts = np.array([cycle[2] for cycle in cycles])

    if len(ranges) == 0:
        return [0, 0, 0, 0]

    total_cycles = counts.sum()
    max_range = ranges.max()
    mean_range = np.average(ranges, weights=counts)
    pseudo_damage = np.sum((ranges ** 3) * counts)

    return [
        total_cycles,
        max_range,
        mean_range,
        pseudo_damage
    ]


def extract_features(file):
    stress = pd.read_csv(file, header=None).iloc[:, 0]

    stress = pd.to_numeric(
        stress,
        errors="coerce"
    ).dropna().to_numpy()

    q25, median, q75 = np.percentile(
        stress,
        [25, 50, 75]
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
        np.sqrt(np.mean(stress ** 2)),
        np.mean(np.abs(stress)),
    ]

    changes = np.diff(stress)

    dynamic = [
        np.mean(np.abs(changes)),
        np.std(changes),
        np.max(np.abs(changes)),
        np.sum(np.abs(changes)),
    ]

    fatigue = rainflow_features(stress)

    return basic + dynamic + fatigue

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing test CSV files"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file"
    )

    args = parser.parse_args()

    input_folder = Path(args.input)
    output_file = Path(args.output)
    
    files = sorted(input_folder.glob("*.csv"))


    if len(files) == 0:
        raise FileNotFoundError(
            f"No CSV files found in {input_folder.resolve()}"
        )

    model_path = Path(__file__).parent / "model.joblib"
    model = joblib.load(model_path)

    results = []

    for file in files:
        features = extract_features(file)
        features = np.array(features).reshape(1, -1)

        log_prediction = model.predict(features)[0]
        prediction = np.exp(log_prediction)

        results.append({
            "file_id": file.name,
            "prediction": prediction
        })

    predictions = pd.DataFrame(
        results,
        columns=["file_id", "prediction"]
    )

    predictions.to_csv(
        output_file,
        index=False
    )

    print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    main()
