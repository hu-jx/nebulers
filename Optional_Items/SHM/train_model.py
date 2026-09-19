import pandas as pd
import numpy as np
import rainflow
import joblib


from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV, KFold


BASE = Path(".")
data_folder = BASE / "Optional_Items" / "SHM" / "SHM_Datasets" / "Train"
label_file = BASE / "Optional_Items" / "SHM" / "SHM_Datasets" / "Train_Labels.csv"

labels = pd.read_csv(label_file)


def rainflow_features(stress):
    cycles = list(rainflow.extract_cycles(stress))
    # each cycle is (range, mean, count, i_start, i_end)
    ranges = np.array([c[0] for c in cycles])
    counts = np.array([c[2] for c in cycles])

    if len(ranges) == 0:
        return [0, 0, 0, 0]

    total_cycles = counts.sum()
    max_range = ranges.max()
    mean_range = np.average(ranges, weights=counts)
    # crude Miner's-rule-style proxy: sum of (range^m * count), m=3 is a common default for metals
    pseudo_damage = np.sum((ranges ** 3) * counts)

    return [total_cycles, max_range, mean_range, pseudo_damage]

def extract_features(file):
    stress = pd.read_csv(file, header=None).iloc[:, 0]
    stress = pd.to_numeric(stress, errors="coerce").dropna().to_numpy()

    q25, median, q75 = np.percentile(stress, [25, 50, 75] )

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

X = []
y = []

for _, row in labels.iterrows():
    file_path = data_folder / row["filename"]

    features = extract_features(file_path)

    X.append(features)
    y.append(row["damage"])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("First features:", X[0])
print("First damage value:", y[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

parameter_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 3, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    parameter_grid,
    scoring="neg_mean_absolute_error",
    cv=cv
)

grid.fit(X_train, np.log(y_train))

model = grid.best_estimator_

print("Best parameters:", grid.best_params_)

model.fit(X_train, np.log(y_train)) 

predictions = np.exp(model.predict(X_test))

mape = mean_absolute_percentage_error(y_test, predictions)

print("Predictions:", predictions)
print("Actual values:", y_test)
print("MAPE:", mape)


def run_loo(X, y, use_log, best_params):
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in loo.split(X):
        model = RandomForestRegressor(
            **best_params,
            random_state=42
        )

        if use_log:
            model.fit(X[train_idx], np.log(y[train_idx]))
            preds[test_idx] = np.exp(
                model.predict(X[test_idx])
            )
        else:
            model.fit(X[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X[test_idx])

    mape = mean_absolute_percentage_error(y, preds)

    return mape, preds

mape_raw, preds_raw = run_loo(
    X,
    y,
    use_log=False,
    best_params=grid.best_params_
)

mape_log, preds_log = run_loo(
    X,
    y,
    use_log=True,
    best_params=grid.best_params_
)

print("LOO-CV MAPE (raw):", mape_raw)
print("LOO-CV MAPE (log):", mape_log)

final_model = RandomForestRegressor(
    **grid.best_params_,
    random_state=42
)

final_model.fit(X, np.log(y))

joblib.dump(final_model, "model.joblib")