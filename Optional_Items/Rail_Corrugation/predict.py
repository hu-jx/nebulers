import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


TRAIN_FEATURE_FILE = "rail_features_final.csv"
MODEL_OUTPUT_FILE = "rail_corrugation_model.joblib"

LABELS = ["Normal", "Side I", "Side II"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}
INT_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

INNER_SPLITS = 3
RANDOM_STATE = 42


# makes the final svm with the frozen settings
def create_svm(random_state):
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            ),
        ),
    ])


# makes the final xgboost model with the frozen settings
def create_xgb(random_state):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,
    )


# trains xgboost with balanced sample weights
def fit_xgb(X_train, y_train, random_state):
    model = create_xgb(random_state)

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
    )

    return model


# combines svm and xgboost probabilities for the meta-model
def get_base_probs(svm_model, xgb_model, X_data):
    return np.hstack([
        svm_model.predict_proba(X_data),
        xgb_model.predict_proba(X_data),
    ])


# builds oof meta-features, then retrains both base models on all train data
def train_final_stack(X_train, y_train):
    oof_meta = np.zeros(
        (len(X_train), 6),
        dtype=float,
    )

    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for fold_number, (train_idx, val_idx) in enumerate(
        inner_cv.split(X_train, y_train),
        start=1,
    ):
        print(
            f"building meta features: fold {fold_number}/{INNER_SPLITS}"
        )

        X_inner_train = X_train.iloc[train_idx]
        X_inner_val = X_train.iloc[val_idx]
        y_inner_train = y_train[train_idx]

        seed = RANDOM_STATE + fold_number

        svm_model = create_svm(seed)
        svm_model.fit(
            X_inner_train,
            y_inner_train,
        )

        xgb_model = fit_xgb(
            X_inner_train,
            y_inner_train,
            seed,
        )

        oof_meta[val_idx] = get_base_probs(
            svm_model,
            xgb_model,
            X_inner_val,
        )

    meta_model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    meta_model.fit(
        oof_meta,
        y_train,
    )

    print("training final svm on all 272 files")
    final_svm = create_svm(
        RANDOM_STATE
    )
    final_svm.fit(
        X_train,
        y_train,
    )

    print("training final xgboost on all 272 files")
    final_xgb = fit_xgb(
        X_train,
        y_train,
        RANDOM_STATE,
    )

    return final_svm, final_xgb, meta_model


# trains the frozen model and saves everything the app needs for inference
def main():
    train_df = pd.read_csv(
        TRAIN_FEATURE_FILE
    )

    feature_columns = [
        column
        for column in train_df.columns
        if column not in ["file_id", "label"]
    ]

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df["label"].map(
        LABEL_TO_INT
    ).to_numpy()

    if np.isnan(y_train).any():
        raise ValueError(
            "found an unexpected label in the training data"
        )

    print(
        "train shape:",
        X_train.shape,
    )

    print()
    print(
        train_df["label"].value_counts().to_string()
    )
    print()

    final_svm, final_xgb, meta_model = train_final_stack(
        X_train,
        y_train,
    )

    model_bundle = {
        "svm": final_svm,
        "xgb": final_xgb,
        "meta_model": meta_model,
        "feature_columns": feature_columns,
        "labels": INT_TO_LABEL,
        "model_info": {
            "name": "rail_corrugation_stacking",
            "feature_count": len(feature_columns),
            "random_state": RANDOM_STATE,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
        },
    }

    joblib.dump(
        model_bundle,
        MODEL_OUTPUT_FILE,
    )

    print()
    print(
        "saved:",
        MODEL_OUTPUT_FILE,
    )

    print(
        "feature count:",
        len(feature_columns),
    )

    print(
        "sklearn version:",
        sklearn.__version__,
    )

    print(
        "xgboost version:",
        xgboost.__version__,
    )


if __name__ == "__main__":
    main()
