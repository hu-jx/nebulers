import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


TRAIN_FEATURE_FILE = "rail_features_final.csv"
TEST_FEATURE_FILE = "rail_features_test_final.csv"
OUTPUT_FILE = "rail_predictions.csv"

LABELS = ["Normal", "Side I", "Side II"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}
INT_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

INNER_SPLITS = 3
RANDOM_STATE = 42


# makes the final svm with the settings kept after tuning
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


# combines svm and xgboost class probabilities for stacking
def get_base_probs(svm_model, xgb_model, X_data):
    return np.hstack([
        svm_model.predict_proba(X_data),
        xgb_model.predict_proba(X_data),
    ])


# builds oof base predictions for the meta-model, then retrains both base models on all train data
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


# predicts the test set and converts class numbers back to the required labels
def predict_test(
    svm_model,
    xgb_model,
    meta_model,
    X_test,
):
    meta_features = get_base_probs(
        svm_model,
        xgb_model,
        X_test,
    )

    predicted_ints = meta_model.predict(
        meta_features
    )

    return np.array([
        INT_TO_LABEL[int(value)]
        for value in predicted_ints
    ])


# checks that train and test use the exact same feature columns
def check_feature_columns(train_df, test_df):
    train_features = [
        column
        for column in train_df.columns
        if column not in ["file_id", "label"]
    ]

    test_features = [
        column
        for column in test_df.columns
        if column != "file_id"
    ]

    if train_features != test_features:
        missing_in_test = sorted(
            set(train_features) - set(test_features)
        )

        extra_in_test = sorted(
            set(test_features) - set(train_features)
        )

        raise ValueError(
            "train/test feature columns do not match.\n"
            f"missing in test: {missing_in_test}\n"
            f"extra in test: {extra_in_test}"
        )

    return train_features


# trains on all labelled data and writes the final rail submission file
def main():
    train_df = pd.read_csv(
        TRAIN_FEATURE_FILE
    )

    test_df = pd.read_csv(
        TEST_FEATURE_FILE
    )

    feature_columns = check_feature_columns(
        train_df,
        test_df,
    )

    X_train = train_df[
        feature_columns
    ]

    X_test = test_df[
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

    print(
        "test shape:",
        X_test.shape,
    )

    print()
    print(
        "training class counts:"
    )

    print(
        train_df["label"].value_counts().to_string()
    )

    print()

    svm_model, xgb_model, meta_model = train_final_stack(
        X_train,
        y_train,
    )

    predictions = predict_test(
        svm_model,
        xgb_model,
        meta_model,
        X_test,
    )

    submission = pd.DataFrame({
        "file_id": test_df["file_id"],
        "prediction": predictions,
    })

    if len(submission) != 68:
        raise ValueError(
            f"expected 68 predictions, found {len(submission)}"
        )

    if submission["file_id"].duplicated().any():
        raise ValueError(
            "duplicate test file ids found"
        )

    if submission["prediction"].isna().any():
        raise ValueError(
            "missing predictions found"
        )

    invalid_labels = set(
        submission["prediction"]
    ) - set(LABELS)

    if invalid_labels:
        raise ValueError(
            f"invalid prediction labels found: {invalid_labels}"
        )

    submission.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "prediction counts:"
    )

    print(
        submission["prediction"].value_counts().to_string()
    )

    print()
    print(
        "saved:",
        OUTPUT_FILE,
    )

    print(
        "rows:",
        len(submission),
    )


if __name__ == "__main__":
    main()
