import joblib
import pandas as pd

from src.config import (
    ID_CANDIDATES,
    MODELS_DIR,
    N_SPLITS,
    OUTPUTS_DIR,
    SUBMISSION_PATH,
    TARGET_CANDIDATES,
    TEST_PATH,
    TRAIN_PATH,
)
from src.feature_engineering import (
    add_real_estate_features,
    add_train_based_location_features,
    reduce_rare_categories,
)
from src.modeling import evaluate_models, fit_model, predict_prices, split_features_target
from src.preprocessing import basic_clean, remove_training_outliers
from src.utils import detect_column, ensure_dirs, save_json, summarize_dataframe


def main():
    ensure_dirs(MODELS_DIR, OUTPUTS_DIR)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Put your training file at {TRAIN_PATH}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH) if TEST_PATH.exists() else None

    target_col = detect_column(train.columns, TARGET_CANDIDATES)
    if target_col is None:
        raise ValueError(f"Could not detect target column. Rename target to one of {TARGET_CANDIDATES}.")

    id_col = detect_column(test.columns, ID_CANDIDATES) if test is not None else None

    summarize_dataframe(train).to_csv(OUTPUTS_DIR / "data_summary.csv")

    train = basic_clean(train)
    train = remove_training_outliers(train, target_col)
    train = add_real_estate_features(train)

    if test is not None:
        test = basic_clean(test)
        test = add_real_estate_features(test)
        train, test = add_train_based_location_features(train, test, target_col)
        train, test = reduce_rare_categories(train, test, min_count=10)
    else:
        train, _ = add_train_based_location_features(train, train.copy(), target_col)

    X, y = split_features_target(train, target_col)
    scores = evaluate_models(X, y, n_splits=N_SPLITS)
    scores.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    best_model_name = scores.iloc[0]["model"]
    model = fit_model(X, y, model_name=best_model_name)
    joblib.dump(model, MODELS_DIR / "best_model.joblib")

    save_json(
        {
            "target_col": target_col,
            "id_col": id_col,
            "best_model": best_model_name,
            "best_cv_rmsle": float(scores.iloc[0]["mean_rmsle"]),
        },
        OUTPUTS_DIR / "run_metadata.json",
    )

    print("Model comparison:")
    print(scores[["model", "mean_rmsle", "std_rmsle"]])

    if test is not None:
        preds = predict_prices(model, test)
        submission = pd.DataFrame()
        if id_col is not None:
            submission[id_col] = test[id_col]
        else:
            submission["id"] = range(len(test))
        submission[target_col] = preds
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Saved submission to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
