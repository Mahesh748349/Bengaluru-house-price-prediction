import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.utils import rmsle


def split_features_target(df: pd.DataFrame, target_col: str):
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])
    return X, y


def get_feature_columns(X: pd.DataFrame):
    drop_cols = [col for col in X.columns if str(col).lower() in {"price_per_sqft_train"}]
    X = X.drop(columns=drop_cols, errors="ignore")
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return X, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def get_candidate_models():
    models = {
        "ridge": Ridge(alpha=5.0, random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=900,
            learning_rate=0.035,
            max_depth=4,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    except Exception as exc:
        warnings.warn(f"XGBoost unavailable: {exc}")

    return models


def evaluate_models(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    X, numeric_cols, categorical_cols = get_feature_columns(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    log_y = np.log1p(y)

    def log_rmse(y_true_log, y_pred_log):
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_true_log)
        return -rmsle(y_true, y_pred)

    scorer = make_scorer(log_rmse, greater_is_better=True)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, model in get_candidate_models().items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X, log_y, cv=cv, scoring=scorer, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "mean_rmsle": -scores.mean(),
                "std_rmsle": scores.std(),
                "fold_scores": [-float(s) for s in scores],
            }
        )

    return pd.DataFrame(rows).sort_values("mean_rmsle").reset_index(drop=True)


def fit_model(X: pd.DataFrame, y: pd.Series, model_name: str = "extra_trees"):
    X, numeric_cols, categorical_cols = get_feature_columns(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = get_candidate_models()[model_name]
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipe.fit(X, np.log1p(y))
    return pipe


def predict_prices(model, X: pd.DataFrame):
    X, _, _ = get_feature_columns(X)
    return np.maximum(np.expm1(model.predict(X)), 0)
