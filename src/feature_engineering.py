import numpy as np
import pandas as pd


PREMIUM_LOCATION_KEYWORDS = [
    "indiranagar",
    "koramangala",
    "whitefield",
    "hsr",
    "jayanagar",
    "malleshwaram",
    "malleswaram",
    "rajaji nagar",
    "sadashiva nagar",
    "lavelle",
    "mg road",
    "richmond",
    "hebbal",
    "sarjapur",
    "bellandur",
    "marathahalli",
]


def add_real_estate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"total_sqft_clean", "bhk"}.issubset(df.columns):
        df["sqft_per_bhk"] = df["total_sqft_clean"] / df["bhk"].replace(0, np.nan)
        df["is_compact_home"] = (df["sqft_per_bhk"] < 450).astype(int)
        df["is_spacious_home"] = (df["sqft_per_bhk"] > 900).astype(int)

    if {"bath", "bhk"}.issubset(df.columns):
        df["bath_per_bhk"] = df["bath"] / df["bhk"].replace(0, np.nan)
        df["extra_bath_count"] = (df["bath"] - df["bhk"]).clip(lower=0)

    if "balcony" in df.columns:
        df["has_balcony"] = (df["balcony"].fillna(0) > 0).astype(int)

    if "availability_clean" in df.columns:
        df["is_ready_to_move"] = (df["availability_clean"] == "ready_to_move").astype(int)

    if "location_clean" in df.columns:
        location_text = df["location_clean"].fillna("unknown").astype(str)
        pattern = "|".join(PREMIUM_LOCATION_KEYWORDS)
        df["is_premium_location_keyword"] = location_text.str.contains(pattern, regex=True).astype(int)

    if "total_sqft_clean" in df.columns:
        df["log_total_sqft"] = np.log1p(df["total_sqft_clean"])

    if "parking" in df.columns:
        df["has_parking"] = (df["parking"].fillna(0) > 0).astype(int)

    if "furnishing" in df.columns:
        furnishing_clean = df["furnishing"].fillna("unknown").astype(str).str.lower().str.strip()
        df["is_furnished"] = furnishing_clean.isin(["semi-furnished", "fully-furnished"]).astype(int)
        df["is_fully_furnished"] = (furnishing_clean == "fully-furnished").astype(int)

    if "property_type" in df.columns:
        property_type_clean = df["property_type"].fillna("unknown").astype(str).str.lower().str.strip()
        df["is_apartment"] = (property_type_clean == "apartment").astype(int)
        df["is_villa"] = (property_type_clean == "villa").astype(int)
        df["is_independent_house"] = (property_type_clean == "independent house").astype(int)

    return df


def add_train_based_location_features(
    train_df: pd.DataFrame,
    valid_or_test_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    valid_or_test_df = valid_or_test_df.copy()

    if "location_clean" not in train_df.columns:
        return train_df, valid_or_test_df

    counts = train_df["location_clean"].value_counts()
    train_df["location_count"] = train_df["location_clean"].map(counts).fillna(0)
    valid_or_test_df["location_count"] = valid_or_test_df["location_clean"].map(counts).fillna(0)

    if target_col in train_df.columns and "total_sqft_clean" in train_df.columns:
        tmp = train_df.copy()
        tmp["price_per_sqft_train"] = tmp[target_col] / tmp["total_sqft_clean"].replace(0, np.nan)
        loc_pps = tmp.groupby("location_clean")["price_per_sqft_train"].median()
        global_pps = tmp["price_per_sqft_train"].median()

        train_df["location_median_pps"] = train_df["location_clean"].map(loc_pps).fillna(global_pps)
        valid_or_test_df["location_median_pps"] = (
            valid_or_test_df["location_clean"].map(loc_pps).fillna(global_pps)
        )

        premium_cutoff = loc_pps.quantile(0.85)
        train_df["is_premium_location_stat"] = (
            train_df["location_median_pps"] >= premium_cutoff
        ).astype(int)
        valid_or_test_df["is_premium_location_stat"] = (
            valid_or_test_df["location_median_pps"] >= premium_cutoff
        ).astype(int)

    return train_df, valid_or_test_df


def reduce_rare_categories(train_df: pd.DataFrame, test_df: pd.DataFrame, min_count: int = 10):
    train_df = train_df.copy()
    test_df = test_df.copy()
    cat_cols = train_df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        keep = train_df[col].value_counts(dropna=False)
        keep = set(keep[keep >= min_count].index)
        train_df[col] = train_df[col].where(train_df[col].isin(keep), "other")
        if col in test_df.columns:
            test_df[col] = test_df[col].where(test_df[col].isin(keep), "other")

    return train_df, test_df
