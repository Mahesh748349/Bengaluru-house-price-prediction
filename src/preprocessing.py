import re

import numpy as np
import pandas as pd


SQFT_UNIT_FACTORS = {
    "sq. meter": 10.7639,
    "sq.meter": 10.7639,
    "sq meter": 10.7639,
    "sqm": 10.7639,
    "sq. yard": 9.0,
    "sq.yard": 9.0,
    "sq yard": 9.0,
    "sq yards": 9.0,
    "perch": 272.25,
    "acre": 43560.0,
    "acres": 43560.0,
    "cent": 435.6,
    "cents": 435.6,
    "ground": 2400.0,
    "grounds": 2400.0,
    "guntha": 1089.0,
}


def clean_location(value):
    if pd.isna(value):
        return "unknown"
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("'", "")
    text = text.replace(".", " ")
    text = re.sub(r"\s+", " ", text).strip()
    aliases = {
        "electronic city phase ii": "electronic city phase 2",
        "electonics city phase 1": "electronic city phase 1",
        "white field": "whitefield",
        "yelahanka new town": "yelahanka",
    }
    return aliases.get(text, text)


def parse_bhk(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else np.nan


def parse_total_sqft(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower().strip()
    text = text.replace(",", "")

    range_match = re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", text)
    if range_match:
        low, high = map(float, range_match[0])
        return (low + high) / 2

    number_match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not number_match:
        return np.nan

    number = float(number_match.group(1))
    for unit, factor in SQFT_UNIT_FACTORS.items():
        if unit in text:
            return number * factor
    return number


def clean_availability(value):
    if pd.isna(value):
        return "unknown"
    text = str(value).lower().strip()
    if "ready" in text or "immediate" in text:
        return "ready_to_move"
    if re.search(r"\d{2}-[a-z]{3}", text):
        return "future_possession"
    return text


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    if "area" in df.columns and "total_sqft" not in df.columns:
        df["total_sqft"] = df["area"]

    if "location" in df.columns:
        df["location_clean"] = df["location"].map(clean_location)

    if "size" in df.columns:
        df["bhk"] = df["size"].map(parse_bhk)

    if "total_sqft" in df.columns:
        df["total_sqft_clean"] = df["total_sqft"].map(parse_total_sqft)

    if "availability" in df.columns:
        df["availability_clean"] = df["availability"].map(clean_availability)

    return df


def remove_training_outliers(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    mask = pd.Series(True, index=df.index)

    if "total_sqft_clean" in df.columns:
        mask &= df["total_sqft_clean"].isna() | df["total_sqft_clean"].between(150, 100000)

    if "bhk" in df.columns:
        mask &= df["bhk"].isna() | df["bhk"].between(1, 20)

    if "bath" in df.columns:
        mask &= df["bath"].isna() | df["bath"].between(1, 20)

    if {"total_sqft_clean", "bhk"}.issubset(df.columns):
        sqft_per_bhk = df["total_sqft_clean"] / df["bhk"].replace(0, np.nan)
        mask &= sqft_per_bhk.isna() | sqft_per_bhk.between(250, 10000)

    if {"bath", "bhk"}.issubset(df.columns):
        bath_gap = df["bath"] - df["bhk"]
        mask &= bath_gap.isna() | (bath_gap <= 3)

    if target_col in df.columns:
        mask &= df[target_col].notna()
        mask &= df[target_col] >= 0

    return df.loc[mask].reset_index(drop=True)
