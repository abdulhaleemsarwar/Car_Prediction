"""
train_model.py
---------------
Trains the RandomForestRegressor price-prediction pipeline on the PakWheels
used-car dataset and saves everything the Gradio app needs at inference time:

    model.joblib     -> the fitted sklearn Pipeline (preprocessing + model)
    metadata.json     -> dropdown choices, numeric ranges, and eval metrics

This mirrors the original training approach (ColumnTransformer with
StandardScaler for numeric features + OneHotEncoder for categorical features,
feeding a RandomForestRegressor). Run this once; the app then just loads the
saved artifacts and never retrains.

Usage:
    python train_model.py
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "Clean_Data_pakwheels.csv"
MODEL_PATH = "model.joblib"
METADATA_PATH = "metadata.json"

NUMERIC_FEATURES = ["Model Year", "Mileage", "Engine Capacity"]
CATEGORICAL_FEATURES = [
    "Company Name",
    "Model Name",
    "Engine Type",
    "Color",
    "Assembly",
    "Body Type",
    "Transmission Type",
    "Registration Status",
]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Price"


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.iloc[:, 1:]          # drop the stray index column
    df = df.drop("Location", axis=1)
    df = df.drop_duplicates()
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("nums", StandardScaler(), NUMERIC_FEATURES),
            ("cats", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("processor", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )


def export_metadata(df: pd.DataFrame, metrics: dict) -> dict:
    """Collects dropdown choices + numeric ranges so the app can build its
    UI without needing the raw CSV at runtime."""
    categorical_choices = {
        col: sorted(df[col].dropna().unique().tolist())
        for col in CATEGORICAL_FEATURES
    }
    # Company -> Model mapping, so the UI can filter Model Name choices
    # as soon as the user picks a Company Name.
    company_to_models = (
        df.groupby("Company Name")["Model Name"]
        .apply(lambda s: sorted(s.dropna().unique().tolist()))
        .to_dict()
    )
    numeric_ranges = {
        col: {
            "min": int(df[col].min()),
            "max": int(df[col].max()),
            "median": int(df[col].median()),
        }
        for col in NUMERIC_FEATURES
    }
    return {
        "categorical_choices": categorical_choices,
        "company_to_models": company_to_models,
        "numeric_ranges": numeric_ranges,
        "feature_order": FEATURE_ORDER,
        "metrics": metrics,
    }


def main():
    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"  -> {df.shape[0]} rows after cleanup")

    X = df[FEATURE_ORDER]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print("Training RandomForestRegressor pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("Evaluating on held-out test set...")
    preds = pipeline.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
    }
    print(f"  MAE:  {metrics['mae']:,.0f} PKR")
    print(f"  RMSE: {metrics['rmse']:,.0f} PKR")
    print(f"  R2:   {metrics['r2']:.4f}")

    print(f"Saving model to {MODEL_PATH} ...")
    # compress=6 shrinks the on-disk RandomForest considerably with no
    # change to predictions (it's the same fitted trees, just gzipped).
    joblib.dump(pipeline, MODEL_PATH, compress=6)

    print(f"Saving UI metadata to {METADATA_PATH} ...")
    metadata = export_metadata(df, metrics)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
