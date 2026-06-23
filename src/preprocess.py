"""Data preprocessing utilities for the car price prediction project."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/car_data.csv")
TARGET_COLUMN = "Selling_Price"
DROP_COLUMNS = ["Year"]
NUMERIC_FEATURES = ["Present_Price", "Kms_Driven", "Owner", "Car_Age"]
CATEGORICAL_FEATURES = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission"]
REQUIRED_COLUMNS = [
    "Car_Name",
    "Year",
    TARGET_COLUMN,
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
]


def load_dataset(data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the car dataset from disk and validate the required columns."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    dataframe = pd.read_csv(path)
    validate_columns(dataframe.columns)
    return dataframe


def validate_columns(columns: Iterable[str]) -> None:
    """Raise a helpful error when the dataset schema is incomplete."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")


def add_car_age(dataframe: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    """Return a copy of the data with Car_Age derived from Year."""
    working_df = dataframe.copy()
    reference_year = current_year or datetime.now().year
    working_df["Car_Age"] = reference_year - working_df["Year"]
    return working_df


def prepare_features(
    dataframe: pd.DataFrame,
    current_year: int | None = None,
    include_target: bool = True,
) -> pd.DataFrame:
    """Create model-ready features while keeping preprocessing deterministic."""
    validate_columns(dataframe.columns if include_target else list(dataframe.columns) + [TARGET_COLUMN])
    processed_df = add_car_age(dataframe, current_year=current_year)
    processed_df = processed_df.drop(columns=DROP_COLUMNS)
    return processed_df


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a processed dataframe into model features and target values."""
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is required for training.")
    features = dataframe.drop(columns=[TARGET_COLUMN])
    target = dataframe[TARGET_COLUMN]
    return features, target


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build the preprocessing pipeline used before RandomForestRegressor."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def create_inference_dataframe(
    car_name: str,
    present_price: float,
    kms_driven: int,
    fuel_type: str,
    seller_type: str,
    transmission: str,
    owner: int,
    car_age: int,
) -> pd.DataFrame:
    """Create a single-row dataframe matching the model's expected input schema."""
    return pd.DataFrame(
        [
            {
                "Car_Name": car_name,
                "Present_Price": present_price,
                "Kms_Driven": kms_driven,
                "Fuel_Type": fuel_type,
                "Seller_Type": seller_type,
                "Transmission": transmission,
                "Owner": owner,
                "Car_Age": car_age,
            }
        ]
    )
