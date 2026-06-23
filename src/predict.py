"""Prediction helpers used by Flask and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from src.preprocess import create_inference_dataframe

MODEL_PATH = Path("models/car_price_model.pkl")
VALID_FUEL_TYPES = {"Petrol", "Diesel", "CNG"}
VALID_SELLER_TYPES = {"Dealer", "Individual"}
VALID_TRANSMISSIONS = {"Manual", "Automatic"}


def load_model(model_path: str | Path = MODEL_PATH) -> Pipeline:
    """Load the persisted model pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path.resolve()}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def validate_prediction_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce web form input before model inference."""
    try:
        present_price = float(payload["present_price"])
        kms_driven = int(float(payload["kms_driven"]))
        owner = int(payload["owner"])
        car_age = int(payload["car_age"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Please enter valid numeric values for all numeric fields.") from exc

    car_name = str(payload.get("car_name", "")).strip().lower()
    fuel_type = str(payload.get("fuel_type", "")).strip()
    seller_type = str(payload.get("seller_type", "")).strip()
    transmission = str(payload.get("transmission", "")).strip()

    if not car_name:
        raise ValueError("Please enter the car model name.")
    if present_price <= 0:
        raise ValueError("Present price must be greater than 0.")
    if kms_driven < 0:
        raise ValueError("Kilometers driven cannot be negative.")
    if owner < 0:
        raise ValueError("Owner count cannot be negative.")
    if not 0 <= car_age <= 50:
        raise ValueError("Car age must be between 0 and 50 years.")
    if fuel_type not in VALID_FUEL_TYPES:
        raise ValueError("Please select a valid fuel type.")
    if seller_type not in VALID_SELLER_TYPES:
        raise ValueError("Please select a valid seller type.")
    if transmission not in VALID_TRANSMISSIONS:
        raise ValueError("Please select a valid transmission type.")

    return {
        "car_name": car_name,
        "present_price": present_price,
        "kms_driven": kms_driven,
        "fuel_type": fuel_type,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "car_age": car_age,
    }


def predict_price(payload: dict[str, Any], model: Pipeline | None = None) -> float:
    """Predict the selling price in lakhs for a single car."""
    validated = validate_prediction_input(payload)
    estimator = model or load_model()
    input_df = create_inference_dataframe(**validated)
    prediction = estimator.predict(input_df)[0]
    return round(float(max(prediction, 0)), 2)
