"""Pytest coverage for preprocessing, model loading, and prediction."""

from __future__ import annotations

import pandas as pd

from src.predict import load_model, predict_price
from src.preprocess import prepare_features


def test_model_loading() -> None:
    """The trained model artifact should load successfully."""
    model = load_model()
    assert hasattr(model, "predict")


def test_prediction_function_returns_positive_float() -> None:
    """A valid payload should produce a positive numeric prediction."""
    payload = {
        "car_name": "city",
        "present_price": "7.5",
        "kms_driven": "35000",
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "0",
        "car_age": "5",
    }
    prediction = predict_price(payload)
    assert isinstance(prediction, float)
    assert prediction >= 0


def test_data_preprocessing_creates_car_age_and_keeps_car_name() -> None:
    """Preprocessing should derive Car_Age while preserving car model input."""
    raw_df = pd.DataFrame(
        {
            "Car_Name": ["ritz"],
            "Year": [2018],
            "Selling_Price": [4.5],
            "Present_Price": [6.2],
            "Kms_Driven": [32000],
            "Fuel_Type": ["Petrol"],
            "Seller_Type": ["Dealer"],
            "Transmission": ["Manual"],
            "Owner": [0],
        }
    )
    processed_df = prepare_features(raw_df, current_year=2026)
    assert "Car_Age" in processed_df.columns
    assert "Car_Name" in processed_df.columns
    assert processed_df.loc[0, "Car_Age"] == 8
    assert "Year" not in processed_df.columns
