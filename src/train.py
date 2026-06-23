"""Train and persist the car price prediction model."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import (
    DATA_PATH,
    build_preprocessing_pipeline,
    load_dataset,
    prepare_features,
    split_features_target,
)

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "car_price_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessing_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
RANDOM_STATE = 42


def build_model() -> Pipeline:
    """Create the full preprocessing and regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=4,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(data_path: str | Path = DATA_PATH) -> dict[str, float]:
    """Train the model, save artifacts, and return evaluation metrics."""
    dataframe = load_dataset(data_path)
    processed_df = prepare_features(dataframe)
    features, target = split_features_target(processed_df)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "r2_score": round(r2_score(y_test, predictions), 4),
        "mae": round(mean_absolute_error(y_test, predictions), 4),
        "rmse": round(sqrt(mean_squared_error(y_test, predictions)), 4),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(model.named_steps["preprocessor"], PREPROCESSOR_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    training_metrics = train_model()
    print(f"Model saved to {MODEL_PATH}")
    print(json.dumps(training_metrics, indent=2))
