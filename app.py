"""Flask web application for car price prediction."""

from __future__ import annotations

import logging
from typing import Any

from flask import Flask, render_template, request

from src.predict import predict_price

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


def get_form_defaults() -> dict[str, Any]:
    """Return default form values for the home page."""
    return {
        "car_name": "",
        "present_price": "",
        "kms_driven": "",
        "fuel_type": "Petrol",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "0",
        "car_age": "",
    }


@app.route("/", methods=["GET"])
def home() -> str:
    """Render the prediction form."""
    return render_template("index.html", form_data=get_form_defaults())


@app.route("/predict", methods=["POST"])
def predict() -> str:
    """Validate request data, run inference, and render the result."""
    form_data = get_form_defaults() | request.form.to_dict()
    try:
        prediction = predict_price(form_data)
        return render_template(
            "index.html",
            form_data=form_data,
            prediction=f"₹ {prediction:.2f} Lakhs",
        )
    except ValueError as exc:
        return render_template("index.html", form_data=form_data, error=str(exc)), 400
    except Exception as exc:  # pragma: no cover - defensive production guard
        app.logger.exception("Prediction failed: %s", exc)
        return render_template(
            "index.html",
            form_data=form_data,
            error="Something went wrong while predicting. Please try again.",
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
