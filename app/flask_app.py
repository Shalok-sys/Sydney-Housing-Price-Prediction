# Flask web app for the Sydney housing price prediction tool.
# Run it with, python3 app/flask_app.py

import json
import math
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "..", "models", "best_model.joblib")
SCHEMA_PATH = os.path.join(APP_DIR, "..", "models", "feature_schema.json")

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH) as f:
    schema = json.load(f)

# Houses and townhouses report land area, apartments and units report
# internal floor area. The model keeps these as separate features.
HOUSE_LIKE = ["House", "Townhouse", "Villa"]


def to_float(value, field, minimum, maximum, required=True):
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError(f"{field} is required")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number")
    if number < minimum or number > maximum:
        raise ValueError(f"{field} must be between {minimum:g} and {maximum:g}")
    return number


def build_input_row(form):
    suburb = form.get("suburb")
    if suburb not in schema["suburb_options"]:
        raise ValueError("Unknown suburb")

    property_type = form.get("property_type")
    if property_type not in schema["property_type_options"]:
        raise ValueError("Unknown property type")

    sale_method = form.get("sale_method") or "Unknown"
    if sale_method not in schema["sale_method_options"]:
        sale_method = "Unknown"

    bedrooms = to_float(form.get("bedrooms"), "Bedrooms", 0, 8)
    bathrooms = to_float(form.get("bathrooms"), "Bathrooms", 0, 6)
    parking = to_float(form.get("parking_spaces"), "Parking spaces", 0, 10, required=False)
    area = to_float(form.get("area_sqm"), "Area", 10, 20000, required=False)

    is_house_like = property_type in HOUSE_LIKE

    row = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking_spaces": parking,
        "land_size_sqm": area if is_house_like else None,
        "floor_area_sqm": None if is_house_like else area,
        "bath_per_bed": (bathrooms / bedrooms) if bedrooms else None,
        "suburb": suburb,
        "property_type": property_type,
        "sale_method": sale_method,
    }
    return pd.DataFrame([row]), suburb, property_type


def support_warning(suburb, property_type):
    """
    Say plainly when the training data cannot back up an estimate.

    Some suburb and type combinations have very few sales behind them,
    and Parramatta houses have none at all, so an estimate there is an
    extrapolation rather than something learned from comparable sales.
    """
    key = f"{suburb}|{property_type}"
    if key not in schema["observed_combinations"]:
        return (
            f"There are no {property_type.lower()} sales in {suburb} in the training "
            "data, so this figure is an extrapolation from other suburbs and types. "
            "Treat it as a rough indication only."
        )
    for item in schema["thin_support"]:
        if item["suburb"] == suburb and item["property_type"] == property_type:
            one = item["count"] == 1
            return (
                f"Only {item['count']} {property_type.lower()} "
                f"{'sale' if one else 'sales'} in {suburb} "
                f"{'sits' if one else 'sit'} behind this estimate, so it is much "
                "less reliable than the range suggests."
            )
    return None


@app.route("/")
def index():
    return render_template("index.html", schema=schema)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.get_json(silent=True) or {}
    try:
        input_row, suburb, property_type = build_input_row(form)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    # The model predicts the log of the price, so convert back to dollars
    price = float(math.exp(model.predict(input_row)[0]))

    error_pct = schema["suburb_typical_error_pct"].get(
        suburb, schema["overall_typical_error_pct"]
    )
    margin = price * error_pct / 100

    return jsonify({
        "price": price,
        "low": max(0, price - margin),
        "high": price + margin,
        "error_pct": error_pct,
        "suburb": suburb,
        "warning": support_warning(suburb, property_type),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
