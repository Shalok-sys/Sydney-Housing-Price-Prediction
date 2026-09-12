# Flask web app for the Sydney housing price prediction tool.
# Run it with, python3 app/flask_app.py

import json
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


def to_float(value, field, minimum=0.0, maximum=1_000_000.0):
    """Read one number from the submitted form and keep it in range."""
    if value is None or value == "":
        raise ValueError(f"{field} is required")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number")
    if number < minimum or number > maximum:
        raise ValueError(f"{field} must be between {minimum:g} and {maximum:g}")
    return number


def build_input_row(form):
    """Turn the submitted form into the one row the model expects."""
    suburb = form.get("suburb")
    if suburb not in schema["suburb_options"]:
        raise ValueError("Unknown suburb")

    property_type = form.get("property_type")
    if property_type not in schema["property_type_options"]:
        raise ValueError("Unknown property type")

    year_built = to_float(form.get("year_built"), "Year built", 1850, schema["current_year"])
    description = (form.get("agent_description") or "")[:2000]

    # Units do not have their own land size, so leave it empty
    is_unit = property_type == "Unit"
    land_size = None if is_unit else to_float(form.get("land_size_sqm"), "Land size", 0, 20000)

    row = {
        "bedrooms": to_float(form.get("bedrooms"), "Bedrooms", 0, 20),
        "bathrooms": to_float(form.get("bathrooms"), "Bathrooms", 0, 20),
        "parking_spaces": to_float(form.get("parking_spaces"), "Parking spaces", 0, 20),
        "land_size_sqm": land_size,
        "floor_area_sqm": to_float(form.get("floor_area_sqm"), "Floor area", 10, 5000),
        "distance_to_cbd_km": to_float(form.get("distance_to_cbd_km"), "Distance to the CBD", 0, 200),
        "distance_to_station_km": to_float(form.get("distance_to_station_km"), "Distance to a station", 0, 100),
        "distance_to_school_km": to_float(form.get("distance_to_school_km"), "Distance to a school", 0, 100),
        "property_age": schema["current_year"] - year_built,
        "renovated": 1 if form.get("renovated") else 0,
        "has_pool": 1 if form.get("has_pool") else 0,
        "days_on_market": to_float(form.get("days_on_market"), "Days on market", 0, 2000),
        "description_length": len(description),
        "mentions_view": int("view" in description.lower()),
        "suburb": suburb,
        "property_type": property_type,
    }
    return pd.DataFrame([row])


@app.route("/")
def index():
    return render_template("index.html", schema=schema)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.get_json(silent=True) or {}
    try:
        input_row = build_input_row(form)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    price = float(model.predict(input_row)[0])

    # Show a range built from how far off the model usually is in this
    # suburb, measured by cross validation on the training data
    error_pct = schema["suburb_typical_error_pct"].get(
        form.get("suburb"), schema["overall_typical_error_pct"]
    )
    margin = price * error_pct / 100

    return jsonify({
        "price": price,
        "low": max(0, price - margin),
        "high": price + margin,
        "error_pct": error_pct,
        "suburb": form.get("suburb"),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
