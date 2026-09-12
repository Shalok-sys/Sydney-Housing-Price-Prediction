# Streamlit app for the Sydney housing price prediction tool.
# Run it with, streamlit run app/streamlit_app.py

import json
import os

import joblib
import pandas as pd
import streamlit as st

# Work out paths relative to this file, so the app runs from any folder
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "..", "models", "best_model.joblib")
SCHEMA_PATH = os.path.join(APP_DIR, "..", "models", "feature_schema.json")


@st.cache_resource
def load_model_and_schema():
    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    return model, schema


st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏠")
st.title("Sydney Housing Price Predictor")
st.write(
    "Enter the details of a property below to get a predicted sale price. "
    "This tool was trained on a simulated dataset of Mosman, Parramatta "
    "and Liverpool sales and is a prototype, not a certified valuation."
)

model, schema = load_model_and_schema()
current_year = schema["current_year"]

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        suburb = st.selectbox("Suburb", schema["suburb_options"])
        property_type = st.selectbox("Property type", schema["property_type_options"])
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
        parking_spaces = st.number_input("Parking spaces", min_value=0, max_value=10, value=1)
        year_built = st.number_input(
            "Year built", min_value=1850, max_value=current_year, value=2000
        )

    with col2:
        is_unit = property_type == "Unit"
        land_size_sqm = st.number_input(
            "Land size in square metres",
            min_value=0.0,
            value=0.0 if is_unit else 400.0,
            disabled=is_unit,
            help="Units usually do not have their own land size",
        )
        floor_area_sqm = st.number_input(
            "Floor area in square metres", min_value=10.0, value=150.0
        )
        distance_to_cbd_km = st.number_input(
            "Distance to the CBD in kilometres", min_value=0.0, value=15.0
        )
        distance_to_station_km = st.number_input(
            "Distance to the nearest station in kilometres", min_value=0.0, value=1.0
        )
        distance_to_school_km = st.number_input(
            "Distance to the nearest school in kilometres", min_value=0.0, value=1.0
        )
        days_on_market = st.number_input("Days on market", min_value=0, value=30)

    renovated = st.checkbox("Recently renovated")
    has_pool = st.checkbox("Has a pool")
    agent_description = st.text_area(
        "Agent description, optional",
        placeholder="Write a short listing description, for example mention any view",
    )

    submitted = st.form_submit_button("Predict sale price")

if submitted:
    # Turn the raw inputs into the same engineered features used in training
    property_age = current_year - year_built
    description_length = len(agent_description)
    mentions_view = int("view" in agent_description.lower())

    land_value = None if is_unit else land_size_sqm

    input_row = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking_spaces": parking_spaces,
        "land_size_sqm": land_value,
        "floor_area_sqm": floor_area_sqm,
        "distance_to_cbd_km": distance_to_cbd_km,
        "distance_to_station_km": distance_to_station_km,
        "distance_to_school_km": distance_to_school_km,
        "property_age": property_age,
        "renovated": int(renovated),
        "has_pool": int(has_pool),
        "days_on_market": days_on_market,
        "description_length": description_length,
        "mentions_view": mentions_view,
        "suburb": suburb,
        "property_type": property_type,
    }])

    predicted_price = model.predict(input_row)[0]

    st.success(f"Predicted sale price, ${predicted_price:,.0f}")
    st.caption(
        "This is a prototype estimate from a simulated dataset. "
        "Treat it as a guide, especially for unique or high value properties."
    )

    with st.expander("Show the input row sent to the model"):
        st.dataframe(input_row)
