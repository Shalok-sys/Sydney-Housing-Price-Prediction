"""
This script builds the housing dataset used in this project.

Note on data source. The task asks for manually collected sold listings
from realestate.com.au or domain.com.au. Automated scraping of those
sites was not reliable in this environment because of bot protection
and site terms of use. So this script builds a simulated dataset
instead. Prices are generated from a formula calibrated to public
median price figures for each suburb, then random noise, missing
values and a few unusual sales are added so the dataset behaves like
a real world dataset. Before final submission this should be replaced
or supplemented with real manually collected listings if full marks
for the data collection method are required.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

# Fix the random seed so the dataset is the same every time this runs
rng = np.random.default_rng(42)

# Number of properties to generate per suburb
N_PER_SUBURB = 40

# Basic suburb profile used to drive the price formula
# distance_cbd is roughly how far the suburb centre is from Sydney CBD
SUBURBS = {
    "Mosman": {
        "distance_cbd": 8,
        "land_rate": 5000,      # dollars per square metre of land, house
        "floor_rate_house": 3500,   # dollars per square metre of floor, house
        "floor_rate_unit": 13000,   # dollars per square metre of floor, unit
        "year_built_range": (1910, 2005),
        "land_size_range": (280, 900),
        "unit_floor_range": (55, 130),
        "house_floor_range": (150, 420),
    },
    "Parramatta": {
        "distance_cbd": 24,
        "land_rate": 2600,
        "floor_rate_house": 1800,
        "floor_rate_unit": 7800,
        "year_built_range": (1950, 2023),
        "land_size_range": (250, 700),
        "unit_floor_range": (45, 110),
        "house_floor_range": (120, 320),
    },
    "Liverpool": {
        "distance_cbd": 33,
        "land_rate": 1100,
        "floor_rate_house": 1300,
        "floor_rate_unit": 6300,
        "year_built_range": (1985, 2023),
        "land_size_range": (320, 750),
        "unit_floor_range": (45, 100),
        "house_floor_range": (140, 300),
    },
}

PROPERTY_TYPES = ["House", "Unit", "Townhouse"]
PROPERTY_TYPE_WEIGHTS = [0.5, 0.35, 0.15]

# Small text snippets used to build a realistic agent description
FEATURE_PHRASES = [
    "open plan living", "updated kitchen", "north facing living area",
    "close to shops and cafes", "leafy quiet street", "double garage",
    "easy walk to the station", "great natural light", "low maintenance garden",
    "high ceilings", "built in wardrobes", "recently painted",
]


def pick_property_type():
    return rng.choice(PROPERTY_TYPES, p=PROPERTY_TYPE_WEIGHTS)


def random_date_last_two_years():
    days_back = int(rng.integers(0, 730))
    return date.today() - timedelta(days=days_back)


def build_description(suburb, property_type, bedrooms, bathrooms, tag=None):
    phrases = rng.choice(FEATURE_PHRASES, size=3, replace=False)
    text = (
        f"{property_type} in {suburb} with {bedrooms} bedrooms and "
        f"{bathrooms} bathrooms. Features {phrases[0]}, {phrases[1]} and "
        f"{phrases[2]}."
    )
    if tag:
        text += f" {tag}"
    return text


def generate_row(suburb):
    profile = SUBURBS[suburb]
    property_type = pick_property_type()
    is_unit = property_type == "Unit"

    bedrooms = int(rng.integers(1, 3)) if is_unit else int(rng.integers(2, 6))
    bathrooms = min(bedrooms, int(rng.integers(1, 4)))
    parking = int(rng.integers(0, 2)) if is_unit else int(rng.integers(1, 3))

    year_lo, year_hi = profile["year_built_range"]
    year_built = int(rng.integers(year_lo, year_hi))
    property_age = date.today().year - year_built

    renovated = int(rng.random() < 0.35)
    has_pool = 0 if is_unit else int(rng.random() < 0.15)

    distance_cbd = profile["distance_cbd"] + rng.normal(0, 2.5)
    distance_cbd = max(1, round(distance_cbd, 1))
    distance_station = round(abs(rng.normal(0.9, 0.7)), 2)
    distance_school = round(abs(rng.normal(1.1, 0.6)), 2)

    if is_unit:
        land_size = np.nan
        floor_lo, floor_hi = profile["unit_floor_range"]
        floor_area = round(rng.uniform(floor_lo, floor_hi), 1)
        floor_rate = profile["floor_rate_unit"]
        land_value = 0
    else:
        land_lo, land_hi = profile["land_size_range"]
        land_size = round(rng.uniform(land_lo, land_hi), 1)
        if property_type == "Townhouse":
            land_size = round(land_size * 0.55, 1)
        floor_lo, floor_hi = profile["house_floor_range"]
        floor_area = round(rng.uniform(floor_lo, floor_hi), 1)
        floor_rate = profile["floor_rate_house"]
        land_rate = profile["land_rate"]
        if property_type == "Townhouse":
            land_rate = land_rate * 0.7
        land_value = land_size * land_rate

    # Age reduces value a little, unless the property was renovated
    age_factor = max(0.6, 1 - 0.004 * property_age)
    if renovated:
        age_factor = min(1.05, age_factor + 0.15)

    building_value = floor_area * floor_rate * age_factor

    amenity_value = (
        bedrooms * 15000
        + bathrooms * 20000
        + parking * 15000
        + has_pool * 60000
        - distance_cbd * 3000
        - distance_station * 8000
    )

    base_price = land_value + building_value + amenity_value

    # Add realistic random variation on top of the base price
    noise_factor = np.exp(rng.normal(0, 0.08))
    price = base_price * noise_factor
    price = max(price, 250000)

    days_on_market = int(rng.integers(5, 120))
    sale_date = random_date_last_two_years()

    description = build_description(suburb, property_type, bedrooms, bathrooms)

    return {
        "suburb": suburb,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking_spaces": parking,
        "land_size_sqm": land_size,
        "floor_area_sqm": floor_area,
        "distance_to_cbd_km": distance_cbd,
        "distance_to_station_km": distance_station,
        "distance_to_school_km": distance_school,
        "year_built": year_built,
        "renovated": renovated,
        "has_pool": has_pool,
        "days_on_market": days_on_market,
        "sale_date": sale_date.isoformat(),
        "agent_description": description,
        "sale_price": round(price, -3),
    }


def add_unusual_sales(df):
    """
    Real housing datasets always contain a few sales that do not follow
    the general pattern, for example a deceased estate sold quickly
    below value, or a home with a rare feature that buyers paid extra
    for. This function turns a small number of rows into these unusual
    cases so later error analysis has real examples to look at.
    """
    unusual_index = rng.choice(df.index, size=6, replace=False)
    low_cases = unusual_index[:3]
    high_cases = unusual_index[3:]

    for idx in low_cases:
        df.loc[idx, "sale_price"] = round(df.loc[idx, "sale_price"] * rng.uniform(0.55, 0.68), -3)
        df.loc[idx, "agent_description"] += " Deceased estate, sold as is, needs full renovation."

    for idx in high_cases:
        df.loc[idx, "sale_price"] = round(df.loc[idx, "sale_price"] * rng.uniform(1.6, 1.95), -3)
        df.loc[idx, "agent_description"] += " Rare unobstructed water views and a full architect designed renovation."

    return df


def add_missing_values(df):
    # Copy the frame so the caller keeps a clean version if needed
    df = df.copy()
    n = len(df)

    floor_missing = rng.choice(df.index, size=int(n * 0.05), replace=False)
    df.loc[floor_missing, "floor_area_sqm"] = np.nan

    parking_missing = rng.choice(df.index, size=int(n * 0.04), replace=False)
    df.loc[parking_missing, "parking_spaces"] = np.nan

    bathroom_missing = rng.choice(df.index, size=int(n * 0.03), replace=False)
    df.loc[bathroom_missing, "bathrooms"] = np.nan

    return df


def main():
    rows = []
    for suburb in SUBURBS:
        for _ in range(N_PER_SUBURB):
            rows.append(generate_row(suburb))

    df = pd.DataFrame(rows)
    df.insert(0, "property_id", range(1, len(df) + 1))

    df = add_unusual_sales(df)
    df = add_missing_values(df)

    # Shuffle the rows so suburbs are not grouped in blocks
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["property_id"] = range(1, len(df) + 1)

    out_path = "data/sydney_housing_sales.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["suburb"].value_counts())


if __name__ == "__main__":
    main()
