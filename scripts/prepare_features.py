"""
Turns the collected listings into the table used for modelling.

Three jobs. Remove rows that are not comparable residential sales, work
out the area figures, and derive the features the listings imply rather
than state.

Enrichment from individual listing pages arrives gradually, so the
description, feature tag and floor area columns may be complete, empty,
or anywhere in between. Every derivable feature is computed regardless,
and the coverage of each is reported at the end. The notebook decides
which ones are complete enough to model with, using the threshold
below, so the same code works before, during and after enrichment.

Usage
    python3 scripts/prepare_features.py
"""

import json
import os

import numpy as np
import pandas as pd

SOURCE = "data/raw/collected_listings.csv"
OUTPUT = "data/processed/model_ready.csv"
COVERAGE_REPORT = "data/processed/feature_coverage.json"

# A feature is only worth modelling with once it describes this share of
# the dataset. Below it, imputation would be inventing most of the column.
MIN_COVERAGE = 0.40

MAX_BEDROOMS = 8
MAX_BATHROOMS = 6

# An apartment does not sit on its own land parcel. When the area shown
# for one is this large it is the whole block, not the apartment.
MAX_APARTMENT_AREA = 400

HOUSE_LIKE = ["House", "Townhouse", "Villa"]
UNIT_LIKE = ["Apartment", "Unit"]

CURRENT_YEAR = 2026

# Words that signal something the structured columns do not record
POOL_WORDS = ["pool"]
VIEW_WORDS = ["view", "outlook", "aspect", "harbour", "water"]
RENOVATION_WORDS = ["renovat", "refurbish", "rebuilt", "restored", "updated"]


def load():
    df = pd.read_csv(SOURCE)
    numeric = ["bedrooms", "bathrooms", "parking_spaces", "land_size_sqm",
               "floor_area_sqm", "year_built", "sale_price"]
    for column in numeric:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ["agent_description", "features_list"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("")
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    return df


def drop_rows(df):
    """Remove rows that cannot be modelled or are not a single dwelling."""
    removed = []

    def cut(mask, reason):
        nonlocal df
        for _, row in df[mask].iterrows():
            removed.append((row["address"], row["suburb"], reason))
        df = df[~mask].copy()

    cut(df["bedrooms"].isna(), "no bedroom count, cannot be modelled")
    cut(df["sale_price"].isna(), "no sale price, nothing to predict")
    cut(df["bedrooms"] > MAX_BEDROOMS,
        f"more than {MAX_BEDROOMS} bedrooms, a block of dwellings rather than one home")
    cut(df["bathrooms"] > MAX_BATHROOMS,
        f"more than {MAX_BATHROOMS} bathrooms, a block of dwellings rather than one home")

    return df, removed


def resolve_areas(df):
    """
    Work out which area figure belongs in which column.

    The results pages put one ambiguous area against every property, land
    for a house and internal floor area for an apartment. Individual
    listing pages state floor area properly in its own column. So an
    explicitly collected floor area is trusted first, and only where none
    exists is the ambiguous column reassigned by property type.
    """
    collected_floor = df["floor_area_sqm"].copy()
    ambiguous = df["land_size_sqm"].copy()

    is_unit_like = df["property_type"].isin(UNIT_LIKE)

    # For a flat, an unexplained area is its floor area, not its land
    inferred_floor = ambiguous.where(is_unit_like)
    df["floor_area_sqm"] = collected_floor.fillna(inferred_floor)

    # Land size only ever describes something that sits on its own land
    df["land_size_sqm"] = ambiguous.where(~is_unit_like)

    # A flat reporting a very large area is describing the whole block
    block_area = df["floor_area_sqm"] > MAX_APARTMENT_AREA
    df.loc[block_area & is_unit_like, "floor_area_sqm"] = np.nan

    return df, int((block_area & is_unit_like).sum()), int(collected_floor.notna().sum())


def mentions(series, words):
    pattern = "|".join(words)
    return series.str.lower().str.contains(pattern, regex=True, na=False)


def add_features(df):
    """Derive what the listings imply. Empty text yields an empty feature."""
    has_text = (df["agent_description"].str.strip() != "")
    combined = (df["agent_description"].fillna("") + " " + df["features_list"].fillna(""))

    # These are only meaningful where there was text to read
    df["description_length"] = df["agent_description"].str.len().where(has_text)
    df["mentions_pool"] = mentions(combined, POOL_WORDS).astype(float).where(has_text)
    df["mentions_view"] = mentions(combined, VIEW_WORDS).astype(float).where(has_text)
    df["mentions_renovation"] = mentions(combined, RENOVATION_WORDS).astype(float).where(has_text)

    df["property_age"] = (CURRENT_YEAR - df["year_built"]).where(df["year_built"].notna())

    df["bath_per_bed"] = (df["bathrooms"] / df["bedrooms"]).replace([np.inf, -np.inf], np.nan)

    earliest = df["sale_date"].min()
    df["days_since_first_sale"] = (df["sale_date"] - earliest).dt.days

    return df


# Features whose blanks carry meaning rather than absence. A flat has no
# land and a house reports no strata floor area, so a low fill rate here
# is the shape of the housing stock, not missing work. These are always
# modelled and the threshold does not apply to them.
# The order is fixed deliberately. A random forest samples features by
# column index at each split, so reordering this list shifts the results
# of a seeded fit even when the features themselves are unchanged.
STRUCTURAL_FEATURES = [
    "bedrooms", "bathrooms", "parking_spaces",
    "land_size_sqm", "floor_area_sqm", "bath_per_bed",
]

# Features that exist only where a listing page has been read. Here a
# blank really does mean not collected yet, so imputing a mostly empty
# column would be inventing the feature rather than measuring it.
ENRICHMENT_FEATURES = [
    "description_length", "mentions_pool",
    "mentions_view", "mentions_renovation", "property_age",
]


def write_coverage(df):
    """Record which features are complete enough for the notebook to use."""
    coverage = {}
    for name in STRUCTURAL_FEATURES + ENRICHMENT_FEATURES:
        coverage[name] = round(float(df[name].notna().mean()), 4)

    usable = list(STRUCTURAL_FEATURES) + [
        name for name in ENRICHMENT_FEATURES if coverage[name] >= MIN_COVERAGE
    ]

    report = {
        "min_coverage": MIN_COVERAGE,
        "coverage": coverage,
        "structural_features": STRUCTURAL_FEATURES,
        "enrichment_features": ENRICHMENT_FEATURES,
        "usable_numeric_features": usable,
        "rows": int(len(df)),
    }
    os.makedirs(os.path.dirname(COVERAGE_REPORT), exist_ok=True)
    with open(COVERAGE_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    df = load()
    starting = len(df)

    df, removed = drop_rows(df)
    df, block_areas, collected_floor = resolve_areas(df)
    df = add_features(df)
    report = write_coverage(df)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Started with {starting} collected listings")
    print(f"Removed {len(removed)}")
    for address, suburb, reason in removed:
        print(f"  {address}, {suburb}")
        print(f"      {reason}")
    print()
    print(f"Floor areas taken straight from a listing page  {collected_floor}")
    print(f"Flat areas blanked as the whole block           {block_areas}")
    print()
    print(f"Wrote {len(df)} rows to {OUTPUT}")
    print()
    print("Feature coverage")
    print("  structural, always modelled, blanks carry meaning")
    for name in STRUCTURAL_FEATURES:
        print(f"    use   {name:<22} {report['coverage'][name]:6.1%}")
    print(f"  from enrichment, modelled once coverage reaches {MIN_COVERAGE:.0%}")
    for name in ENRICHMENT_FEATURES:
        value = report["coverage"][name]
        mark = "use " if value >= MIN_COVERAGE else "skip"
        print(f"    {mark}  {name:<22} {value:6.1%}")
    print()
    print("Rows per suburb")
    print(df["suburb"].value_counts().to_string())


if __name__ == "__main__":
    main()
