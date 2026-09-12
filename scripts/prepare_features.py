"""
Turns the collected listings into the table used for modelling.

Two jobs. First, remove rows that are not comparable residential sales,
since a boarding house or a whole apartment block sold as one lot is a
different kind of transaction to a family home. Second, work out the
features the listings do not state directly.

Every exclusion is printed with its reason, so the decisions are visible
rather than buried, and the counts can be quoted in the report.

Usage
    python3 scripts/prepare_features.py
"""

import os

import pandas as pd

SOURCE = "data/raw/collected_listings.csv"
OUTPUT = "data/processed/model_ready.csv"

# Above these a listing is a block of dwellings or a boarding house
# rather than a single home, so it is not comparable to the rest
MAX_BEDROOMS = 8
MAX_BATHROOMS = 6

# An apartment does not sit on its own land parcel. When the area shown
# for one is this large it is the whole block, not the apartment.
MAX_APARTMENT_AREA = 400

HOUSE_LIKE = ["House", "Townhouse", "Villa"]
UNIT_LIKE = ["Apartment", "Unit"]


def load():
    df = pd.read_csv(SOURCE)
    for column in ["bedrooms", "bathrooms", "parking_spaces", "land_size_sqm", "sale_price"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
    return df


def drop_rows(df):
    """Remove rows that cannot be modelled or are not a single dwelling."""
    removed = []

    def cut(mask, reason):
        nonlocal df
        hit = df[mask]
        for _, row in hit.iterrows():
            removed.append((row["address"], row["suburb"], reason))
        df = df[~mask].copy()

    cut(df["bedrooms"].isna(), "no bedroom count, cannot be modelled")
    cut(df["sale_price"].isna(), "no sale price, nothing to predict")
    cut(df["bedrooms"] > MAX_BEDROOMS, f"more than {MAX_BEDROOMS} bedrooms, a block of dwellings rather than one home")
    cut(df["bathrooms"] > MAX_BATHROOMS, f"more than {MAX_BATHROOMS} bathrooms, a block of dwellings rather than one home")

    return df, removed


def split_area(df):
    """
    Separate the one area column into two that mean different things.

    The sites report land area for a house and internal floor area for
    an apartment, but both arrive in the same column. Left together the
    model would read the difference between 600 and 100 as a land effect
    when it is really the difference between a house and a flat.
    """
    df["land_size_sqm"] = df["land_size_sqm"].where(df["property_type"].isin(HOUSE_LIKE))
    floor = df["land_size_sqm"].copy()

    source = pd.to_numeric(pd.read_csv(SOURCE)["land_size_sqm"], errors="coerce")
    source = source.reindex(df.index)

    df["floor_area_sqm"] = source.where(df["property_type"].isin(UNIT_LIKE))

    # An apartment showing a very large area is reporting the whole block
    too_big = df["floor_area_sqm"] > MAX_APARTMENT_AREA
    df.loc[too_big, "floor_area_sqm"] = pd.NA

    return df, int(too_big.sum())


def add_features(df):
    # How far through the collection window the sale happened. The window
    # is short so this mostly guards against a drift being missed.
    earliest = df["sale_date"].min()
    df["days_since_first_sale"] = (df["sale_date"] - earliest).dt.days

    df["sale_month"] = df["sale_date"].dt.month

    # Bathrooms and parking relative to size, a rough fitout signal
    df["bath_per_bed"] = (df["bathrooms"] / df["bedrooms"]).replace([float("inf")], pd.NA)

    return df


def main():
    df = load()
    starting = len(df)

    df, removed = drop_rows(df)
    df, block_areas = split_area(df)
    df = add_features(df)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"Started with {starting} collected listings")
    print(f"Removed {len(removed)}")
    for address, suburb, reason in removed:
        print(f"  {address}, {suburb}")
        print(f"      {reason}")
    print()
    print(f"Blanked {block_areas} apartment areas that were the whole block, not the apartment")
    print()
    print(f"Wrote {len(df)} rows to {OUTPUT}")
    print()
    print("Rows per suburb")
    print(df["suburb"].value_counts().to_string())
    print()
    print("Rows per property type")
    print(df["property_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
