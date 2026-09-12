"""
Single source of truth for the collected dataset.

Every column here is something a sold listing page actually shows, so
the person collecting the data never has to guess or estimate. Anything
the model needs but a listing does not publish, such as distance to the
city or whether the place has a pool, is derived later in code from the
columns below. Keeping that split strict is what stops estimated values
being recorded as if they were observed facts.
"""

# Suburbs being collected. Swap this list to change the study area.
# The task sheet asks for three suburbs with substantially different
# housing markets, and for at least 30 properties from each.
SUBURBS = {
    "Mosman": {
        "state": "NSW",
        "postcode": "2088",
        "distance_to_cbd_km": 8.0,
        "market_note": "Premium harbourside, high land value, mostly older larger homes",
    },
    "Parramatta": {
        "state": "NSW",
        "postcode": "2150",
        "distance_to_cbd_km": 24.0,
        "market_note": "Middle ring, many newer apartments, strong transport links",
    },
    "Liverpool": {
        "state": "NSW",
        "postcode": "2170",
        "distance_to_cbd_km": 33.0,
        "market_note": "Outer growth corridor, newer housing, larger blocks, more affordable",
    },
}

MIN_ROWS_TOTAL = 100
MIN_ROWS_PER_SUBURB = 30

# Only these two sites are used, as named in the task sheet
ALLOWED_SOURCE_DOMAINS = ["realestate.com.au", "domain.com.au"]

PROPERTY_TYPES = ["House", "Unit", "Apartment", "Townhouse", "Villa"]

SALE_METHODS = ["Auction", "Private treaty", "Unknown"]

# Columns to collect, in the order they should appear in the CSV.
# required says whether a row is unusable without it.
COLUMNS = [
    {
        "name": "listing_url",
        "type": "text",
        "required": True,
        "note": "Full link to the sold listing. This is the evidence the row is real.",
    },
    {
        "name": "address",
        "type": "text",
        "required": True,
        "note": "Street address as shown on the listing",
    },
    {
        "name": "suburb",
        "type": "category",
        "required": True,
        "allowed": list(SUBURBS.keys()),
        "note": "Must match one of the suburbs being collected",
    },
    {
        "name": "postcode",
        "type": "integer",
        "required": True,
        "min": 1000,
        "max": 9999,
        "note": "Four digit postcode",
    },
    {
        "name": "property_type",
        "type": "category",
        "required": True,
        "allowed": PROPERTY_TYPES,
        "note": "As labelled on the listing",
    },
    {
        "name": "bedrooms",
        "type": "integer",
        "required": True,
        "min": 0,
        "max": 15,
        "note": "Bedroom icon count on the listing",
    },
    {
        "name": "bathrooms",
        "type": "integer",
        "required": True,
        "min": 0,
        "max": 15,
        "note": "Bathroom icon count on the listing",
    },
    {
        "name": "parking_spaces",
        "type": "integer",
        "required": False,
        "min": 0,
        "max": 20,
        "note": "Car icon count. Leave blank if the listing does not show one.",
    },
    {
        "name": "land_size_sqm",
        "type": "number",
        "required": False,
        "min": 0,
        "max": 20000,
        "note": "Land size in square metres. Usually blank for apartments.",
    },
    {
        "name": "sale_price",
        "type": "number",
        "required": True,
        "min": 50000,
        "max": 100000000,
        "note": "Sold price in dollars. Skip the listing entirely if the price is undisclosed.",
    },
    {
        "name": "sale_date",
        "type": "date",
        "required": True,
        "note": "Date sold, written as YYYY MM DD",
    },
    {
        "name": "sale_method",
        "type": "category",
        "required": False,
        "allowed": SALE_METHODS,
        "note": "Auction or private treaty when the listing says so",
    },
    {
        "name": "agent_description",
        "type": "text",
        "required": False,
        "note": "The listing description text. Used for the text features.",
    },
    {
        "name": "features_list",
        "type": "text",
        "required": False,
        "note": "Property features shown as tags, separated by a semicolon",
    },
    {
        "name": "date_collected",
        "type": "date",
        "required": True,
        "note": "The date you collected this row, written as YYYY MM DD",
    },
]

COLUMN_NAMES = [column["name"] for column in COLUMNS]

REQUIRED_COLUMNS = [column["name"] for column in COLUMNS if column["required"]]


def column_by_name(name):
    for column in COLUMNS:
        if column["name"] == name:
            return column
    return None
