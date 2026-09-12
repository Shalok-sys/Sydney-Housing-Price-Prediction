"""
Takes raw rows pasted from the browser extension and adds them to the
dataset safely.

The extension will not always give back perfectly clean values. Prices
come back as "$1,425,000", dates in whatever format the listing used,
property types as "Apartment" one time and "apartment" the next. This
script normalises all of that, drops anything unusable, skips listings
already collected, and appends the rest.

Usage
    1. Paste the extension output into data/raw/inbox.txt
    2. python3 scripts/ingest_listings.py
    3. python3 scripts/validate_dataset.py data/raw/collected_listings.csv

Nothing is ever overwritten. Rows are only added.
"""

import csv
import io
import os
import re
import sys
from datetime import date, datetime

from collection_schema import (
    ALLOWED_SOURCE_DOMAINS,
    COLUMN_NAMES,
    PROPERTY_TYPES,
    SALE_METHODS,
    SUBURBS,
)

INBOX = "data/raw/inbox.txt"
DATASET = "data/raw/collected_listings.csv"

# Spellings the sites use, mapped to the types the schema allows
TYPE_SYNONYMS = {
    "apartment": "Apartment",
    "apartment / unit / flat": "Apartment",
    "unit": "Unit",
    "flat": "Unit",
    "house": "House",
    "acreage": "House",
    "semi detached": "House",
    "semi": "House",
    "duplex": "Townhouse",
    "townhouse": "Townhouse",
    "terrace": "Townhouse",
    "villa": "Villa",
}

METHOD_SYNONYMS = {
    "auction": "Auction",
    "sold at auction": "Auction",
    "private treaty": "Private treaty",
    "private sale": "Private treaty",
    "private": "Private treaty",
    "sold prior to auction": "Private treaty",
}

DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y",
    "%b %d %Y", "%B %d %Y", "%d %b %y", "%Y/%m/%d",
]


def clean_number(value):
    """Pull a plain number out of text like $1,425,000 or 650 sqm."""
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    digits = re.sub(r"[^0-9.]", "", text)
    if digits in ("", "."):
        return ""
    try:
        number = float(digits)
    except ValueError:
        return ""
    return str(int(number)) if number.is_integer() else str(number)


def clean_date(value):
    """Read a date written in any of the common listing formats."""
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text == "":
        return ""
    text = re.sub(r"^(sold|sold on|settled)\s+", "", text, flags=re.I).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def clean_choice(value, synonyms, allowed):
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    if text == "":
        return ""
    if text in synonyms:
        return synonyms[text]
    for option in allowed:
        if option.lower() == text:
            return option
    return ""


def clean_suburb(value):
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    for suburb in SUBURBS:
        if suburb.lower() == text.lower():
            return suburb
    return ""


def clean_text(value, limit=4000):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()[:limit]


def clean_features(value):
    text = clean_text(value)
    if text == "":
        return ""
    parts = [p.strip() for p in re.split(r"[;,|]", text) if p.strip()]
    return ";".join(parts)


def normalise(raw):
    """Turn one parsed row into schema shaped values."""
    return {
        "listing_url": clean_text(raw.get("listing_url"), 500),
        "address": clean_text(raw.get("address"), 200),
        "suburb": clean_suburb(raw.get("suburb")),
        "postcode": clean_number(raw.get("postcode")),
        "property_type": clean_choice(raw.get("property_type"), TYPE_SYNONYMS, PROPERTY_TYPES),
        "bedrooms": clean_number(raw.get("bedrooms")),
        "bathrooms": clean_number(raw.get("bathrooms")),
        "parking_spaces": clean_number(raw.get("parking_spaces")),
        "land_size_sqm": clean_number(raw.get("land_size_sqm")),
        "sale_price": clean_number(raw.get("sale_price")),
        "sale_date": clean_date(raw.get("sale_date")),
        "sale_method": clean_choice(raw.get("sale_method"), METHOD_SYNONYMS, SALE_METHODS),
        "agent_description": clean_text(raw.get("agent_description")),
        "features_list": clean_features(raw.get("features_list")),
        "date_collected": clean_date(raw.get("date_collected")) or date.today().isoformat(),
    }


def reasons_to_reject(row):
    """A row is only rejected when it cannot be used at all."""
    problems = []
    url = row["listing_url"]
    if not url:
        problems.append("no listing url")
    elif not any(domain in url.lower() for domain in ALLOWED_SOURCE_DOMAINS):
        problems.append("listing url is not from one of the two sites")
    if not row["suburb"]:
        problems.append("suburb is missing or not one of the three being collected")
    if not row["sale_price"]:
        problems.append("no usable sale price")
    if not row["sale_date"]:
        problems.append("no usable sale date")
    if not row["property_type"]:
        problems.append("property type not recognised")
    return problems


def read_existing():
    if not os.path.exists(DATASET):
        return [], set()
    with open(DATASET, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen = {r.get("listing_url", "").strip().lower() for r in rows}
    return rows, seen


def parse_inbox(text):
    """Read pasted rows, with or without a header line."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # Drop a header line if the extension included one
    if lines[0].lower().replace(" ", "").startswith("listing_url,"):
        lines = lines[1:]

    parsed = []
    reader = csv.reader(io.StringIO("\n".join(lines)))
    for fields in reader:
        if not any(f.strip() for f in fields):
            continue
        # Pad or trim so a short row still lines up with the columns
        fields = (fields + [""] * len(COLUMN_NAMES))[:len(COLUMN_NAMES)]
        parsed.append(dict(zip(COLUMN_NAMES, fields)))
    return parsed


def main():
    if not os.path.exists(INBOX):
        print(f"Nothing to read. Paste the extension output into {INBOX} first.")
        return 2

    with open(INBOX, encoding="utf-8") as f:
        text = f.read()

    raw_rows = parse_inbox(text)
    if not raw_rows:
        print(f"{INBOX} has no rows in it.")
        return 2

    existing, seen_urls = read_existing()

    added, duplicates, rejected = [], 0, []
    for raw in raw_rows:
        row = normalise(raw)
        problems = reasons_to_reject(row)
        if problems:
            rejected.append((row.get("address") or row.get("listing_url") or "unnamed row", problems))
            continue
        key = row["listing_url"].strip().lower()
        if key in seen_urls:
            duplicates += 1
            continue
        seen_urls.add(key)
        added.append(row)

    if added:
        write_header = not os.path.exists(DATASET)
        os.makedirs(os.path.dirname(DATASET), exist_ok=True)
        with open(DATASET, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMN_NAMES)
            if write_header:
                writer.writeheader()
            writer.writerows(added)

    print(f"Read {len(raw_rows)} rows from {INBOX}")
    print(f"  added      {len(added)}")
    print(f"  duplicate  {duplicates}")
    print(f"  rejected   {len(rejected)}")

    if rejected:
        print()
        print("Rejected rows, these were not added")
        for name, problems in rejected[:15]:
            print(f"  {name}")
            for problem in problems:
                print(f"      {problem}")

    total = len(existing) + len(added)
    print()
    print(f"{DATASET} now holds {total} properties")

    counts = {}
    for row in existing + added:
        counts[row.get("suburb", "")] = counts.get(row.get("suburb", ""), 0) + 1
    for suburb in SUBURBS:
        print(f"  {suburb:<12} {counts.get(suburb, 0)}")

    print()
    print("Next, clear the inbox and run the validator")
    print(f"  : > {INBOX}")
    print(f"  python3 scripts/validate_dataset.py {DATASET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
