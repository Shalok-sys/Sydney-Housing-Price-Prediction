"""
Adds detail from individual listing pages to listings already collected.

The results pages used in the first pass show only summary cards, so the
description, the feature tags and the internal floor area were never
captured. Those sit on each property's own page. This script takes a
second batch of rows collected from those pages and merges them into the
rows already stored, matching on the listing url.

It only ever fills empty cells. A value already in the dataset is never
replaced, so re running this is safe and a partial second pass cannot
damage what the first pass collected.

Use this rather than ingest_listings.py for a detail pass. Ingest treats
a url it has already seen as a duplicate and skips it, which is correct
for new listings and wrong for extra detail about old ones.

Usage
    1. Paste the detail rows into data/raw/inbox.txt
    2. python3 scripts/enrich_listings.py
    3. : > data/raw/inbox.txt
"""

import csv
import os
import sys

import pandas as pd

from collection_schema import COLUMN_NAMES
from ingest_listings import normalise, parse_inbox

INBOX = "data/raw/inbox.txt"
DATASET = "data/raw/collected_listings.csv"

# The url identifies the property, and the collection date describes
# this pass rather than the property, so neither is ever filled in
NEVER_FILL = ["listing_url", "date_collected"]

FILLABLE = [name for name in COLUMN_NAMES if name not in NEVER_FILL]


def is_empty(value):
    return value is None or str(value).strip() == "" or str(value).lower() == "nan"


def main():
    if not os.path.exists(DATASET):
        print(f"No collected dataset at {DATASET}. Run the first pass before enriching.")
        return 2

    if not os.path.exists(INBOX):
        print(f"Nothing to read. Paste the detail rows into {INBOX} first.")
        return 2

    with open(INBOX, encoding="utf-8") as f:
        raw_rows = parse_inbox(f.read())

    if not raw_rows:
        print(f"{INBOX} has no rows in it.")
        return 2

    stored = pd.read_csv(DATASET, dtype=str).fillna("")
    for name in COLUMN_NAMES:
        if name not in stored.columns:
            stored[name] = ""

    by_url = {url.strip().lower(): index for index, url in stored["listing_url"].items()}

    filled_counts = {name: 0 for name in FILLABLE}
    matched, unmatched, already_complete = 0, [], 0

    for raw in raw_rows:
        row = normalise(raw)
        key = row["listing_url"].strip().lower()
        if key not in by_url:
            unmatched.append(row.get("address") or row.get("listing_url") or "unnamed row")
            continue

        index = by_url[key]
        matched += 1
        filled_any = False

        for name in FILLABLE:
            new_value = row.get(name)
            if is_empty(new_value):
                continue
            if not is_empty(stored.at[index, name]):
                continue
            stored.at[index, name] = new_value
            filled_counts[name] += 1
            filled_any = True

        if not filled_any:
            already_complete += 1

    stored = stored[COLUMN_NAMES]
    stored.to_csv(DATASET, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Read {len(raw_rows)} detail rows from {INBOX}")
    print(f"  matched an existing listing   {matched}")
    print(f"  nothing new to add            {already_complete}")
    print(f"  no matching listing url       {len(unmatched)}")

    if unmatched:
        print()
        print("These did not match anything already collected. If they are new")
        print("properties rather than detail, run ingest_listings.py instead.")
        for name in unmatched[:10]:
            print(f"  {name}")

    print()
    print("Cells filled by column")
    for name in FILLABLE:
        if filled_counts[name]:
            print(f"  {name:<20} {filled_counts[name]}")

    print()
    print("Coverage across the whole dataset now")
    for name in ["agent_description", "features_list", "floor_area_sqm", "year_built"]:
        present = (~stored[name].apply(is_empty)).mean() * 100
        print(f"  {name:<20} {present:5.1f} percent")

    print()
    print("Next, clear the inbox and rebuild")
    print(f"  : > {INBOX}")
    print("  python3 scripts/prepare_features.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
