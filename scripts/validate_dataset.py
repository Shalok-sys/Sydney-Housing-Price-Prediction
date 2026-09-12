"""
Checks a collected listings CSV before it is used for modelling.

Run it as often as you like while collecting, it is safe to run on a
half finished file. It reports two kinds of problem. Errors mean a row
cannot be used as it stands. Warnings mean the file is usable but
something is worth a look.

Usage
    python3 scripts/validate_dataset.py data/raw/collected_listings.csv
"""

import sys
from datetime import date, datetime

import pandas as pd

from collection_schema import (
    ALLOWED_SOURCE_DOMAINS,
    COLUMNS,
    COLUMN_NAMES,
    MIN_ROWS_PER_SUBURB,
    MIN_ROWS_TOTAL,
    REQUIRED_COLUMNS,
    SUBURBS,
    column_by_name,
)

errors = []
warnings = []


def add_error(message):
    errors.append(message)


def add_warning(message):
    warnings.append(message)


def check_columns(df):
    missing = [name for name in COLUMN_NAMES if name not in df.columns]
    if missing:
        add_error(f"Missing columns, {', '.join(missing)}")

    extra = [name for name in df.columns if name not in COLUMN_NAMES]
    if extra:
        add_warning(f"Columns not in the schema, {', '.join(extra)}")


def check_required_values(df):
    for name in REQUIRED_COLUMNS:
        if name not in df.columns:
            continue
        blank = df[df[name].isna() | (df[name].astype(str).str.strip() == "")]
        if len(blank) > 0:
            rows = ", ".join(str(i + 2) for i in blank.index[:8])
            add_error(f"{name} is required but blank on {len(blank)} rows, CSV lines {rows}")


def check_numbers(df):
    for column in COLUMNS:
        name = column["name"]
        if name not in df.columns or column["type"] not in ("integer", "number"):
            continue

        values = pd.to_numeric(df[name], errors="coerce")
        bad_format = df[values.isna() & df[name].notna() & (df[name].astype(str).str.strip() != "")]
        if len(bad_format) > 0:
            rows = ", ".join(str(i + 2) for i in bad_format.index[:8])
            add_error(f"{name} has values that are not numbers on {len(bad_format)} rows, CSV lines {rows}")

        low, high = column.get("min"), column.get("max")
        if low is not None and high is not None:
            out_of_range = df[(values < low) | (values > high)]
            if len(out_of_range) > 0:
                rows = ", ".join(str(i + 2) for i in out_of_range.index[:8])
                add_error(
                    f"{name} should be between {low:g} and {high:g}, "
                    f"{len(out_of_range)} rows are outside that, CSV lines {rows}"
                )


def check_categories(df):
    for column in COLUMNS:
        name = column["name"]
        if name not in df.columns or "allowed" not in column:
            continue
        present = df[name].dropna().astype(str).str.strip()
        present = present[present != ""]
        unknown = sorted(set(present) - set(column["allowed"]))
        if unknown:
            add_error(
                f"{name} has values outside the allowed list, {', '.join(unknown[:8])}. "
                f"Allowed values are {', '.join(column['allowed'])}"
            )


def check_dates(df):
    today = date.today()
    for name in ["sale_date", "date_collected"]:
        if name not in df.columns:
            continue
        parsed = pd.to_datetime(df[name], errors="coerce")
        bad = df[parsed.isna() & df[name].notna() & (df[name].astype(str).str.strip() != "")]
        if len(bad) > 0:
            rows = ", ".join(str(i + 2) for i in bad.index[:8])
            add_error(f"{name} could not be read as a date on {len(bad)} rows, CSV lines {rows}")

        future = df[parsed.dt.date > today]
        if len(future) > 0:
            rows = ", ".join(str(i + 2) for i in future.index[:8])
            add_error(f"{name} is in the future on {len(future)} rows, CSV lines {rows}")

    if "sale_date" in df.columns:
        sale = pd.to_datetime(df["sale_date"], errors="coerce")
        span_days = (sale.max() - sale.min()).days if sale.notna().any() else 0
        if span_days > 365 * 3:
            add_warning(
                f"Sale dates span {span_days} days. A wide window mixes different "
                "market conditions, which is worth mentioning in the report."
            )


def check_sources(df):
    if "listing_url" not in df.columns:
        return
    urls = df["listing_url"].dropna().astype(str)
    off_site = urls[~urls.str.contains("|".join(ALLOWED_SOURCE_DOMAINS), case=False, na=False)]
    if len(off_site) > 0:
        add_error(
            f"{len(off_site)} rows have a listing_url that is not from "
            f"{' or '.join(ALLOWED_SOURCE_DOMAINS)}"
        )

    duplicate_urls = urls[urls.duplicated()]
    if len(duplicate_urls) > 0:
        add_error(f"{len(duplicate_urls)} rows repeat a listing_url that is already in the file")


def check_duplicates(df):
    if "address" not in df.columns:
        return
    key = df["address"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    repeated = key[key.duplicated() & (key != "")]
    if len(repeated) > 0:
        add_warning(
            f"{len(repeated)} rows share an address with another row. "
            "Check these are genuinely different sales."
        )


def check_quotas(df):
    total = len(df)
    if total < MIN_ROWS_TOTAL:
        add_error(f"The task sheet asks for at least {MIN_ROWS_TOTAL} properties, the file has {total}")

    if "suburb" not in df.columns:
        return
    counts = df["suburb"].value_counts()
    for suburb in SUBURBS:
        count = int(counts.get(suburb, 0))
        if count < MIN_ROWS_PER_SUBURB:
            add_error(
                f"{suburb} has {count} properties, the task sheet asks for "
                f"at least {MIN_ROWS_PER_SUBURB} per suburb"
            )


def report_coverage(df):
    print("Rows per suburb")
    if "suburb" in df.columns:
        counts = df["suburb"].value_counts()
        for suburb in SUBURBS:
            count = int(counts.get(suburb, 0))
            mark = "ok " if count >= MIN_ROWS_PER_SUBURB else "low"
            print(f"  {mark}  {suburb:<12} {count}")
    print()

    print("Completeness per column")
    for name in COLUMN_NAMES:
        if name not in df.columns:
            print(f"  missing column  {name}")
            continue
        filled = df[name].notna() & (df[name].astype(str).str.strip() != "")
        pct = filled.mean() * 100 if len(df) else 0
        print(f"  {pct:5.1f} percent  {name}")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    path = sys.argv[1]
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=True)
    except FileNotFoundError:
        print(f"Could not find {path}")
        return 2
    except pd.errors.EmptyDataError:
        print(f"{path} is empty. Start from the template in data/raw.")
        return 2

    print(f"Checking {path}")
    print(f"Found {len(df)} rows and {len(df.columns)} columns")
    print()

    check_columns(df)
    check_required_values(df)
    check_numbers(df)
    check_categories(df)
    check_dates(df)
    check_sources(df)
    check_duplicates(df)
    check_quotas(df)

    report_coverage(df)

    if warnings:
        print(f"Warnings, {len(warnings)}")
        for message in warnings:
            print(f"  {message}")
        print()

    if errors:
        print(f"Errors, {len(errors)}")
        for message in errors:
            print(f"  {message}")
        print()
        print("Fix the errors above, then run this again.")
        return 1

    print("All checks passed. This file is ready for the modelling pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
