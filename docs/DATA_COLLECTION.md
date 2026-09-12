# Collecting the dataset

This project needs at least 100 sold properties, with at least 30 from
each of the three suburbs. The data is collected by hand from public
sold listings, which is what the task sheet asks for. The Claude browser
extension is used to help read each page and write the row, so the
typing is faster, but you are still the one browsing the sites in a
normal browser session at normal speed.

## Why it is done this way

The task sheet says to manually construct the dataset from
realestate.com.au or domain.com.au. Both sites prohibit automated
scraping in their terms of use, and running a crawler against them would
breach that, so no crawler is used here. Reading pages you have opened
yourself and transcribing what is visible is a different thing, and it
matches the method the task sheet describes.

Record what the listing actually shows. If a listing does not publish a
figure, leave that cell blank. Do not estimate a value and record it as
though it came from the page, because the whole point of the provenance
columns is that every number can be traced back to its source.

## The suburbs

Set in `scripts/collection_schema.py`, change that file to study
somewhere else.

| Suburb | Postcode | Market |
|---|---|---|
| Mosman | 2088 | Premium harbourside, high land value, older larger homes |
| Parramatta | 2150 | Middle ring, many newer apartments, strong transport links |
| Liverpool | 2170 | Outer growth corridor, newer housing, larger blocks |

These are three Sydney suburbs with clearly different housing markets,
which is what the task sheet asks for.

## Where to browse

Sold listings, one page per suburb. Work through the result pages.

**realestate.com.au**
```
https://www.realestate.com.au/sold/in-mosman,+nsw+2088/list-1
https://www.realestate.com.au/sold/in-parramatta,+nsw+2150/list-1
https://www.realestate.com.au/sold/in-liverpool,+nsw+2170/list-1
```

**domain.com.au**
```
https://www.domain.com.au/sold-listings/mosman-nsw-2088/
https://www.domain.com.au/sold-listings/parramatta-nsw-2150/
https://www.domain.com.au/sold-listings/liverpool-nsw-2170/
```

Skip any listing where the price is not disclosed, since sale price is
the value being predicted and a row without it is unusable.

Try to collect a mix of property types rather than only houses, because
the model needs to see units and townhouses to learn anything about
them.

## The prompt for the Claude browser extension

Open a suburb's sold listings page, then give the extension this. It
reads the listings that are on screen and returns rows you can paste
straight into the CSV.

```
Read the sold property listings visible on this page. For each listing,
give me one CSV row with exactly these columns in this order:

listing_url,address,suburb,postcode,property_type,bedrooms,bathrooms,parking_spaces,land_size_sqm,sale_price,sale_date,sale_method,agent_description,features_list,date_collected

Rules:
- Output only the CSV rows, no header, no commentary.
- Wrap every field in double quotes so commas inside text are safe.
- sale_price as digits only, no dollar sign, no commas. Example, 1425000
- sale_date and date_collected as YYYY-MM-DD.
- property_type must be one of House, Unit, Apartment, Townhouse, Villa.
- sale_method must be Auction, Private treaty, or Unknown.
- features_list separated by semicolons. Example, Pool;Air conditioning
- agent_description is the listing description, on one line.
- Leave a field empty if the listing does not show it. Never guess a
  value, an empty cell is correct and a made up number is not.
- Skip any listing with no disclosed sold price.
- date_collected is today's date.
```

If the extension cannot see a field, that field comes back empty, which
is the correct outcome. The validator reports how complete each column
is, and missing data is a normal thing to discuss in the report.

## Saving what you collect

Start from the template, which already has the right header row.

```
cp data/raw/collected_listings_template.csv data/raw/collected_listings.csv
```

Paste the rows under the header as you go.

## Checking as you go

Run this whenever you like, including on a half finished file.

```
python3 scripts/validate_dataset.py data/raw/collected_listings.csv
```

It reports errors, which are rows that cannot be used, and warnings,
which are things worth a look. It also shows how many rows each suburb
has and how complete each column is, so you can see what is left to do.
It passes only when the file meets the task sheet quotas and every value
is in range.

Things it will catch: a price that is not a number, a property type
outside the allowed list, a sale date in the future, a listing URL that
is not from one of the two sites, the same listing collected twice, and
any suburb still short of 30 properties.

## When the file passes

Tell me and I will run the modelling pipeline against the real data,
which means regenerating the feature engineering, the three models, the
cross validation, the error analysis and the saved model behind the web
app. Several things in the current notebook are written around the old
dataset and will be rewritten to match whatever the real data shows.

## What gets derived later, and why

These are not collected because a listing does not publish them. They
are worked out in code from the columns above, so it is always clear
which numbers were observed and which were calculated.

| Derived | How |
|---|---|
| `distance_to_cbd_km` | Suburb centroid distance held in `scripts/collection_schema.py` |
| `has_pool` | Whether `features_list` or `agent_description` mentions a pool |
| `mentions_view` | Whether `agent_description` mentions a view |
| `renovated` | Whether `agent_description` mentions renovation |
| `description_length` | Character count of `agent_description` |
| `price_per_sqm` | `sale_price` divided by `land_size_sqm`, for analysis only |
| `days_since_sale` | Difference between `sale_date` and `date_collected` |
