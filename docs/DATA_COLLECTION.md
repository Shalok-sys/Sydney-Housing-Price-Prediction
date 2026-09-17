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

Do not paste into the CSV by hand. Paste into the inbox file instead and
let the ingest script tidy it up.

```
1. Paste the extension output into data/raw/inbox.txt
2. python3 scripts/ingest_listings.py
3. : > data/raw/inbox.txt
```

The ingest script cleans up whatever shape the extension gives back. It
turns `$4,250,000` into `4250000`, reads `14/03/2026` and `2 May 2026`
and `20 June 2026` all as proper dates, strips `sqm` and `m²` off land
sizes, maps `apartment / unit / flat` to `Apartment` and `private sale`
to `Private treaty`, and rewrites feature lists to use semicolons.

It also protects the file. Rows are only ever appended, never
overwritten. A listing already collected is skipped, so pasting the same
batch twice is harmless. Rows that cannot be used, such as a listing
with no disclosed price or one from a suburb outside the three, are
rejected and reported with the reason rather than being silently added.

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

## Second pass, detail from individual listing pages

The results pages show summary cards only. The description, the feature
tags, the internal floor area and sometimes the build year sit on each
property's own page. Collecting those adds several features and claims
the text data credit the task sheet encourages.

This is an update to listings already collected, not new listings, so it
uses a different script. `ingest_listings.py` treats a url it has seen
before as a duplicate and skips it, which is right for new properties
and wrong here.

```
1. Paste the detail rows into data/raw/inbox.txt
2. python3 scripts/enrich_listings.py
3. : > data/raw/inbox.txt
```

`enrich_listings.py` matches each row to a stored listing by its url and
fills only the empty cells. It never replaces a value that is already
there, so running it twice is harmless and a half finished pass cannot
damage the first one. It reports how many cells it filled and the
coverage of each enrichment column across the whole dataset.

Open a property from the search results, give the extension the detail
prompt, then go back and open the next one. There is no need to do all
of them. Coverage of 40 percent is the point at which a feature starts
being modelled, so roughly 90 of the 231 listings is the target worth
aiming at.

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
