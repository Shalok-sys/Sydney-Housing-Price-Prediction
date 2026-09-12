# Sydney Housing Price Prediction and Decision Support System

A machine learning mini project that predicts Sydney property sale
prices for three suburbs, Mosman, Parramatta and Liverpool, and gives
those predictions through a small web application.

## Project layout

```
data/raw/            the collected listings, as transcribed
data/processed/      the cleaned modelling table
notebooks/           the main analysis notebook, parts 1 to 5
scripts/             collection schema, ingest, validation, cleaning, notebook build
models/              the saved model, its feature list and its error figures
app/                 the Flask web application
app/templates/       the page markup
app/static/          the stylesheet and the form script
app/screenshots/     screenshots of the running app
requirements.txt     python packages needed to run everything
```

## The dataset

231 real sold listings collected by hand from realestate.com.au and
domain.com.au, the two sites named in the task sheet, using the Claude
browser extension to read each results page and transcribe the visible
fields. No crawler was run against either site. Both prohibit automated
scraping in their terms of use, and the task sheet asks for the dataset
to be constructed manually.

Every row carries the URL of the listing it came from and the date it
was collected, so any figure in the analysis can be traced back to its
source. Only fields the listings actually displayed were recorded, and
empty cells mean the listing did not publish that value rather than that
a value was estimated.

After cleaning, 228 properties are used for modelling. See
`docs/DATA_COLLECTION.md` for how to collect more, and the notebook for
what was removed and why.

Two things to know about this data. It is about 85 percent apartments
and units, which is what actually sold in these suburbs over the seven
month window, so the model is strong on strata and weak on houses.
Parramatta returned no house sales at all, and the app says so rather
than quietly guessing.

## Acknowledgement of GenAI use

Claude, an AI coding assistant, was used to help plan this project and
write the collection tooling, notebook, model code and Flask app.
Every number and result quoted in the notebook comes from actually
running the code, nothing was written in first and made up. Review,
understand and where useful extend this work in your own words before
submitting, and repeat this acknowledgement in the written report as
required by the task instructions.

## Setup

1. Create and activate a virtual environment, optional but recommended.
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the packages.
   ```
   pip install -r requirements.txt
   ```

## Rebuilding the dataset and notebook

The dataset and notebook are already included in this repository, so
this step is only needed after collecting more listings.

```
python3 scripts/ingest_listings.py
python3 scripts/validate_dataset.py data/raw/collected_listings.csv
python3 scripts/prepare_features.py
python3 scripts/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/sydney_housing_price_prediction.ipynb
```

Running the notebook also retrains and saves the final model to
`models/best_model.joblib` with its feature list in
`models/feature_schema.json`, which the app below needs.

## Running the notebook directly

```
jupyter notebook notebooks/sydney_housing_price_prediction.ipynb
```

## Running the web application

```
python3 app/flask_app.py
```

Then open `http://127.0.0.1:5000` in a browser.

How to use it. Pick the suburb and property type, fill in the property
details, and press Estimate price. The result panel shows the estimated
sale price together with a likely range. That range is not a guess, it
comes from how far the model is typically off in that suburb, measured
by cross validation, so the estimate is never presented as a single
exact figure. The agent description box is optional, the model uses its
length and whether it mentions a view.

Two details worth knowing. Choosing Unit switches the land size input
off, because units have no land of their own and the model was trained
that way. Values submitted from the form are validated on the server,
so out of range or missing entries return a clear message rather than a
broken estimate.

Screenshots are in `app/screenshots/`, covering the empty form, a
completed estimate, the same estimate in dark mode, and the mobile
layout.

## Reproducing results

The models use a fixed random seed, so rerunning the commands above on
the same collected data gives the same numbers shown in the notebook.
