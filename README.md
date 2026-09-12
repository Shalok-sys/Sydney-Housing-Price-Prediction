# Sydney Housing Price Prediction and Decision Support System

A machine learning mini project that predicts Sydney property sale
prices for three suburbs, Mosman, Parramatta and Liverpool, and gives
those predictions through a small web application.

## Project layout

```
data/                the housing dataset used in the notebook
notebooks/           the main analysis notebook, parts 1 to 5
scripts/             the scripts used to build the dataset and notebook
models/              the saved model, its feature list and its error figures
app/                 the Flask web application
app/templates/       the page markup
app/static/          the stylesheet and the form script
app/screenshots/     screenshots of the running app
requirements.txt     python packages needed to run everything
```

## Note on the dataset

The task asks for sold listings collected manually from a site such as
realestate.com.au or domain.com.au. Automated scraping of those sites
was not reliable in this environment because of bot protection and
site terms of use, so `scripts/generate_dataset.py` builds a simulated
dataset instead. It uses a sale price formula calibrated to public
median price ranges for each suburb, then adds realistic noise,
missing values and a handful of unusual sales. This is clearly stated
again inside the notebook. Before final submission, check whether the
unit requires genuine manually collected listings for full marks on
the data collection criterion, and replace or supplement this dataset
if needed.

## Acknowledgement of GenAI use

Claude, an AI coding assistant, was used to help plan this project and
write the dataset generator, notebook, model code and Flask app.
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
this step is only needed if you want to regenerate them.

```
python3 scripts/generate_dataset.py
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

All random steps in the dataset generator and the models use a fixed
random seed, so rerunning the commands above should give the same
numbers shown in the notebook.
